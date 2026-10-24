import heapq
import json
import difflib
import os
import time

import numpy as np
import collections
from prompt import (
    entity_extract_prompt,
    _build_answer_prompt,
    _build_forceful_answer_prompt,
)
from sklearn.metrics.pairwise import cosine_similarity
import re



# Scale-control parameters for edge-enabled retrieval, defined in one place so
# ablation experiments can tune them uniformly.
MIN_TRAVERSAL_EDGE_SIMILARITY = 0.25
MAX_PATH_HOPS = 4
MAX_PATH_COST = 2.40


# Edge texts depend only on the graph cache and the embedding model, so they
# must not be re-encoded for every QA query.
_EDGE_EMBEDDING_CACHE = {}

def calculate_similarity(emb1, emb2_list):
    """Cosine similarity between one vector and a collection of vectors."""
    if len(emb2_list) == 0:
        return []
    return cosine_similarity([emb1], emb2_list)[0]



def _dedupe_preserve_order(items):
    seen = set()
    ordered_items = []

    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered_items.append(text)

    return ordered_items


def _normalize_entity_name(value):
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _entity_acronym(value):
    tokens = _normalize_entity_name(value).split()
    # Keep "of" so that Department of Defense -> DOD; only articles and
    # conjunctions are ignored.
    ignored = {"a", "an", "the", "for", "and", "to", "in"}
    letters = [token[0] for token in tokens if token and token not in ignored]
    return "".join(letters) if len(letters) >= 2 else ""


def _entity_name_match_quality(query_entity, graph_entity):
    """Entity-name linking only; this never compares a query with an E summary."""
    query_name = _normalize_entity_name(query_entity)
    graph_name = _normalize_entity_name(graph_entity)
    if not query_name or not graph_name:
        return 0.0, ""
    if query_name == graph_name:
        return 1.0, "exact"

    query_acronym = _entity_acronym(query_name)
    graph_acronym = _entity_acronym(graph_name)
    if (
        (query_name.replace(" ", "") == graph_acronym and graph_acronym)
        or (graph_name.replace(" ", "") == query_acronym and query_acronym)
    ):
        return 0.96, "alias"

    query_tokens = set(query_name.split())
    graph_tokens = set(graph_name.split())
    shorter = min(len(query_tokens), len(graph_tokens))
    if shorter >= 2 and (
        query_tokens.issubset(graph_tokens)
        or graph_tokens.issubset(query_tokens)
    ):
        return 0.90, "lexical"

    ratio = difflib.SequenceMatcher(None, query_name, graph_name).ratio()
    if ratio >= 0.90:
        return 0.88, "lexical"
    return 0.0, ""


def _is_hard_entity_reason(reason):
    reason = str(reason or "")
    return any(
        marker in reason
        for marker in ("exact", "alias", "lexical")
    )



def _score_a_propositions(propositions, proposition_embeddings, query_embedding,
                          focus_embedding, query_entities, structural_entities):
    """Batch-score non-center A evidence inside one selected D ball."""
    if not propositions:
        return []

    semantic_scores = calculate_similarity(query_embedding, proposition_embeddings)
    focus_scores = calculate_similarity(focus_embedding, proposition_embeddings)
    normalized_query_entities = {
        _normalize_entity_name(entity) for entity in query_entities
        if _normalize_entity_name(entity)
    }
    normalized_structural_entities = {
        _normalize_entity_name(entity) for entity in structural_entities
        if _normalize_entity_name(entity)
    }

    results = []
    for proposition, semantic_score, focus_score in zip(
            propositions, semantic_scores, focus_scores):
        proposition_entities = {
            _normalize_entity_name(entity)
            for entity in proposition.get("entity_list", [])
            if _normalize_entity_name(entity)
        }
        entity_score = min(
            1.0,
            len(proposition_entities & normalized_query_entities)
            / max(1, len(normalized_query_entities)),
        )
        structural_score = 1.0 if (
            proposition_entities & normalized_structural_entities
        ) else 0.0
        total = (
            0.50 * float(semantic_score)
            + 0.20 * entity_score
            + 0.20 * float(focus_score)
            + 0.10 * structural_score
        )
        results.append((total, {
            "semantic": round(float(semantic_score), 4),
            "entity": round(entity_score, 4),
            "query_focus": round(float(focus_score), 4),
            "structural": round(structural_score, 4),
        }))
    return results

def _build_semantic_fallback_context(query_emb, e_balls_dict, Propositions, db_helper, preferred_e_ids=None,
                                     top_e=4, top_a=10):
    context_sections = []

    candidate_e_ids = []
    if preferred_e_ids:
        candidate_e_ids.extend(preferred_e_ids)
    if db_helper is not None:
        try:
            candidate_e_ids.extend(db_helper.search_e_level(query_emb, top_k=top_e))
        except Exception as e:
            print(f"E-level vector fallback search failed: {e}")

    candidate_e_ids = _dedupe_preserve_order(candidate_e_ids)
    c_lines = []
    for e_idx in candidate_e_ids:
        e_ball = e_balls_dict.get(e_idx)
        if not e_ball:
            continue
        summary = str(e_ball.get("summary", "")).strip()
        if summary:
            c_lines.append(f"{e_idx}: {summary}")

    if c_lines:
        context_sections.append(
            "Relevant mid-level clusters:\n" + "\n".join(f"- {line}" for line in c_lines)
        )

    a_lines = []
    if db_helper is not None:
        try:
            for a_idx in db_helper.search_a_level(query_emb, top_k=top_a):
                try:
                    prop_idx = int(str(a_idx).split("_")[-1])
                except (TypeError, ValueError):
                    continue
                if 0 <= prop_idx < len(Propositions):
                    proposition = str(Propositions[prop_idx].get("proposition", "")).strip()
                    if proposition:
                        a_lines.append(proposition)
        except Exception as e:
            print(f"A-level vector fallback search failed: {e}")

    a_lines = _dedupe_preserve_order(a_lines)
    if a_lines:
        context_sections.append(
            "Candidate supporting facts:\n" + "\n".join(f"- {line}" for line in a_lines)
        )

    return "\n\n".join(context_sections)

def _clean_answer_text(answer):
    if answer is None:
        return "unknown"

    text = str(answer).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"^(short answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    if not text:
        return "unknown"

    text = text.splitlines()[0].strip()
    if not text:
        return "unknown"

    return text


def _answer_from_context(query, context, llm):
    context = str(context or "").strip()
    if not context:
        return "unknown"

    try:
        answer_prompt = _build_answer_prompt()
        answer = (answer_prompt | llm).invoke({
            "query": query,
            "context": context
        }).content
        cleaned = _clean_answer_text(answer)
        # Optimization 4: if the first attempt returns unknown, retry with the
        # stronger prompt.
        if cleaned.lower() == "unknown":
            forceful_prompt = _build_forceful_answer_prompt()
            answer2 = (forceful_prompt | llm).invoke({
                "query": query,
                "context": context
            }).content
            cleaned2 = _clean_answer_text(answer2)
            if cleaned2.lower() != "unknown":
                return cleaned2
        return cleaned
    except Exception as e:
        print(f"Answer generation from context failed: {e}")
        return "unknown"





def _get_cached_edge_embeddings(edges, embedding_model):
    """Batch-encode unique edge texts once and reuse them within the process."""
    unique_texts = _dedupe_preserve_order(
        edge.get("relation", "") for edge in edges
    )
    cache_key = (id(embedding_model), tuple(unique_texts))
    cached = _EDGE_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not unique_texts:
        result = ([], np.empty((0, 0), dtype=np.float32))
    else:
        embeddings = np.asarray(
            embedding_model.encode(
                unique_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
        result = (unique_texts, embeddings)

    # A process typically handles one graph and one embedding model; clear
    # stale graph caches to avoid unbounded memory growth in multi-dataset runs.
    _EDGE_EMBEDDING_CACHE.clear()
    _EDGE_EMBEDDING_CACHE[cache_key] = result
    return result


def _bounded_shortest_path(
    graph,
    edge_weights,
    edge_similarities,
    start_node,
    target_node,
    max_hops=MAX_PATH_HOPS,
    max_cost=MAX_PATH_COST,
    min_edge_similarity=MIN_TRAVERSAL_EDGE_SIMILARITY,
):
    """Dijkstra with hop-count, accumulated-cost, and minimum edge-relevance constraints."""
    if start_node == target_node:
        return [start_node], 0.0

    start_state = (start_node, 0)
    pq = [(0.0, start_node, 0)]
    best_cost = {start_state: 0.0}
    parent_state = {}
    target_state = None

    while pq:
        current_cost, current_node, hops = heapq.heappop(pq)
        state = (current_node, hops)
        if current_cost > best_cost.get(state, float("inf")):
            continue
        if current_node == target_node:
            target_state = state
            break
        if hops >= max_hops:
            continue

        for neighbor in graph.get(current_node, []):
            edge_similarity = edge_similarities.get(
                (current_node, neighbor), 0.0
            )
            if edge_similarity < min_edge_similarity:
                continue

            next_hops = hops + 1
            next_cost = current_cost + edge_weights.get(
                (current_node, neighbor), 1.0
            )
            if next_cost > max_cost:
                continue

            next_state = (neighbor, next_hops)
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                parent_state[next_state] = state
                heapq.heappush(
                    pq, (next_cost, neighbor, next_hops)
                )

    if target_state is None:
        return None, None

    path = []
    state = target_state
    while True:
        path.append(state[0])
        if state == start_state:
            break
        state = parent_state.get(state)
        if state is None:
            return None, None
    path.reverse()
    return path, best_cost[target_state]


def _minimum_spanning_seed_paths(seed_ids, pair_paths):
    """
    Kruskal over the seed layer. Values of pair_paths are (cost, original_path);
    returns the original graph paths selected into the minimum spanning forest.
    """
    parent = {seed_id: seed_id for seed_id in seed_ids}
    rank = {seed_id: 0 for seed_id in seed_ids}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        return True

    candidates = []
    for (left, right), (cost, path) in pair_paths.items():
        candidates.append((float(cost), str(left), str(right), path))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    selected_paths = []
    for cost, left, right, path in candidates:
        if union(left, right):
            selected_paths.append(path)
            if len(selected_paths) >= max(0, len(seed_ids) - 1):
                break
    return selected_paths



def semantic_set_similarity(set1, set2, model):
    """Cosine similarity between the mean embeddings of two tag sets."""

    # vec1 = np.mean(model.encode(list(set1)), axis=0)
    # vec2 = np.mean(model.encode(list(set2)), axis=0)
    #
    # sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # return sim
    if not set1 or not set2:
        return 0.0

        # Make sure set elements are strings and convert to lists (avoids
        # set-related type issues).
    list1 = [str(item) for item in set1 if item]  # filter out empty elements
    list2 = [str(item) for item in set2 if item]

    # Re-check non-emptiness (guard against all-empty-string sets).
    if not list1 or not list2:
        return 0.0

    # Encode (list1/list2 are now valid non-empty string lists).
    vec1 = np.mean(model.encode(list1), axis=0)
    vec2 = np.mean(model.encode(list2), axis=0)

    # Cosine similarity with divide-by-zero protection.
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return sim








# ============================================================
# P1: D-ball entry + two-stage ("lexical -> semantic") in-ball recall
# + global quantity fallback. Only the retrieval entry and recall method are
# replaced; the hierarchy and edges stay unchanged. All thresholds can be
# overridden by environment variables.
# ============================================================
_P1_B_CENTROID_CACHE = {}




def _get_a_embeddings_idx(Propositions, embedding_model, db_helper):
    """A-level proposition idx(int) -> embedding. Prefer the chroma cache; encode online for the rest."""
    idx_to_emb = {}
    a_col = getattr(db_helper, "a_collection", None) if db_helper is not None else None
    if a_col is not None:
        try:
            data = a_col.get(include=["embeddings"])
            for _id, emb in zip(data.get("ids", []), data.get("embeddings", [])):
                try:
                    idx_to_emb[int(str(_id).split("_")[-1])] = emb
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            print(f"P1: failed to read chroma A-level vectors; missing items will be encoded online: {e}")
    return idx_to_emb


def _get_d_ball_centroids(D_balls, Propositions, embedding_model, db_helper):
    """D-ball centroid = mean of member A-proposition vectors. Cached by id(D_balls) so a 50-question run computes it once."""
    cache_key = id(D_balls)
    if cache_key in _P1_B_CENTROID_CACHE:
        return _P1_B_CENTROID_CACHE[cache_key]
    idx_to_emb = _get_a_embeddings_idx(Propositions, embedding_model, db_helper)
    centroids = {}
    for ball in D_balls:
        embs = []
        for idx in ball.get("all_indices", []):
            try:
                i = int(str(idx).split("_")[-1])
            except (TypeError, ValueError):
                continue
            emb = idx_to_emb.get(i)
            if emb is None and 0 <= i < len(Propositions):
                try:
                    emb = embedding_model.encode([Propositions[i]["proposition"]])[0]
                except Exception:
                    emb = None
            if emb is not None:
                embs.append(emb)
        if embs:
            centroids[ball["idx"]] = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
    _P1_B_CENTROID_CACHE[cache_key] = centroids
    return centroids


# ==================== v18 dual-pool retrieval main pipeline (current framework, no D-level routing) ====================

_CHUNK_CACHE = {}

MAX_CONTEXT_CHARS = 48000  # same budget convention as v5


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _flag(name):
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _chunk_gate(query, embedding_model):
    """Return (set_of_chunk_ids, ranked_ids) by raw-text similarity."""
    path = os.environ.get("GBRAG_RAW_CORPUS_PATH", "")
    if not path or not os.path.exists(path):
        return set(), []
    try:
        if path not in _CHUNK_CACHE:
            with open(path, encoding="utf-8") as f:
                corpus = json.load(f)
            texts = [str(x.get("text", "")) for x in corpus]
            embs = np.asarray(embedding_model.encode(texts, show_progress_bar=False), dtype=np.float32)
            _CHUNK_CACHE[path] = ({int(x.get("chunk_id", i)): t for i, (x, t) in enumerate(zip(corpus, texts))}, embs)
        chunk_map, embs = _CHUNK_CACHE[path]
        q = np.asarray(embedding_model.encode([query])[0], dtype=np.float32)
        sims = calculate_similarity(q, embs)
        k = max(1, _env_int("GBRAG_V17_TOP_K_CHUNKS", 16))
        order = np.argsort(-np.asarray(sims))[:k]
        ids = [list(chunk_map.keys())[int(i)] for i in order]
        return set(ids), ids
    except Exception as exc:
        print(f"v18 raw-chunk gate failed; falling back to global D balls: {exc}")
        return set(), []


def _strict_edge_entity_match(edge, query_entities):
    """Soft entity gate, same semantics as v17."""
    if not query_entities:
        return True
    shared = {str(x).strip().lower() for x in edge.get("shared_entities", []) if str(x).strip()}
    if not shared:
        return False
    for q in query_entities:
        qn = _normalize_entity_name(q.get("name", q) if isinstance(q, dict) else q)
        if not qn:
            continue
        for entity in shared:
            en = _normalize_entity_name(entity)
            if qn == en or (len(qn.split()) >= 2 and (qn in en or en in qn)):
                return True
    return None


def _best_props_for_e(eid, e_by_id, d_map, Propositions, idx_to_emb, query_emb, embedding_model, limit):
    """Top-`limit` (sim, text) A propositions inside E ball `eid` by query similarity."""
    c = e_by_id.get(eid, {})
    candidates = []
    for bid in c.get("d_indices", []):
        ball = d_map.get(bid, {})
        for raw in ball.get("all_indices", []):
            try:
                idx = int(str(raw).split("_")[-1])
                prop = Propositions[idx]
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            text = str(prop.get("proposition", "")).strip()
            if not text:
                continue
            emb = idx_to_emb.get(idx)
            if emb is None:
                emb = embedding_model.encode([prop["proposition"]])[0]
                idx_to_emb[idx] = emb
            sim = float(calculate_similarity(query_emb, [emb])[0])
            candidates.append((sim, text))
    return sorted(candidates, reverse=True)[:limit]


def Graph_Rag_pipeline(query, E_balls, D_balls, edges, Propositions,
                       llm, embedding_model, db_helper, enable_edge_propagation=True,
                       return_trace=False,
                       experiment_config=None):
    started = time.perf_counter()
    no_chunk_gate = _flag("GBRAG_V18_NO_CHUNK_GATE")
    no_entity_c = _flag("GBRAG_V18_NO_ENTITY_C")
    no_domain_bonus = _flag("GBRAG_V18_NO_DOMAIN_BONUS")
    no_path = _flag("GBRAG_V18_NO_PATH")
    no_fallback = _flag("GBRAG_V18_NO_FALLBACK")
    no_supp = _flag("GBRAG_V18_NO_SUPP")
    supp_only = _flag("GBRAG_V18_SUPP_ONLY")
    # Low-budget ablations: GBRAG_V18_NO_BRIDGE=1 disables bridge propositions;
    # GBRAG_V18_NO_C_SUMMARY=1 disables E-level summaries.
    no_bridge = _flag("GBRAG_V18_NO_BRIDGE")
    no_c_summary = _flag("GBRAG_V18_NO_C_SUMMARY")

    query_emb = embedding_model.encode([query])[0]

    # --- FLAT baseline (for experiments): plain proposition retrieval ---
    # GBRAG_V18_FLAT=1 skips the ball gating / D-ball recall / path activation /
    # dual-pool assembly and directly takes the semantic top-K over all A-level
    # propositions (K defaults to 32, aligned with the average v18 context
    # proposition count), to quantify the increment of "balls + paths" over
    # flat proposition retrieval.
    if _flag("GBRAG_V18_FLAT"):
        flat_top_k = _env_int("GBRAG_V18_FLAT_TOP_K", 32)
        idx_to_emb_flat = _get_a_embeddings_idx(Propositions, embedding_model, db_helper)
        flat_sims = calculate_similarity(query_emb, list(idx_to_emb_flat.values()))
        flat_ranked = sorted(zip(flat_sims, idx_to_emb_flat.keys()), reverse=True)
        flat_facts = []
        for _, idx in flat_ranked[:flat_top_k]:
            try:
                t = str(Propositions[idx]["proposition"]).strip()
            except (TypeError, KeyError, IndexError):
                continue
            if t and t not in flat_facts:
                flat_facts.append(t)
        context = "Candidate supporting facts:\n" + "\n".join(f"- {t}" for t in flat_facts)
        if not context.strip():
            context = "Candidate supporting facts:\n(none)"
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
        answer_started = time.perf_counter()
        answer = _answer_from_context(query, context, llm)
        end = time.perf_counter()
        trace = {
            "retrieved_context": context, "experiment_id": "DCB_v18_dual_pool",
            "p1_top_d": [], "v15_top_chunk_ids": [],
            "v15_gated_d_count": 0, "v15_entity_e_candidate_count": 0,
            "v15_entity_d_bonus_count": 0, "v11_parent_e_seeds": [],
            "selected_paths": [], "context_paths": [],
            "activated_node_type": "flat_proposition", "activated_node_count": len(flat_facts),
            "path_activated": False, "path_count": 0,
            "avg_path_length": 0.0, "max_path_length": 0,
            "activated_relation_count": 0,
            "selected_fact_count": len(flat_facts),
            "v18_supplement_count": 0, "v18_bridge_count": 0, "v18_e_summary_count": 0,
            "v18_middle_added": 0, "v18_entity_added": 0,
            "v18_fallback_used": False,
            "v18_ablation": {"flat": True, "flat_top_k": flat_top_k},
            "retrieval_seconds": round(end - started, 6),
            "answer_generation_seconds": round(end - answer_started, 6),
            "path_search_seconds": round(end - started, 6),
            "v11_path_edge_candidates": 0,
        }
        return (answer, trace) if return_trace else answer

    else:
        gated_chunks, top_chunk_ids = _chunk_gate(query, embedding_model)

    entity_data = {}
    try:
        res = (entity_extract_prompt | llm).invoke({"query": query})
        entity_data = json.loads(str(res.content).replace("```json", "").replace("```", "").strip())
    except Exception as exc:
        print(f"v18 entity extraction failed; continuing with the vector path: {exc}")
    query_entities = entity_data.get("query_entities", []) if isinstance(entity_data, dict) else []
    query_focus = str(entity_data.get("query_focus", query) if isinstance(entity_data, dict) else query).strip()
    focus_emb = embedding_model.encode([query_focus])[0]

    top_d = _env_int("GBRAG_V17_TOP_B", 4)
    score_keep = _env_int("GBRAG_V11_SCORE_KEEP", 5)
    final_keep = _env_int("GBRAG_V11_FINAL_KEEP", 15)
    e_limit = _env_int("GBRAG_V17_MAX_C_SEEDS", 8)
    path_hops = _env_int("GBRAG_V11_MAX_PATH_HOPS", 4)
    middle_facts = _env_int("GBRAG_V18_MIDDLE_FACTS", 3)
    entity_facts = _env_int("GBRAG_V18_ENTITY_C_FACTS", 2)
    min_props = _env_int("GBRAG_P1_MIN_PROPS", 8)
    fallback_min_sim = float(os.environ.get("GBRAG_P1_FALLBACK_MIN_SIM", "0.45"))
    fallback_top_a = _env_int("GBRAG_P1_FALLBACK_TOP_A", 30)
    top_bridge = _env_int("GBRAG_P1_TOP_BRIDGE", 10)
    bridge_min_sim = float(os.environ.get("GBRAG_P1_BRIDGE_MIN_SIM", "0.60"))

    all_centroids = _get_d_ball_centroids(D_balls, Propositions, embedding_model, db_helper)
    d_map = {b["idx"]: b for b in D_balls}
    e_by_id = {c["idx"]: c for c in E_balls}
    prop_by_num = {i: p for i, p in enumerate(Propositions)}

    # --- entity-activated E balls (ablation: NO_ENTITY_C) ---
    entity_e = []
    if not no_entity_c:
        entity_norms = [_normalize_entity_name(x.get("name", x) if isinstance(x, dict) else x) for x in query_entities]
        for c in E_balls:
            c_entities = [_normalize_entity_name(x) for x in c.get("entity_list", [])]
            if entity_norms and any(q and (q == ce or (len(q.split()) >= 2 and (q in ce or ce in q)))
                                    for q in entity_norms for ce in c_entities):
                entity_e.append(c["idx"])
    entity_d = {str(bid) for eid in entity_e for bid in e_by_id.get(eid, {}).get("d_indices", [])}

    # --- Top-J chunk gate over D-ball candidates (soft, with global supplement) ---
    def d_matches_gate(ball):
        if not gated_chunks:
            return True
        for raw in ball.get("all_indices", []):
            try:
                p = prop_by_num[int(str(raw).split("_")[-1])]
                if int(p.get("id")) in gated_chunks:
                    return True
            except (TypeError, ValueError, KeyError, IndexError):
                continue
        return False

    centroids = {bid: emb for bid, emb in all_centroids.items() if d_matches_gate(d_map.get(bid, {}))}
    if len(centroids) < _env_int("GBRAG_V12_MIN_GATED_B", 4):
        centroids = dict(all_centroids)
    else:
        # Recall safeguard: keep a few globally closest D balls as a supplement.
        global_rank = sorted(
            zip(calculate_similarity(query_emb, list(all_centroids.values())), all_centroids.keys()),
            reverse=True,
        )
        for _, bid in global_rank[:_env_int("GBRAG_V16_GLOBAL_B_SUPPLEMENT", 4)]:
            if bid not in centroids:
                centroids[bid] = all_centroids[bid]

    # --- D-ball recall (+ optional entity bonus, ablation: NO_DOMAIN_BONUS) ---
    ranked_pairs = []
    for sim, bid in zip(calculate_similarity(query_emb, list(centroids.values())), centroids.keys()):
        bonus = 0.03 if (not no_domain_bonus and str(bid) in entity_d) else 0.0
        ranked_pairs.append((float(sim) + bonus, bid))
    selected_d = [bid for _, bid in sorted(ranked_pairs, key=lambda x: x[0], reverse=True)[:top_d]]

    # --- in-ball A-proposition scoring (identical to v5/v17) ---
    idx_to_emb = _get_a_embeddings_idx(Propositions, embedding_model, db_helper)
    pooled = []
    for bid in selected_d:
        ball = d_map.get(bid)
        if not ball:
            continue
        props, embs = [], []
        for raw_idx in ball.get("all_indices", []):
            try:
                idx = int(str(raw_idx).split("_")[-1])
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(Propositions):
                emb = idx_to_emb.get(idx)
                if emb is None:
                    emb = embedding_model.encode([Propositions[idx]["proposition"]])[0]
                    idx_to_emb[idx] = emb
                props.append(Propositions[idx]); embs.append(emb)
        scored = _score_a_propositions(props, embs, query_emb, focus_emb, query_entities, structural_entities=query_entities)
        for (score, detail), prop in sorted(zip(scored, props), key=lambda x: x[0][0], reverse=True)[:score_keep]:
            pooled.append((str(prop["proposition"]).strip(), float(score), float(detail.get("semantic", 0.0)), bid))
    pooled_raw_count = len(pooled)
    unique = {}
    for item in pooled:
        unique[item[0]] = max(unique.get(item[0], item), item, key=lambda x: x[1])
    final_pool = sorted(unique.values(), key=lambda x: x[2], reverse=True)[:final_keep]
    max_semantic = max((x[2] for x in final_pool), default=0.0)

    # --- parent E balls ---
    d_to_e = {}
    for c in E_balls:
        for bid in c.get("d_indices", []):
            d_to_e.setdefault(str(bid), c["idx"])
    seed_e = [eid for eid in dict.fromkeys(d_to_e.get(str(b)) for b in selected_d if str(b) in d_to_e) if eid is not None][:e_limit]

    # --- path search (ablation: NO_PATH) ---
    paths, edge_info = [], {}
    min_edge_sim = float(os.environ.get("GBRAG_V17_MIN_EDGE_SIM", "0.35"))
    if not no_path and edges:
        edge_texts, edge_embs = _get_cached_edge_embeddings(edges, embedding_model)
        relation_similarity = {}
        if edge_texts:
            relation_similarity = {t: float(s) for t, s in zip(edge_texts, calculate_similarity(query_emb, edge_embs))}
        graph = collections.defaultdict(set); weights = {}; sims = {}
        for edge in edges:
            u, w = edge.get("node_E_1"), edge.get("node_E_2")
            sim = relation_similarity.get(str(edge.get("relation", "")), 0.0)
            entity_match = _strict_edge_entity_match(edge, query_entities)
            if sim < min_edge_sim or entity_match is False or (entity_match is None and sim < 0.55):
                continue
            graph[u].add(w); graph[w].add(u)
            weight = max(0.1, 1.0 - sim)
            for a, b in ((u, w), (w, u)):
                if sim > sims.get((a, b), -1.0):
                    sims[(a, b)] = sim; weights[(a, b)] = weight; edge_info[(a, b)] = edge.get("relation", "")
        path_started = time.perf_counter()
        pair_paths = {}
        for i, start in enumerate(seed_e):
            for target in seed_e[i + 1:]:
                path, cost = _bounded_shortest_path(graph, weights, sims, start, target, max_hops=path_hops,
                                                        max_cost=float(os.environ.get("GBRAG_V11_MAX_PATH_COST", "2.4")),
                                                        min_edge_similarity=min_edge_sim)
                if path:
                    pair_paths[(start, target)] = (cost, path)
        paths = _minimum_spanning_seed_paths(seed_e, pair_paths)
    else:
        path_started = time.perf_counter()
    path_e = set(seed_e)
    for p in paths:
        path_e.update(p)

    # --- fact pools ---
    # facts from final_pool keyed by parent C
    e_to_facts = collections.defaultdict(list)
    for text, _, _, bid in final_pool:
        eid = d_to_e.get(str(bid))
        if eid is not None:
            e_to_facts[eid].append(text)
    # supplementary middle-C facts: NOT chunk-gated (double gating fix)
    middle_added = []
    for eid in path_e:
        if eid not in e_to_facts:
            facts = [t for _, t in _best_props_for_e(eid, e_by_id, d_map, Propositions, idx_to_emb, query_emb, embedding_model, middle_facts)]
            e_to_facts[eid].extend(facts)
            middle_added.extend(facts)
    # entity-activated E balls outside the path set contribute their best facts
    # too. The entity matcher is intentionally loose (v17 only used it for a
    # +0.03 bonus), so cap how many balls may contribute: GBRAG_V18_ENTITY_C_MAX_BALLS,
    # ranked by each ball's best proposition similarity. 0 = unlimited (v18.0 behavior).
    entity_added = []
    entity_ball_facts = []
    for eid in entity_e:
        if eid in path_e or eid in e_to_facts:
            continue
        scored = _best_props_for_e(eid, e_by_id, d_map, Propositions, idx_to_emb, query_emb, embedding_model, entity_facts)
        if scored:
            entity_ball_facts.append((scored[0][0], eid, scored))
    max_entity_balls = _env_int("GBRAG_V18_ENTITY_C_MAX_BALLS", 0)
    if max_entity_balls > 0:
        entity_ball_facts.sort(key=lambda x: x[0], reverse=True)
        entity_ball_facts = entity_ball_facts[:max_entity_balls]
    for _, eid, scored in entity_ball_facts:
        facts = [t for _, t in scored]
        e_to_facts[eid].extend(facts)
        entity_added.extend(facts)

    emitted = set()
    for facts in e_to_facts.values():
        for t in facts:
            emitted.add(t)

    # --- bridge propositions (v5-style, from activated E balls) ---
    activated_c = set(seed_e) | path_e
    bridge_texts = []
    if edges and activated_c and not no_bridge:
        try:
            cand_edges = [e for e in edges if e.get("node_E_1") in activated_c or e.get("node_E_2") in activated_c]
            relations = [str(e.get("relation", "")).strip() for e in cand_edges]
            rel_embs = embedding_model.encode([r for r in relations if r]) if any(relations) else []
            rel_sims = calculate_similarity(query_emb, list(rel_embs)) if len(rel_embs) else []
            ranked = sorted(zip(rel_sims, [r for r in relations if r]), key=lambda x: x[0], reverse=True)
            for sim, text in ranked:
                if len(bridge_texts) >= top_bridge or sim < bridge_min_sim:
                    break
                if not text or text in bridge_texts or text in emitted:
                    continue
                bridge_texts.append(text)
        except Exception as exc:
            print(f"v18 bridge-proposition collection failed: {exc}")

    # --- E-level summaries (v5-style) ---
    c_lines = []
    if not no_c_summary:
        try:
            for e_idx in db_helper.search_e_level(query_emb, top_k=3):
                e_ball = e_by_id.get(e_idx)
                if e_ball and str(e_ball.get("summary", "")).strip():
                    line = f"{e_idx}: {e_ball['summary']}"
                    if line not in c_lines:
                        c_lines.append(line)
        except Exception as exc:
            print(f"v18 E-level summary supplement failed: {exc}")

    # --- context assembly ---
    fallback_triggered = (pooled_raw_count < min_props) or (max_semantic < fallback_min_sim)
    # Ablation GBRAG_V18_NO_FALLBACK=1: the trigger is still recorded (visible
    # in the trace) but the normal dual-pool assembly is used instead of the
    # global semantic fallback.
    fallback_used = fallback_triggered and not no_fallback
    supplement = []
    if fallback_used:
        context = _build_semantic_fallback_context(
            query_emb, e_by_id, Propositions, db_helper,
            preferred_e_ids=None, top_e=4, top_a=fallback_top_a,
        )
    elif no_path or not paths:
        # flat v5-style fallback of the dual-pool assembly
        sections = []
        if c_lines:
            sections.append("Relevant mid-level clusters:\n" + "\n".join(f"- {l}" for l in c_lines))
        if bridge_texts:
            sections.append("Multi-hop bridge clues:\n" + "\n".join(f"- {t}" for t in bridge_texts))
        all_facts = []
        for eid in seed_e:
            for t in e_to_facts.get(eid, []):
                if t not in all_facts:
                    all_facts.append(t)
        for t, _, _, _ in final_pool:
            if t not in all_facts:
                all_facts.append(t)
        sections.append("Candidate supporting facts:\n" + "\n".join(f"- {t}" for t in all_facts))
        context = "\n\n".join(sections)
    elif supp_only:
        # Ablation GBRAG_V18_SUPP_ONLY=1: keep only the supplementary evidence
        # pool (drop the path-ordered segments).
        sections = []
        supplement = []
        for facts in e_to_facts.values():
            for t in facts:
                if t not in supplement:
                    supplement.append(t)
        for t in entity_added + middle_added:
            if t and t not in supplement:
                supplement.append(t)
        if supplement:
            sections.append("=== Supplementary Evidence Pool ===\n\n" + "\n".join(f"- {t}" for t in supplement))
        if c_lines:
            sections.append("Relevant mid-level clusters:\n" + "\n".join(f"- {l}" for l in c_lines))
        if bridge_texts:
            sections.append("Multi-hop bridge clues:\n" + "\n".join(f"- {t}" for t in bridge_texts))
        context = "\n\n".join(sections)
    else:
        parts = []
        used_c, used_e = set(), set()
        path_emitted = set()
        for path in paths:
            chain = []
            for pos, eid in enumerate(path):
                if eid not in used_c:
                    facts = list(dict.fromkeys(e_to_facts.get(eid, [])))
                    if facts:
                        chain.append(f"[Regional facts of {eid}]\n" + "\n".join(f"- {x}" for x in facts))
                        path_emitted.update(facts)
                    used_c.add(eid)
                if pos + 1 < len(path):
                    edge_text = edge_info.get((eid, path[pos + 1])) or edge_info.get((path[pos + 1], eid))
                    if edge_text and edge_text not in used_e:
                        chain.append(f"==> [Edge Bridge Proposition] {edge_text}")
                        used_e.add(edge_text)
            if chain:
                parts.append("\n".join(chain))
        sections = []
        if parts:
            sections.append("=== Ordered Multi-hop Evidence Context ===\n\n" + "\n\n".join(parts))
        # supplementary pool: seed-C facts not on any selected path + entity-C facts
        supplement = []
        # seed-C facts not covered by any selected path
        for eid in seed_e:
            if eid not in used_c:
                for t in dict.fromkeys(e_to_facts.get(eid, [])):
                    if t not in supplement:
                        supplement.append(t)
        # entity-activated C facts and middle-C facts (both already ungated);
        # skip middle facts already emitted on a path
        for t in entity_added + middle_added:
            if t and t not in supplement and t not in path_emitted:
                supplement.append(t)
        if supplement and not no_supp:
            sections.append("=== Supplementary Evidence Pool ===\n\n" + "\n".join(f"- {t}" for t in supplement))
        if c_lines:
            sections.append("Relevant mid-level clusters:\n" + "\n".join(f"- {l}" for l in c_lines))
        if bridge_texts:
            sections.append("Multi-hop bridge clues:\n" + "\n".join(f"- {t}" for t in bridge_texts))
        context = "\n\n".join(sections)
    if not context.strip():
        context = "Candidate supporting facts:\n" + "\n".join(f"- {x[0]}" for x in final_pool)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    answer_started = time.perf_counter()
    answer = _answer_from_context(query, context, llm)
    end = time.perf_counter()
    lengths = [max(0, len(p) - 1) for p in paths]
    trace = {
        "retrieved_context": context, "experiment_id": "DCB_v18_dual_pool",
        "p1_top_d": selected_d, "v15_top_chunk_ids": top_chunk_ids,
        "v15_gated_d_count": len(centroids), "v15_entity_e_candidate_count": len(entity_e),
        "v15_entity_d_bonus_count": len(entity_d), "v11_parent_e_seeds": seed_e,
        "selected_paths": paths, "context_paths": paths,
        "activated_node_type": "e_ball", "activated_node_count": len(path_e),
        "path_activated": bool(paths), "path_count": len(paths),
        "avg_path_length": round(sum(lengths) / len(lengths), 6) if lengths else 0.0,
        "max_path_length": max(lengths, default=0),
        "activated_relation_count": sum(max(0, len(p) - 1) for p in paths),
        "selected_fact_count": sum(len(x) for x in e_to_facts.values()),
        "v18_supplement_count": len(supplement) if (not fallback_used and not no_supp) else 0,
        "v18_bridge_count": len(bridge_texts), "v18_e_summary_count": len(c_lines),
        "v18_middle_added": len(middle_added), "v18_entity_added": len(entity_added),
        "v18_fallback_used": fallback_used,
        "v18_ablation": {"no_chunk_gate": no_chunk_gate, "no_entity_c": no_entity_c,
                          "no_domain_bonus": no_domain_bonus, "no_path": no_path,
                          "no_fallback": no_fallback, "no_supp": no_supp,
                          "supp_only": supp_only,
                          "no_bridge": no_bridge, "no_c_summary": no_c_summary},
        "retrieval_seconds": round(end - started, 6),
        "answer_generation_seconds": round(end - answer_started, 6),
        "path_search_seconds": round(time.perf_counter() - path_started, 6),
        "v11_path_edge_candidates": len(edge_info),
    }
    return (answer, trace) if return_trace else answer
