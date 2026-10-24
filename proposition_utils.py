import ast
import json
import re
from typing import List

import numpy as np

from prompt import entity_prompt, time_prompt, tag_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import nltk
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from json_tool import append_json_item, start_json_list_file, finish_json_list_file


def generate_embeddings(context_str, embed_model):
    if isinstance(context_str, str):
        sentences = nltk.sent_tokenize(context_str.strip())
    else:
        sentences = context_str

    embedding = embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embedding.tolist()

    return sentences, embeddings


def clean_list(items: List[str], max_items: int) -> List[str]:
    cleaned = []
    seen = set()

    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip().strip("\"'")
        if not normalized:
            continue

        key = normalized.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

        if len(cleaned) >= max_items:
            break

    return cleaned

def parse_keyed_list(raw_content: str, key: str) -> List[str]:
    # Strip possible Markdown wrappers from the LLM output (e.g. ```python ... ```)
    raw_content = re.sub(r"```[a-zA-Z]*", "", raw_content)
    raw_content = raw_content.replace("```", "")

    # Support two formats: key = [...] (primary) and JSON "key": [...] (defensive).
    patterns = [
        fr"{key}\s*=\s*(\[.*?\])",
        fr'"{key}"\s*:\s*(\[.*?\])',
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, raw_content, re.DOTALL)
        if match:
            break
    if not match:
        return []

    try:
        # ast.literal_eval is strict about formatting; clean stray spaces first.
        list_str = match.group(1).strip()
        parsed = ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item).strip() for item in parsed if str(item).strip()]

def llm_abstract_concurrent(
    pure_props_filename: str,
    llm,
    embed_model,
    collection,
    output_filename: str,
    max_workers: int = 4,
) -> List[dict]:
    """
    Concurrent LLM metadata extraction: reuses entity_prompt / time_prompt /
    tag_prompt and processes multiple propositions with a ThreadPoolExecutor.

    Output format is identical to the legacy llm_abstract: propositions_data.json
    fields: idx, id, proposition, entity_list, extend_key_list, time_entity_list
    """
    from json_tool import start_json_list_file, append_json_item, finish_json_list_file

    with open(pure_props_filename, "r", encoding="utf-8") as f:
        propositions = json.load(f)

    entity_chain = entity_prompt | llm
    time_chain = time_prompt | llm
    tag_chain = tag_prompt | llm

    proposition_key_lists = []
    start_json_list_file(output_filename)
    is_first_item = True
    global_idx = 0

    def process_single(pro_dict: dict, chunk_id: int) -> dict:
        """Process one proposition and return its proposition_info dict."""
        text = pro_dict["text"]
        ctx = pro_dict["source_chunk"]

        # Three LLM calls per proposition (reusing the same chains).
        entity_resp = entity_chain.invoke({"Proposition": text, "SupportContext": ctx})
        time_resp = time_chain.invoke({"Proposition": text, "SupportContext": ctx})
        tag_resp = tag_chain.invoke({"Proposition": text})

        entity_list = clean_list(parse_keyed_list(str(entity_resp.content), "entity_list"), max_items=3)
        time_list = clean_list(parse_keyed_list(str(time_resp.content), "time_entity_list"), max_items=3)
        tag_list = clean_list(parse_keyed_list(str(tag_resp.content), "extend_key_list"), max_items=2)

        return {
            "idx": None,  # filled in uniformly below
            "id": chunk_id,
            "proposition": text,
            "entity_list": entity_list,
            "extend_key_list": tag_list,
            "time_entity_list": time_list,
        }

    # Flatten all propositions into a single list, each tagged with its chunk_id.
    all_tasks = []
    for item in propositions:
        chunk_id = item["id"]
        for pro_dict in item["propositions"]:
            all_tasks.append((pro_dict, chunk_id))

    print(f"[Stats] {len(all_tasks)} propositions, {max_workers} concurrent workers...")

    # --- Pre-flight health probe: run 3 samples through the LLM; abort early
    # if none of them yields a parseable entity list. ---
    probe_ok = 0
    for pro_dict, _cid in all_tasks[:3]:
        try:
            _r = entity_chain.invoke({"Proposition": pro_dict["text"], "SupportContext": pro_dict["source_chunk"]})
            if parse_keyed_list(str(_r.content), "entity_list"):
                probe_ok += 1
        except Exception:
            pass
    if probe_ok == 0:
        raise RuntimeError(
            "LLM entity-parsing probe failed: none of the first 3 samples produced an entity_list. "
            "The LLM endpoint is likely in a bad state; aborting early to avoid wasting hours of compute. "
            "Check that the model behind GBRAG_LLM_BASE_URL produces valid output."
        )
    print(f"[OK] LLM probe passed ({probe_ok}/3 samples produced entities)")

    # Concurrent execution.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, pro_dict, chunk_id): (pro_dict, chunk_id)
            for pro_dict, chunk_id in all_tasks
        }

        # Collect results in completion order (dict preserves idx order).
        results = [None] * len(all_tasks)
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Extraction"):
            try:
                result = future.result()
                # Locate this result's position in all_tasks.
                pro_dict, chunk_id = futures[future]
                task_idx = all_tasks.index((pro_dict, chunk_id))
                results[task_idx] = result
            except Exception as e:
                pro_dict, chunk_id = futures[future]
                task_idx = all_tasks.index((pro_dict, chunk_id))
                print(f"[FAIL] Proposition processing failed (chunk_id={chunk_id}): {e}")
                results[task_idx] = {
                    "idx": None,
                    "id": chunk_id,
                    "proposition": pro_dict["text"],
                    "entity_list": [],
                    "extend_key_list": [],
                    "time_entity_list": [],
                }

    # Write JSON in original order and track the all-empty rate (early signal
    # of an abnormal LLM state).
    all_empty_count = 0
    for local_idx, result in enumerate(results):
        result["idx"] = f"A_{global_idx}"
        proposition_key_lists.append(result)
        is_first_item = append_json_item(output_filename, result, is_first_item)
        global_idx += 1
        if not result["entity_list"] and not result["extend_key_list"] and not result["time_entity_list"]:
            all_empty_count += 1
        if (local_idx + 1) % 1000 == 0:
            print(f"[Stats] Progress {local_idx + 1}/{len(results)}, all-empty {all_empty_count} ({all_empty_count / (local_idx + 1):.1%})")

    finish_json_list_file(output_filename)
    print(f"[OK] Done: {len(results)} propositions processed, all-empty {all_empty_count} ({all_empty_count / max(len(results), 1):.1%})")
    return proposition_key_lists

def cal_cos(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos = np.dot(emb1, emb2) / (norm1 * norm2)
    return cos

def jaccard_similarity(s1, s2):
    """
    Lexical Jaccard similarity between two entity strings (word-level overlap).
    Both sides are lowercased and leading articles are stripped.
    """

    def clean_str(s):

        s = str(s).lower().strip()
        # Strip a leading article (the / a / an) and the following space.

        s = re.sub(r'^(the|a|an)\s+', '', s)

        return set(s.split())

    set1 = clean_str(s1)
    set2 = clean_str(s2)

    # If cleaning leaves an empty set, return 0.
    if not set1 or not set2:
        return 0.0

    # Compute the intersection and union sizes.
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Return the Jaccard index.
    return intersection / union

def cluster_similar_entities(all_entity, model, threshold=0.75, jaccard_threshold=0.55):
    origin_cluster_entity = []
    all_entity_embed = model.encode(all_entity)
    extinct_entity_idx = []
    for i in range(len(all_entity)):
        if i not in extinct_entity_idx:
            group_idx = []
            group = []
            group_idx.append(i)
            group.append(all_entity[i])
            for j in range(i + 1, len(all_entity)):
                cos = cal_cos(all_entity_embed[i], all_entity_embed[j])
                lex_sim = jaccard_similarity(all_entity[i], all_entity[j])
                if lex_sim >= jaccard_threshold or cos > threshold:
                    group_idx.append(j)
                    group.append(all_entity[j])

            if group_idx:
                extinct_entity_idx.extend(group_idx)
                origin_cluster_entity.append(group)
        else:
            continue
    # print(origin_cluster_entity)

    return origin_cluster_entity

def normalize_entities_rule_based(Propositions, model):
    """
    Entity normalization:
      1. Collect every entity that appears in any proposition.
      2. Cluster synonymous entities with semantic + lexical similarity.
      3. Pick the shortest entity of each cluster as the canonical name.
      4. Return a mapping dict in {canonical_name: [all variants]} format.
      5. Also rewrite each proposition's entity_list to canonical names.
    """
    all_entity = set()
    for pro in Propositions:
        if pro["entity_list"] and isinstance(pro["entity_list"], list):
            for entity in pro["entity_list"]:
                if isinstance(entity, str):
                    clean_entity = entity.strip()
                    if clean_entity:
                        all_entity.add(clean_entity)

    # Cluster synonymous entities.
    entity_clusters = cluster_similar_entities(list(all_entity), model)

    # Build the legacy map_dic (variant -> canonical) to rewrite entity_list.
    map_dic = {}

    for cluster in entity_clusters:
        if len(cluster) == 1:
            map_dic[cluster[0]] = cluster[0]
        else:
            candidates_entity = [e for e in cluster if not e.lower().startswith("the")]
            if not candidates_entity:
                candidates_entity = cluster
            shortest_entity = sorted(candidates_entity, key=len)[0]

            for entity in cluster:
                map_dic[entity] = shortest_entity

    # Rewrite each proposition's entity_list to canonical names.
    for pro in Propositions:
        if pro["entity_list"] and isinstance(pro["entity_list"], list):
            new_entity_list = []
            for entity in pro["entity_list"]:
                clean_entity = entity.strip()
                map_name = map_dic.get(clean_entity, clean_entity)
                new_entity_list.append(map_name)
            pro["entity_list"] = list(set(new_entity_list))

    # Build the new-format map {canonical_name: [variant list]}.
    reverse_map = {}
    for entity, canonical in map_dic.items():
        if canonical not in reverse_map:
            reverse_map[canonical] = []
        if entity not in reverse_map[canonical]:
            reverse_map[canonical].append(entity)

    return Propositions, reverse_map
