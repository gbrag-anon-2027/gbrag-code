# -*- coding: utf-8 -*-
"""P1/P2 variant experiment runner.

Variants (selected by the GBRAG_VARIANT environment variable):
  edge_entity     P1a: edges replaced by "pure entity-overlap edges"
                    (relation text = shared entity names, no boundary propositions)
  edge_similarity P1b: edges replaced by "pure similarity edges"
                    (nearest cross-ball centroid pairs, relation text = most
                    similar proposition pair)
  kmeans          P2: B/C balls replaced by K-means clustering
                    (edges use entity-overlap rule, same as P1a)

Data loading reuses the same caches as main (only the in-memory edges/balls
are swapped; no cache or index is rewritten), evaluation goes through the same
evaluator pipeline, and outputs match the v18 structure for direct comparison.

Environment variables (same as main, except GBRAG_VARIANT):
  GBRAG_DATASETS / GBRAG_BATCH_SIZE / GBRAG_OUTPUT_ROOT (a per-variant suffix
  is appended automatically)
  GBRAG_KM_B_PER_C  number of B balls per K-means C ball (default 4)
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GBRAG_CUDA_DEVICES", "0"))
os.environ.setdefault("BALL_MAIN_DEVICE", "cuda:0")
# NOTE: do not setdefault CONTEXT_METRIC_DEVICE here — placing the T5 evidence
# evaluator on GPU can exhaust VRAM; run it on CPU instead.


import json
import itertools
import numpy as np

from model import BallModel
from file_path import File_path
from dataset_map import DATASET_MAP
from graph_builder import build_full_graph
from json_tool import load_qa_benchmark
from evaluator import run_qa_benchmark_with_metrics
from experiment_config import get_experiment_config
from edge_config import EdgeMode, set_edge_mode
import graph_retriever as v5


def _entity_set(c_ball):
    out = set()
    for x in c_ball.get("entity_list", []) or []:
        n = v5._normalize_entity_name(str(x))
        if n:
            out.add(n)
    return out


def build_entity_edges(C_balls, max_edges):
    """Entity-overlap edges: connect C-ball pairs sharing entities; the
    relation text is the shared entity names (no proposition evidence)."""
    pairs = []
    for a, b in itertools.combinations(C_balls, 2):
        shared = _entity_set(a) & _entity_set(b)
        if shared:
            pairs.append((len(shared), a["idx"], b["idx"], sorted(shared)))
    pairs.sort(key=lambda x: (-x[0], str(x[1]), str(x[2])))
    edges = []
    for _, i, j, shared in pairs[:max_edges]:
        edges.append({"node_C_1": i, "node_C_2": j,
                      "relation": ", ".join(shared),
                      "shared_entities": list(shared)})
    return edges


def build_similarity_edges(C_balls, Propositions, embedding_model, db_helper, max_edges):
    """Similarity edges: take top pairs by summary-vector similarity instead of
    the boundary-proposition screening (semantic deviation + shared entities).

    The linking criterion is purely semantic similarity; the relation text is the
    concatenation of both C-ball summaries.
    """
    centroids, summaries = {}, {}
    for c in C_balls:
        summary = str(c.get("summary", "")).strip()
        text = summary if summary else str(c.get("idx", ""))
        summaries[c["idx"]] = text
        centroids[c["idx"]] = np.asarray(embedding_model.encode([text])[0], dtype=np.float32)
    ids = list(centroids)
    X = np.stack([centroids[i] for i in ids])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = X @ X.T
    cand = []
    for a_i, b_i in itertools.combinations(range(len(ids)), 2):
        cand.append((float(sim[a_i, b_i]), ids[a_i], ids[b_i]))
    cand.sort(key=lambda x: x[0], reverse=True)
    by_id = {c["idx"]: c for c in C_balls}
    edges = []
    for s, i, j in cand[:max_edges]:
        shared = sorted(_entity_set(by_id[i]) & _entity_set(by_id[j]))
        edges.append({"node_C_1": i, "node_C_2": j,
                      "relation": f"{summaries[i]} <=> {summaries[j]}",
                      "shared_entities": shared})
    return edges


def _pack_balls(labels, prefix, Xn, Propositions, b_per_c, sub_method, sub_kw=None):
    """Pack clustering labels into C/B ball structures (shared by P2 variants).

    labels: a C-level cluster label per proposition (label -1 is noise and is
    assigned to the nearest centroid cluster);
    sub_method: how B balls are split inside each C ball
    ("kmeans"/"agglomerative"/"hdbscan").
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    labels = np.asarray(labels)
    # Assign noise points (label -1) to the nearest centroid.
    n_c = int(labels.max()) + 1
    if (labels < 0).any():
        centers = np.zeros((n_c, Xn.shape[1]), dtype=np.float32)
        counts = np.bincount(labels[labels >= 0], minlength=n_c)
        for li, x in zip(labels, Xn):
            if li >= 0:
                centers[li] += x
        centers /= np.maximum(counts[:, None], 1)
        noise = np.where(labels < 0)[0]
        labels = labels.copy()
        labels[noise] = (centers @ Xn[noise].T).argmax(axis=0)
    C_balls, B_balls = [], []
    for label in range(n_c):
        member_idx = np.where(labels == label)[0]
        if len(member_idx) == 0:
            continue
        c_idx = f"{prefix}_C{label}"
        entities = []
        for mi in member_idx:
            for e in Propositions[int(mi)].get("entity_list", []) or []:
                if e not in entities:
                    entities.append(e)
                if len(entities) >= 60:
                    break
        k_b = max(1, min(b_per_c, len(member_idx)))
        Xsub = Xn[member_idx]
        if sub_method == "kmeans":
            sub = KMeans(n_clusters=k_b, n_init=4, random_state=0).fit(Xsub).labels_
        elif sub_method == "agglomerative":
            from sklearn.neighbors import kneighbors_graph
            k = min(15, len(member_idx) - 1)
            conn = kneighbors_graph(Xsub, n_neighbors=k, mode="connectivity",
                                    include_self=False) if k >= 2 else None
            sub = AgglomerativeClustering(n_clusters=k_b, linkage="ward",
                                          connectivity=conn).fit(Xsub).labels_
        else:  # hdbscan
            import hdbscan
            min_cs = max(5, len(member_idx) // max(k_b, 1) // 2)
            sub = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=3).fit_predict(Xsub)
            # Assign noisy B points round-robin to existing clusters so that
            # every proposition still has a home.
            if (sub < 0).any() and len(set(sub[sub >= 0])) > 0:
                pos = sorted(set(sub[sub >= 0]))
                mapping = {p: i % max(k_b, 1) for i, p in enumerate(pos)}
                sub = np.array([mapping.get(s, i % max(k_b, 1)) if s >= 0 else i % max(k_b, 1)
                                for i, s in enumerate(sub)])
        b_indices = []
        for bl in range(max(sub) + 1):
            b_idx = f"{prefix}_B{label}_{bl}"
            b_indices.append(b_idx)
            B_balls.append({"idx": b_idx,
                            "all_indices": [f"A_{int(member_idx[i])}" for i in np.where(sub == bl)[0]]})
        C_balls.append({"idx": c_idx, "b_indices": b_indices,
                        "entity_list": entities, "summary": ""})
    return C_balls, B_balls


def _embed_matrix(Propositions, embedding_model, db_helper):
    idx_to_emb = v5._get_a_embeddings_idx(Propositions, embedding_model, db_helper)
    X = np.stack([np.asarray(idx_to_emb[i], dtype=np.float32) for i in range(len(Propositions))])
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def build_kmeans_balls(Propositions, embedding_model, db_helper, n_c, b_per_c=4):
    """P2a: K-means version of the B/C balls (partitional clustering)."""
    from sklearn.cluster import KMeans
    Xn = _embed_matrix(Propositions, embedding_model, db_helper)
    labels = KMeans(n_clusters=n_c, n_init=4, random_state=0).fit_predict(Xn)
    return _pack_balls(labels, "KM", Xn, Propositions, b_per_c, "kmeans")


def build_agglomerative_balls(Propositions, embedding_model, db_helper, n_c, b_per_c=4):
    """P2b: hierarchical-clustering version of the B/C balls (Ward linkage with
    a kNN connectivity graph to avoid O(n^2))."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    Xn = _embed_matrix(Propositions, embedding_model, db_helper)
    n = len(Xn)
    conn = kneighbors_graph(Xn, n_neighbors=15, mode="connectivity", include_self=False)
    labels = AgglomerativeClustering(n_clusters=n_c, linkage="ward",
                                     connectivity=conn).fit_predict(Xn)
    return _pack_balls(labels, "AGG", Xn, Propositions, b_per_c, "agglomerative")


def build_hdbscan_balls(Propositions, embedding_model, db_helper, n_c, b_per_c=4):
    """P2c: density-based version of the B/C balls (HDBSCAN; the cluster count is
    decided by the data, n_c is only a reference)."""
    import hdbscan
    Xn = _embed_matrix(Propositions, embedding_model, db_helper)
    min_cs = max(10, len(Xn) // max(n_c, 1))
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=3).fit_predict(Xn)
    return _pack_balls(labels, "HDB", Xn, Propositions, b_per_c, "hdbscan")


def main():
    variant = os.environ.get("GBRAG_VARIANT", "").strip()
    assert variant in ("edge_entity", "edge_similarity", "kmeans",
                       "agglomerative", "hdbscan"), \
        "GBRAG_VARIANT must be one of edge_entity / edge_similarity / kmeans / agglomerative / hdbscan"
    batch_size = int(os.environ.get("GBRAG_BATCH_SIZE", "200"))
    datasets = [x.strip() for x in os.environ.get("GBRAG_DATASETS", "2WikiMultiHopQA").split(",") if x.strip()]
    output_root = os.environ.get("GBRAG_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs_variants"))
    config = get_experiment_config()
    current = os.path.abspath(__file__)
    fp = File_path(current)
    model = BallModel(current)
    embedding = model._get_embedding_model()
    llm = model._get_llm_model()
    set_edge_mode(EdgeMode.WITH_EDGES)
    b_per_c = int(os.environ.get("GBRAG_KM_B_PER_C", "4"))

    for dataset_name in datasets:
        if dataset_name not in DATASET_MAP:
            raise ValueError(dataset_name)
        props_path = fp._get_propositions_data_path(dataset_name)
        with open(props_path, encoding="utf-8") as f:
            propositions = json.load(f)
        graph_dir = fp._get_graph_chroma_db_path_with_edges(dataset_name)
        C, B, D, edges, db = build_full_graph(propositions, embedding, llm, graph_dir)

        if variant == "edge_entity":
            run_edges = build_entity_edges(C, max_edges=len(edges))
            run_C, run_B = C, B
        elif variant == "edge_similarity":
            run_edges = build_similarity_edges(C, propositions, embedding, db, max_edges=len(edges))
            run_C, run_B = C, B
        else:
            if variant == "kmeans":
                run_C, run_B = build_kmeans_balls(propositions, embedding, db, n_c=len(C), b_per_c=b_per_c)
            elif variant == "agglomerative":
                run_C, run_B = build_agglomerative_balls(propositions, embedding, db, n_c=len(C), b_per_c=b_per_c)
            else:  # hdbscan
                run_C, run_B = build_hdbscan_balls(propositions, embedding, db, n_c=len(C), b_per_c=b_per_c)
            run_edges = build_entity_edges(run_C, max_edges=len(edges))

        print(f"[{variant}] {dataset_name}: edges {len(edges)} -> variant edges {len(run_edges)}; "
              f"C {len(C)}->{len(run_C)}, B {len(B)}->{len(run_B)}")

        os.environ["GBRAG_RAW_CORPUS_PATH"] = fp._get_chunk_path(dataset_name)
        qa_path = fp._get_qa_path(dataset_name)
        qa_items = load_qa_benchmark(qa_path)
        os.environ["GBRAG_OUTPUT_ROOT"] = os.path.join(output_root, f"{variant}_b{batch_size}")
        run_qa_benchmark_with_metrics(
            qa_items=qa_items, qa_benchmark_path=qa_path,
            C_balls=run_C, B_balls=run_B, edges=run_edges,
            Propositions=propositions, llm=llm, embedding_model=embedding,
            db_helper=db, edge_propagation=True, experiment_config=config,
            max_new_samples=batch_size)


if __name__ == "__main__":
    main()
