# -*- coding: utf-8 -*-
"""Text-RAG baselines: BM25 / dense top-k chunk retrieval + the same generator
and evaluation protocol as the v18 GBRAG pipeline.

Purpose: main comparison in the paper (GBRAG vs. text RAG).

Environment variables:
  GBRAG_DATASETS              dataset names (comma-separated)
  GBRAG_OUTPUT_ROOT           output root directory (default ../outputs_naive_rag)
  GBRAG_BATCH_SIZE            max number of QA items to evaluate (0 = all)
  GBRAG_CUDA_DEVICES          GPU ids (same as main)
  GBRAG_RAG_MODE              dense (default) | bm25
  GBRAG_RAG_TOP_K           number of chunks to retrieve (default 10)
  GBRAG_RAG_MAX_CHARS         context character budget (default 48000; 4000 for
                              equal-budget comparison)
  GBRAG_RAG_TAG               output file-name tag (default {mode}_k{top_k}_b{1k-budget})

Outputs (same structure as v18, ready for direct table filling):
  {OUTPUT_ROOT}/{dataset}/naive_rag_predictions_{tag}.jsonl
  {OUTPUT_ROOT}/{dataset}/naive_rag_metrics_summary_{tag}.json
All metrics (EM/F1/NLI/ER/EP/CQT/token) reuse the evaluator implementation so
the numbers are directly comparable with GBRAG.
"""
import json
import math
import os
import re
import time

import numpy as np

if "GBRAG_CUDA_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GBRAG_CUDA_DEVICES"]

from model import BallModel
from file_path import File_path
from dataset_map import DATASET_MAP
from json_tool import load_qa_benchmark
from evaluator import (
    _ensure_answer_list,
    evaluate_prediction,
    summarize_benchmark_results,
    ContextMetricCalculator,
)
import graph_retriever as v5


def _tokenize(text):
    return re.findall(r"\w+", str(text).lower())


class BM25:
    """BM25 implementation without third-party dependencies."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [_tokenize(d) for d in docs]
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1)
        df = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(self.doc_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q_tokens = set(_tokenize(query))
        out = np.zeros(len(self.doc_tokens), dtype=np.float64)
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            dl = len(toks)
            for t in q_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                score += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out[i] = score
        return out


def _load_corpus_chunks(chunk_path):
    with open(chunk_path, encoding="utf-8") as f:
        corpus = json.load(f)
    ids, texts = [], []
    for i, item in enumerate(corpus):
        ids.append(int(item.get("chunk_id", i)))
        texts.append(str(item.get("text", "")))
    return ids, texts


def _get_chunk_embeddings(texts, embedding_model, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    print("naive_rag: encoding chunk embeddings and caching ->", cache_path)
    embs = np.asarray(
        embedding_model.encode(texts, show_progress_bar=True, batch_size=64),
        dtype=np.float32,
    )
    np.save(cache_path, embs)
    return embs


def run_dataset(dataset_name, mode, top_k, max_chars, tag, output_root, batch_size):
    file_tool = File_path(os.path.abspath(__file__))
    chunk_path = file_tool._get_chunk_path(dataset_name)
    qa_path = file_tool._get_qa_path(dataset_name)

    ball_model = BallModel(os.path.abspath(__file__))
    llm = ball_model._get_llm_model()
    embedding_model = ball_model._get_embedding_model()

    print(f"\n===== naive_rag [{mode}] dataset {dataset_name} =====")
    print("chunk file:", chunk_path)
    ids, texts = _load_corpus_chunks(chunk_path)
    print("num chunks:", len(texts))

    if mode == "bm25":
        ranker = BM25(texts)
        chunk_embs = None
    else:
        cache_path = str(chunk_path) + f".rag_emb.npy"
        chunk_embs = _get_chunk_embeddings(texts, embedding_model, cache_path)

    qa_items = load_qa_benchmark(qa_path)
    if batch_size > 0:
        qa_items = qa_items[:batch_size]
    print("num QA items:", len(qa_items))

    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"naive_rag_predictions_{tag}.jsonl")
    summary_path = os.path.join(out_dir, f"naive_rag_metrics_summary_{tag}.json")

    done = set()
    rows = []
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(r["sample_id"])
                    rows.append(r)
        print("Resume: already completed", len(done), "items")

    calculator = ContextMetricCalculator()

    for idx, item in enumerate(qa_items):
        sample_id = item.get("_id") or item.get("sample_id") or f"{dataset_name}_{idx}"
        if sample_id in done:
            continue
        question = str(item.get("question", "")).strip()
        gold_answers = _ensure_answer_list(item.get("answer", ""))
        supports = item.get("supporting_sentences", []) or []

        started = time.perf_counter()
        if mode == "bm25":
            scores = ranker.scores(question)
        else:
            q_emb = np.asarray(embedding_model.encode([question])[0], dtype=np.float32)
            denom = np.maximum(np.linalg.norm(chunk_embs, axis=1), 1e-12)
            scores = (chunk_embs / denom[:, None]) @ (q_emb / max(np.linalg.norm(q_emb), 1e-12))
        top_idx = np.argsort(-np.asarray(scores))[:top_k]

        picked = []
        total = 0
        # Truncate each chunk to an equal-share budget (a single Gov chunk can
        # exceed the whole budget; a hard break would leave an empty context).
        per_chunk_cap = max(200, max_chars // top_k)
        for i in top_idx:
            t = texts[int(i)].strip()[:per_chunk_cap]
            if not t:
                continue
            line = f"- {t}"
            if total + len(line) > max_chars:
                break
            picked.append(t)
            total += len(line)
        context = "\n".join(f"- {t}" for t in picked)

        answer = v5._answer_from_context(question, context, llm)
        eval_result = evaluate_prediction(question, answer, gold_answers, llm)
        ctx_result = calculator.compute(question, supports, context, embedding_model)

        row = {
            "sample_id": sample_id,
            "question": question,
            "prediction": answer,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "rag_mode": mode,
            "rag_top_k": top_k,
            "rag_chunk_count": len(picked),
            "em": eval_result["em"],
            "f1": eval_result["f1"],
            "relaxed_accuracy": eval_result["relaxed_accuracy"],
            "relaxed_f1": eval_result["relaxed_f1"],
            "nli_label": eval_result["nli_label"],
            "nli_score": eval_result["nli_score"],
            "is_d_summary_answer": False,
            "evidence_recall": ctx_result["evidence_recall"],
            "all_evidence_hit": ctx_result["all_evidence_hit"],
            "evidence_precision": ctx_result["evidence_precision"],
            "context_token_count": ctx_result["context_token_count"],
            "context_prop_count": ctx_result["context_prop_count"],
            "cqt": ctx_result["cqt"],
        }
        rows.append(row)
        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{dataset_name} {mode}] {idx + 1}/{len(qa_items)}  EM={row['em']} F1={row['f1']}")

    summary = summarize_benchmark_results(rows, dataset_name, qa_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary written to:", summary_path)
    print(json.dumps({k: summary[k] for k in ("sample_count", "avg_em", "avg_f1",
                                              "avg_nli_percent", "evidence_recall",
                                              "avg_context_token_count")}, ensure_ascii=False))


def main():
    mode = os.environ.get("GBRAG_RAG_MODE", "dense").strip().lower()
    assert mode in ("dense", "bm25"), f"GBRAG_RAG_MODE must be dense/bm25, got: {mode}"
    top_k = int(os.environ.get("GBRAG_RAG_TOP_K", "10"))
    max_chars = int(os.environ.get("GBRAG_RAG_MAX_CHARS", "48000"))
    tag = os.environ.get("GBRAG_RAG_TAG", f"{mode}_k{top_k}_b{max_chars // 1000}k")
    output_root = os.environ.get(
        "GBRAG_OUTPUT_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs_naive_rag"),
    )
    batch_size = int(os.environ.get("GBRAG_BATCH_SIZE", "0"))
    datasets = [d.strip() for d in os.environ.get("GBRAG_DATASETS", "GovReport_Geopolitics").split(",") if d.strip()]

    print(f"naive_rag config: mode={mode} top_k={top_k} max_chars={max_chars} tag={tag}")
    print("output root:", output_root)
    for ds in datasets:
        if ds not in DATASET_MAP:
            raise KeyError(f"dataset {ds} not in DATASET_MAP: {list(DATASET_MAP)}")
        run_dataset(ds, mode, top_k, max_chars, tag, output_root, batch_size)


if __name__ == "__main__":
    main()
