import collections
import json
import os
import time
from typing import Any, List, Dict
import re
import difflib
import numpy as np
import ast
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from graph_retriever import Graph_Rag_pipeline
from tqdm import tqdm
from file_path import PROPOSITIONIZER_DIR
from prompt import _build_nli_judge_prompt
from standard_metrics import (
    best_em_f1 as standard_best_em_f1,
    ensure_answer_list as standard_ensure_answer_list,
    exact_match as standard_exact_match,
    token_f1 as standard_token_f1,
)

def _normalize_eval_text_base(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s*'s\b", " ", text)
    return " ".join(text.split())

def _ensure_answer_list(answer_value: Any) -> List[str]:
    return standard_ensure_answer_list(answer_value)


def compute_relaxed_exact_match(prediction: str, gold_answer: str) -> float:
    norm_pred = _normalize_eval_text(prediction)
    norm_gold = _normalize_eval_text(gold_answer)

    # 1. Strict equality.
    if norm_pred == norm_gold:
        return 1.0

    p_tokens = norm_pred.split()
    g_tokens = norm_gold.split()
    if not p_tokens or not g_tokens:
        return 0.0
    p_set = set(p_tokens)
    g_set = set(g_tokens)

    # 2. Subset/superset relations (two tiers):
    #    a) single-token short answers: a subset counts when lengths differ by
    #       at most 1 (e.g. "2006" vs "August 2006").
    if abs(len(p_tokens) - len(g_tokens)) <= 1:
        if all(w in p_set for w in g_set) or all(w in g_set for w in p_set):
            return 1.0
    #    b) multi-word granularity: only "prediction subset of gold" counts
    #       (the model answered coarsely); gold subset of prediction is NOT
    #       accepted (prevents over-answer snooping such as "civilian and military").
    if len(p_set) >= 2 and len(g_set) >= 2 and p_set <= g_set:
        return 1.0

    # 3. Acronym match: either side is an all-caps acronym (or plural acronym)
    #    appearing as a standalone word in the other side
    #    (e.g. "DMSS" vs "Defense Medical Surveillance System (DMSS)"; both directions).
    raw_pred = str(prediction).strip().strip("\"'")
    raw_gold = str(gold_answer).strip().strip("\"'")
    acronym = r"[A-Z][A-Z0-9&-]{1,9}s?"
    if re.fullmatch(acronym, raw_pred) and raw_pred.lower() in g_set:
        return 1.0
    if re.fullmatch(acronym, raw_gold) and raw_gold.lower() in p_set:
        return 1.0

    # 4. Fallback: very high character similarity (covers singular/plural and
    #    prefix/suffix differences).
    if difflib.SequenceMatcher(None, norm_pred, norm_gold).ratio() >= 0.85:
        return 1.0

    return 0.0

def compute_relaxed_token_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = _normalize_eval_text(prediction).split()
    gold_tokens = _normalize_eval_text(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_relaxed_best_em_f1(prediction: str, gold_answers: List[str]) -> Dict[str, Any]:
    best_em = 0.0
    best_f1 = 0.0
    matched_gold_answer = gold_answers[0] if gold_answers else ""

    for gold_answer in gold_answers:
        em = compute_relaxed_exact_match(prediction, gold_answer)
        f1 = compute_relaxed_token_f1(prediction, gold_answer)
        if (em, f1) > (best_em, best_f1):
            best_em = em
            best_f1 = f1
            matched_gold_answer = gold_answer

    return {
        "em": best_em,
        "f1": best_f1,
        "matched_gold_answer": matched_gold_answer
    }


def compute_exact_match(prediction: str, gold_answer: str) -> float:
    """GraphRAG Standard EM; permissive matching lives in compute_relaxed_*"""
    return standard_exact_match(prediction, gold_answer)


def compute_token_f1(prediction: str, gold_answer: str) -> float:
    """GraphRAG Standard token F1."""
    return standard_token_f1(prediction, gold_answer)


def compute_best_em_f1(prediction: str, gold_answers: List[str]) -> Dict[str, Any]:
    """GraphRAG Standard EM/F1 over the available gold answers."""
    return standard_best_em_f1(prediction, gold_answers)

def _parse_nli_label(raw_text: str) -> str:
    if raw_text is None:
        return "neutral"
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    data = json.loads(text)
    label = str(data.get("label", "neutral")).lower().strip()
    if label not in {"entailment", "neutral", "contradiction"}:
        return "neutral"
    return label

def _fallback_nli_label(prediction: str, gold_answer: str) -> str:
    norm_pred = _normalize_eval_text(prediction)
    norm_gold = _normalize_eval_text(gold_answer)
    if not norm_pred or not norm_gold:
        return "neutral"
    if norm_pred == norm_gold or norm_gold in norm_pred or norm_pred in norm_gold:
        return "entailment"
    return "neutral"

def judge_prediction_with_nli(question: str, prediction: str, gold_answers: List[str], llm) -> Dict[str, Any]:
    gold_answers = gold_answers or [""]
    label_rank = {"contradiction": 0, "neutral": 1, "entailment": 2}
    best_result = {
        "nli_label": "neutral",
        "nli_score": 0.0,
        "matched_gold_answer": gold_answers[0],
        "nli_raw_response": None
    }

    chain = _build_nli_judge_prompt() | llm if llm is not None else None

    for gold_answer in gold_answers:
        raw_response = None

        if _fallback_nli_label(prediction, gold_answer) == "entailment":
            label = "entailment"
            raw_response = "rule_fallback"
        elif chain is not None:
            try:
                raw_response = chain.invoke({
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": prediction
                }).content
                label = _parse_nli_label(raw_response)
            except Exception as e:
                raw_response = f"nli_error: {e}"
                label = _fallback_nli_label(prediction, gold_answer)
        else:
            label = _fallback_nli_label(prediction, gold_answer)

        score = 1.0 if label == "entailment" else 0.0
        if label_rank[label] > label_rank[best_result["nli_label"]]:
            best_result = {
                "nli_label": label,
                "nli_score": score,
                "matched_gold_answer": gold_answer,
                "nli_raw_response": raw_response
            }
        if label == "entailment":
            break

    return best_result

def evaluate_prediction(question: str, prediction: str, gold_answers: List[str], llm) -> Dict[str, Any]:
    em_f1_result = compute_best_em_f1(prediction, gold_answers)
    relaxed_result = compute_relaxed_best_em_f1(prediction, gold_answers)
    nli_result = judge_prediction_with_nli(question, prediction, gold_answers, llm)
    return {
        "em": em_f1_result["em"],
        "f1": em_f1_result["f1"],
        "em_f1_matched_gold_answer": em_f1_result["matched_gold_answer"],
        "relaxed_accuracy": relaxed_result["em"],
        "relaxed_f1": relaxed_result["f1"],
        "relaxed_matched_gold_answer": relaxed_result["matched_gold_answer"],
        "nli_label": nli_result["nli_label"],
        "nli_score": nli_result["nli_score"],
        "nli_matched_gold_answer": nli_result["matched_gold_answer"],
        "nli_raw_response": nli_result["nli_raw_response"]
    }


class ContextMetricCalculator:
    """Unified T5 proposition post-processing for both retrieval baselines."""
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, PROPOSITIONIZER_DIR)
        # main.py pins CUDA_VISIBLE_DEVICES; by convention:
        #   cuda:0 = the main program; cuda:1 = this T5 evidence evaluator.
        # Default cuda:1; single-GPU/CPU environments fall back to cuda:0 / cpu.
        self.device = os.environ.get("CONTEXT_METRIC_DEVICE", "cuda:1")
        if not torch.cuda.is_available():
            print("T5 evidence evaluator: no CUDA detected, falling back to CPU (local debugging only).")
            self.device = "cpu"
        elif self.device.startswith("cuda:"):
            device_index = int(self.device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                print(
                    f"T5 evidence evaluator: configured {self.device} is unavailable "
                    f"(this process sees {torch.cuda.device_count()} GPU(s)); "
                    "falling back to cuda:0."
                )
                self.device = "cuda:0"
        print(f"T5 evidence evaluator device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    def propositionize(self, context: str) -> List[str]:
        props = []
        for start in range(0, max(1, len(context)), 970):
            chunk = context[start:start + 1000]
            if not chunk:
                continue
            inputs = self.tokenizer(
                "Extract all independent factual propositions from the following text: " + chunk,
                return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=512)
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            try:
                parsed = ast.literal_eval(raw) if raw.startswith("[") else None
            except (ValueError, SyntaxError):
                parsed = None
            items = parsed if isinstance(parsed, list) else raw.splitlines()
            props.extend(re.sub(r"^[-*•\d.)\s]+", "", str(item)).strip(" \"'") for item in items)
        return list(dict.fromkeys(item for item in props if item))

    def compute(self, question: str, supporting_sentences: List[str], retrieved_context: str,
                embedding_model, evidence_threshold: float = 0.75, cqt_threshold: float = 0.5) -> Dict[str, Any]:
        context = str(retrieved_context or "")
        props = self.propositionize(context)
        supports = [str(item).strip() for item in supporting_sentences or [] if str(item).strip()]
        token_count = len(re.findall(r"\w+|[^\w\s]", context, flags=re.UNICODE))
        result = {"context_token_count": token_count, "context_prop_count": len(props),
                  "evidence_threshold": evidence_threshold, "cqt_threshold": cqt_threshold}
        if not props:
            result.update({"evidence_recall": 0.0, "all_evidence_hit": 0.0, "evidence_precision": 0.0,
                           "evidence_hit_count": 0, "gold_evidence_count": len(supports),
                           "relevant_context_prop_count": 0, "cqt": 0.0})
            return result
        texts = [question] + props + supports
        embeddings = np.asarray(embedding_model.encode(texts), dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        query_emb = embeddings[0]
        prop_emb = embeddings[1:1 + len(props)]
        support_emb = embeddings[1 + len(props):]
        if len(support_emb):
            similarities = support_emb @ prop_emb.T
            support_hits = similarities.max(axis=1) >= evidence_threshold
            prop_hits = similarities.max(axis=0) >= evidence_threshold
        else:
            support_hits = np.zeros(0, dtype=bool)
            prop_hits = np.zeros(len(props), dtype=bool)
        hit_count = int(support_hits.sum())
        relevant_count = int(prop_hits.sum())
        weights = np.asarray([len(re.findall(r"\w+|[^\w\s]", item, flags=re.UNICODE)) for item in props], dtype=float)
        relevance = np.maximum(0.0, (prop_emb @ query_emb - cqt_threshold) / (1.0 - cqt_threshold))
        result.update({
            "evidence_recall": round(hit_count / len(supports), 6) if supports else 0.0,
            "all_evidence_hit": float(bool(supports) and hit_count == len(supports)),
            "evidence_precision": round(relevant_count / len(props), 6),
            "evidence_hit_count": hit_count, "gold_evidence_count": len(supports),
            "relevant_context_prop_count": relevant_count,
            "cqt": float((weights * relevance).sum() / max(weights.sum(), 1.0)),
        })
        return result

def build_run_output_paths(qa_benchmark_path: str, qa_items: List[Dict[str, Any]]) -> Dict[str, str]:
    dataset_name = os.path.basename(os.path.dirname(qa_benchmark_path)) if qa_benchmark_path else ""
    if not dataset_name and qa_items and isinstance(qa_items[0], dict):
        dataset_name = qa_items[0].get("domain", "default_dataset")
    if not dataset_name:
        dataset_name = "default_dataset"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Support a custom output root via environment variable for batched server runs.
    output_root = os.environ.get("GBRAG_OUTPUT_ROOT", os.path.join(project_root, "outputs"))
    log_root_dir = output_root
    dataset_log_dir = os.path.join(log_root_dir, dataset_name)
    os.makedirs(dataset_log_dir, exist_ok=True)

    # Infer a "_no_edges" marker from qa_benchmark_path; if the passed path
    # contains no_edges, the output files get the same suffix.
    experiment_id = os.environ.get("GBRAG_EXPERIMENT_ID", "DCB_edges")
    prompt_variant = os.environ.get("GBRAG_ANSWER_PROMPT_V6", "0") == "1"
    suffix = f"_{experiment_id}"
    if prompt_variant and experiment_id == "DCB_v5":
        suffix += "_answer_prompt_v6"

    return {
        "dataset_name": dataset_name,
        "dataset_log_dir": dataset_log_dir,
        "qa_case_log_path": os.path.join(dataset_log_dir, f"qa_case_log{suffix}.txt"),
        "predictions_path": os.path.join(dataset_log_dir, f"new_ball_predictions{suffix}.jsonl"),
        "path_metrics_path": os.path.join(dataset_log_dir, f"path_activation_metrics{suffix}.jsonl"),
        "timing_metrics_path": os.path.join(dataset_log_dir, f"timing_metrics{suffix}.jsonl"),
        "summary_path": os.path.join(dataset_log_dir, f"new_ball_metrics_summary{suffix}.json")
    }

def summarize_benchmark_results(results: List[Dict[str, Any]], dataset_name: str, qa_benchmark_path: str) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "dataset_name": dataset_name,
            "qa_benchmark_path": qa_benchmark_path,
            "sample_count": 0,
            "avg_em": 0.0,
            "avg_f1": 0.0,
            "avg_nli_score": 0.0
        }

    avg_em = sum(item["em"] for item in results) / total
    avg_f1 = sum(item["f1"] for item in results) / total
    avg_nli = sum(item["nli_score"] for item in results) / total

    def average(key):
        values = [float(item[key]) for item in results if item.get(key) is not None]
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "dataset_name": dataset_name,
        "qa_benchmark_path": qa_benchmark_path,
        "sample_count": total,
        "avg_em": round(avg_em, 4),
        "avg_f1": round(avg_f1, 4),
        "avg_nli_score": round(avg_nli, 4),
        "avg_em_percent": round(avg_em * 100, 2),
        "avg_f1_percent": round(avg_f1 * 100, 2),
        "avg_nli_percent": round(avg_nli * 100, 2),
        "avg_relaxed_accuracy": average("relaxed_accuracy"),
        "avg_relaxed_f1": average("relaxed_f1"),
        "evidence_recall": average("evidence_recall"),
        "all_evidence_hit_rate": average("all_evidence_hit"),
        "evidence_precision": average("evidence_precision"),
        "avg_context_token_count": average("context_token_count"),
        "avg_context_prop_count": average("context_prop_count"),
        "cqt": average("cqt"),
        "exact_match_count": sum(1 for item in results if item["em"] == 1.0),
        "nli_entailment_count": sum(1 for item in results if item["nli_label"] == "entailment"),
    }

def run_qa_benchmark_with_metrics(
    qa_items,
    qa_benchmark_path,
    E_balls,
    D_balls,
    edges,
    Propositions,
    llm,
    embedding_model,
    db_helper,
    edge_propagation=True,
    experiment_config=None,
    max_new_samples=None,
):
    output_paths = build_run_output_paths(qa_benchmark_path, qa_items)
    results = []
    required_metric_keys = {"evidence_recall", "all_evidence_hit", "evidence_precision", "context_token_count", "cqt"}
    if os.path.exists(output_paths["predictions_path"]):
        with open(output_paths["predictions_path"], "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                try:
                    record = json.loads(line)
                    if required_metric_keys.issubset(record):
                        results.append(record)
                except json.JSONDecodeError:
                    continue
    completed_ids = {str(item.get("sample_id")) for item in results}
    context_metric_calculator = ContextMetricCalculator()
    summary = summarize_benchmark_results(results, output_paths["dataset_name"], qa_benchmark_path)

    new_samples_done = 0
    with open(output_paths["qa_case_log_path"], "a", encoding="utf-8") as qa_case_log, open(output_paths["predictions_path"], "a", encoding="utf-8") as predictions_file, open(output_paths["path_metrics_path"], "a", encoding="utf-8") as path_metrics_file, open(output_paths["timing_metrics_path"], "a", encoding="utf-8") as timing_metrics_file:
        for idx, qa_item in enumerate(tqdm(qa_items, desc="QA Benchmark"), start=1):
            if isinstance(qa_item, dict):
                question = str(qa_item.get("question", "")).strip()
                gold_answers = _ensure_answer_list(qa_item.get("answer"))
                sample_id = qa_item.get("_id", f"sample_{idx}")
                sample_type = qa_item.get("type", "")
                domain = qa_item.get("domain", output_paths["dataset_name"])
                source_chunk_id = qa_item.get("source_chunk_id", -1)
                context_file = qa_item.get("context_file", "")
                supporting_sentences = qa_item.get("supporting_sentences", [])
            else:
                question = str(qa_item).strip()
                gold_answers = [""]
                sample_id = f"sample_{idx}"
                sample_type = ""
                domain = output_paths["dataset_name"]
                source_chunk_id = -1
                context_file = ""
                supporting_sentences = []

            if not question:
                print(f"Skipping empty-question sample: {sample_id}")
                continue
            if str(sample_id) in completed_ids:
                continue

            if max_new_samples is not None and new_samples_done >= max_new_samples:
                print(f"Batch limit {max_new_samples} reached; exiting early.")
                break

            print("\n" + "=" * 24 + f" Sample {idx}/{len(qa_items)} " + "=" * 24)
            print(f"sample_id: {sample_id}")
            print(f"question: {question}")
            print(f"gold_answers: {gold_answers}")

            sample_started = time.perf_counter()
            pipeline_result = Graph_Rag_pipeline(
                question,
                E_balls,
                D_balls,
                edges,
                Propositions,
                llm,
                embedding_model,
                db_helper,
                enable_edge_propagation=edge_propagation,
                return_trace=True,
                experiment_config=experiment_config,
            )
            if isinstance(pipeline_result, tuple):
                prediction, retrieval_trace = pipeline_result
            else:
                prediction, retrieval_trace = pipeline_result, {}
            retrieval_seconds = float(retrieval_trace.get("retrieval_seconds", 0.0))
            answer_generation_seconds = float(retrieval_trace.get("answer_generation_seconds", 0.0))
            latency_seconds = round(retrieval_seconds + answer_generation_seconds, 4)

            nli_started = time.perf_counter()
            metric_result = evaluate_prediction(question, prediction, gold_answers, llm)
            nli_evaluation_seconds = time.perf_counter() - nli_started
            t5_started = time.perf_counter()
            retrieval_metric = context_metric_calculator.compute(
                question, supporting_sentences, retrieval_trace.get("retrieved_context", ""), embedding_model
            )
            t5_context_metric_seconds = time.perf_counter() - t5_started
            record = {
                "question_no": idx,
                "sample_id": sample_id,
                "source_chunk_id": source_chunk_id,
                "context_file": context_file,
                "domain": domain,
                "type": sample_type,
                "question": question,
                "gold_answers": gold_answers,
                "prediction": prediction,
                "latency_seconds": latency_seconds,
                "experiment_id": getattr(experiment_config, "experiment_id", "DCB_edges"),
                "retrieval_seconds": round(retrieval_seconds, 6),
                "answer_generation_seconds": round(answer_generation_seconds, 6),
                "nli_evaluation_seconds": round(nli_evaluation_seconds, 6),
                "t5_context_metric_seconds": round(t5_context_metric_seconds, 6),
                "em": round(metric_result["em"], 4),
                "f1": round(metric_result["f1"], 4),
                "relaxed_accuracy": round(metric_result["relaxed_accuracy"], 4),
                "relaxed_f1": round(metric_result["relaxed_f1"], 4),
                "relaxed_matched_gold_answer": metric_result["relaxed_matched_gold_answer"],
                "nli_label": metric_result["nli_label"],
                "nli_score": round(metric_result["nli_score"], 4),
                "em_f1_matched_gold_answer": metric_result["em_f1_matched_gold_answer"],
                "nli_matched_gold_answer": metric_result["nli_matched_gold_answer"],
                "nli_raw_response": metric_result["nli_raw_response"],
                "retrieval_trace": retrieval_trace,
                **retrieval_metric,
            }
            results.append(record)
            new_samples_done += 1

            write_started = time.perf_counter()
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            os.fsync(predictions_file.fileno())
            path_record = {
                "sample_id": sample_id,
                "type": sample_type,
                **retrieval_trace,
                **retrieval_metric,
            }
            path_metrics_file.write(json.dumps(path_record, ensure_ascii=False) + "\n")
            path_metrics_file.flush()
            os.fsync(path_metrics_file.fileno())
            result_write_seconds = time.perf_counter() - write_started
            timing_record = {
                "sample_id": sample_id,
                "experiment_id": record["experiment_id"],
                "retrieval_seconds": record["retrieval_seconds"],
                "answer_generation_seconds": record["answer_generation_seconds"],
                "nli_evaluation_seconds": record["nli_evaluation_seconds"],
                "t5_context_metric_seconds": record["t5_context_metric_seconds"],
                "result_write_seconds": round(result_write_seconds, 6),
                "sample_wall_seconds": round(time.perf_counter() - sample_started, 6),
                "context_token_count": retrieval_metric.get("context_token_count", 0),
            }
            timing_record["path_search_seconds"] = round(
                float(retrieval_trace.get("path_search_seconds", 0.0)), 6
            )
            timing_metrics_file.write(json.dumps(timing_record, ensure_ascii=False) + "\n")
            timing_metrics_file.flush()
            os.fsync(timing_metrics_file.fileno())

            qa_case_log.write(
                f"question_no: {idx}\n"
                f"sample_id: {sample_id}\n"
                f"source_chunk_id: {source_chunk_id}\n"
                f"question: {question}\n"
                f"gold_answers: {' | '.join(gold_answers)}\n"
                f"prediction: {prediction}\n"
                f"{'=' * 80}\n"
            )
            qa_case_log.flush()

            summary = summarize_benchmark_results(
                results,
                output_paths["dataset_name"],
                qa_benchmark_path
            )
            with open(output_paths["summary_path"], "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print("Current sample evaluation result:")
            print(json.dumps({
                "sample_id": record["sample_id"],
                "latency_seconds": record["latency_seconds"],
                "em": record["em"],
                "f1": record["f1"],
                "nli_label": record["nli_label"],
                "prediction": record["prediction"]
            }, indent=2, ensure_ascii=False))

    return results, summary, output_paths


# ============ Number-word normalization (relaxed metrics only; standard metrics unaffected) ============
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}
_ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}


def _normalize_eval_text(text: Any) -> str:
    """Base normalization plus English number-word mapping (three->3, fourth->4)."""
    norm = _normalize_eval_text_base(text)
    tokens = [_NUMBER_WORDS.get(t, _ORDINAL_WORDS.get(t, t)) for t in norm.split()]
    return " ".join(tokens)
