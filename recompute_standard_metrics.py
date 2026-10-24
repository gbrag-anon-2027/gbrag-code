"""Offline, non-mutating GraphRAG-standard evaluation and paired analysis.

This script reads existing prediction/trace JSONL files only.  It never calls
the retriever or an answer model and never writes into an original run
directory.  Reports are written under ``standard_eval_reports/<timestamp>``.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import string
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from standard_metrics import best_em_f1, ensure_answer_list, normalize_answer, token_f1


ROOT = Path(__file__).resolve().parents[1]
QA_DEFAULT = ROOT / "dataset" / "GovReport_Geopolitics" / "qa_benchmark_govreport.json"
V3_DEFAULT = ROOT / "outputs_p1_v3" / "GovReport_Geopolitics" / "new_ball_predictions_DCB_v3.jsonl"
V5_DEFAULT = ROOT / "outputs_p1_v5" / "GovReport_Geopolitics" / "new_ball_predictions_DCB_v5.jsonl"
GR_DEFAULT = ROOT / "baseline" / "GraphRAG" / "new" / "runs" / "GovReport_Geopolitics" / "metrics" / "trace.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def sample_id(row: Dict[str, Any]) -> str:
    value = row.get("sample_id", row.get("_id"))
    return "" if value is None else str(value)


def qa_index(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("_id")): item for item in items if item.get("_id") is not None}


def duplicate_ids(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(sample_id(row) for row in rows)
    return {key: count for key, count in counts.items() if key and count > 1}


def index_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index unique rows; keep the last row, while reporting duplicates."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = sample_id(row)
        if sid:
            indexed[sid] = row
    return indexed


def old_relaxed_normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s*'s\b", " ", text)
    return " ".join(text.split())


def old_relaxed_em(prediction: Any, gold: Any) -> float:
    pred = old_relaxed_normalize(prediction)
    target = old_relaxed_normalize(gold)
    if pred == target:
        return 1.0
    p_tokens, g_tokens = pred.split(), target.split()
    if not p_tokens or not g_tokens:
        return 0.0
    p_set, g_set = set(p_tokens), set(g_tokens)
    if abs(len(p_tokens) - len(g_tokens)) <= 1 and (g_set <= p_set or p_set <= g_set):
        return 1.0
    if len(p_set) >= 2 and len(g_set) >= 2 and p_set <= g_set:
        return 1.0
    raw_pred = str(prediction).strip().strip("\"'")
    raw_gold = str(gold).strip().strip("\"'")
    acronym = r"[A-Z][A-Z0-9&-]{1,9}s?"
    if re.fullmatch(acronym, raw_pred) and raw_pred.lower() in g_set:
        return 1.0
    if re.fullmatch(acronym, raw_gold) and raw_gold.lower() in p_set:
        return 1.0
    return float(difflib.SequenceMatcher(None, pred, target).ratio() >= 0.85)


def old_relaxed_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = old_relaxed_normalize(prediction).split()
    gold_tokens = old_relaxed_normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_relaxed(prediction: Any, golds: Iterable[Any]) -> Dict[str, Any]:
    answers = [str(x) for x in golds] or [""]
    return max(
        ({"em": old_relaxed_em(prediction, answer),
          "f1": old_relaxed_f1(prediction, answer),
          "matched_gold_answer": answer} for answer in answers),
        key=lambda item: (item["em"], item["f1"]),
    )


def row_golds(row: Dict[str, Any], qa: Optional[Dict[str, Any]]) -> List[str]:
    if "gold_answers" in row:
        return ensure_answer_list(row.get("gold_answers"))
    if "answer" in row:
        return ensure_answer_list(row.get("answer"))
    if qa is not None:
        return ensure_answer_list(qa.get("answer"))
    return [""]


def score_row(row: Dict[str, Any], qa: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prediction = row.get("prediction", row.get("answer", ""))
    golds = row_golds(row, qa)
    standard = best_em_f1(prediction, golds)
    relaxed = best_relaxed(prediction, golds)
    result = {
        "sample_id": sample_id(row),
        "type": str(row.get("type", qa.get("type", "") if qa else "")).lower(),
        "question": row.get("question", qa.get("question", "") if qa else ""),
        "prediction": prediction,
        "gold_answers": golds,
        "standard_em": standard["em"],
        "standard_f1": standard["f1"],
        "standard_matched_gold_answer": standard["matched_gold_answer"],
        "relaxed_accuracy": relaxed["em"],
        "relaxed_f1": relaxed["f1"],
        "relaxed_matched_gold_answer": relaxed["matched_gold_answer"],
        "original_em": row.get("em"),
        "original_f1": row.get("f1"),
    }
    # Keep only small, useful retrieval/evidence fields for reports.
    trace = row.get("retrieval_trace")
    if isinstance(trace, dict):
        result["retrieval_trace"] = {
            key: trace.get(key)
            for key in ("p1_comparison_hit", "retrieved_context", "evidence_hit_count",
                        "gold_evidence_count", "evidence_recall", "p1_final_count",
                        "p1_fallback_used", "p1_max_semantic")
            if key in trace
        }
    for key in ("final_context", "retrieved_context", "evidence_hit_count", "gold_evidence_count",
                "evidence_recall", "matched_evidence_count", "p1_comparison_hit"):
        if key in row:
            result[key] = row[key]
    return result


def summarize(scored: Sequence[Dict[str, Any]], include_by_type: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sample_count": len(scored)}
    for metric in ("standard_em", "standard_f1", "relaxed_accuracy", "relaxed_f1"):
        values = [float(row[metric]) for row in scored]
        out[metric] = sum(values) / len(values) if values else None
        out[f"{metric}_percent"] = 100 * out[metric] if values else None
    out["standard_em_count"] = sum(row["standard_em"] == 1.0 for row in scored)
    out["standard_f1_exact_count"] = sum(row["standard_f1"] == 1.0 for row in scored)
    out["relaxed_accuracy_count"] = sum(row["relaxed_accuracy"] == 1.0 for row in scored)
    by_type: Dict[str, Any] = {}
    if include_by_type:
        for kind in sorted({row["type"] for row in scored}):
            subset = [row for row in scored if row["type"] == kind]
            by_type[kind] = summarize(subset, include_by_type=False)
        out["by_type"] = by_type
    return out


def alignment_report(qa_items: Sequence[Dict[str, Any]], datasets: Dict[str, Sequence[Dict[str, Any]]], limit: int) -> Dict[str, Any]:
    expected = [str(item.get("_id")) for item in qa_items[:limit]]
    expected_set = set(expected)
    report: Dict[str, Any] = {"requested_count": limit, "expected_ids": expected}
    for name, rows in datasets.items():
        ids = [sample_id(row) for row in rows]
        unique = set(ids)
        report[name] = {
            "rows_read": len(rows),
            "unique_sample_ids": len(unique),
            "duplicate_sample_ids": duplicate_ids(rows),
            "first_limit_ids": ids[:limit],
            "matched_expected_count": len(expected_set & unique),
            "missing_expected_ids": sorted(expected_set - unique),
            "extra_ids_in_first_limit": sorted(set(ids[:limit]) - expected_set),
            "first_limit_exact_order_match": ids[:limit] == expected,
        }
    common = expected_set.copy()
    for rows in datasets.values():
        common &= {sample_id(row) for row in rows}
    report["all_dataset_common_expected_count"] = len(common & expected_set)
    report["all_dataset_common_expected_100_percent"] = len(common & expected_set) == len(expected_set)
    return report


def get_context(row: Dict[str, Any]) -> str:
    trace = row.get("retrieval_trace")
    if isinstance(trace, dict) and trace.get("retrieved_context") is not None:
        return str(trace.get("retrieved_context"))
    return str(row.get("retrieved_context", row.get("final_context", "")) or "")


def evidence_status(row: Dict[str, Any], qa: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    trace = row.get("retrieval_trace") if isinstance(row.get("retrieval_trace"), dict) else {}
    hit = row.get("matched_evidence_count", row.get("evidence_hit_count", trace.get("evidence_hit_count")))
    total = row.get("gold_evidence_count", trace.get("gold_evidence_count"))
    if hit is None or total is None:
        supports = (qa or {}).get("supporting_sentences", [])
        context = normalize_answer(get_context(row))
        hit = sum(bool(normalize_answer(sentence) and normalize_answer(sentence) in context)
                  for sentence in supports)
        total = len(supports)
    try:
        hit_i, total_i = int(hit), int(total)
    except (TypeError, ValueError):
        hit_i, total_i = 0, 0
    return {"hit_count": hit_i, "gold_count": total_i,
            "recall": (hit_i / total_i if total_i else None),
            "all_hit": bool(total_i and hit_i >= total_i)}


def comparison_attribution(scored: Sequence[Dict[str, Any]], qa_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    incorrect = [row for row in scored if row["type"] == "comparison" and row["standard_em"] != 1.0]
    counts = Counter()
    details: List[Dict[str, Any]] = []
    for row in incorrect:
        trace = row.get("retrieval_trace") if isinstance(row.get("retrieval_trace"), dict) else {}
        detector = row.get("p1_comparison_hit", trace.get("p1_comparison_hit"))
        evidence = evidence_status(row, qa_by_id.get(row["sample_id"]))
        if detector is False:
            category = "comparison_detector_missed"
        elif evidence["hit_count"] == 0:
            category = "both_sides_evidence_missed"
        elif evidence["gold_count"] >= 2 and evidence["hit_count"] < evidence["gold_count"]:
            category = "only_one_side_or_partial_evidence"
        elif row["relaxed_accuracy"] == 1.0 or row["standard_f1"] > 0:
            category = "evidence_present_but_granularity_or_normalization"
        elif evidence["all_hit"]:
            category = "both_sides_evidence_present_answer_wrong"
        else:
            category = "evidence_or_answer_uncertain"
        counts[category] += 1
        details.append({"sample_id": row["sample_id"], "category": category,
                        "standard_em": row["standard_em"], "standard_f1": row["standard_f1"],
                        "relaxed_accuracy": row["relaxed_accuracy"], "evidence": evidence,
                        "detector_flag": detector, "prediction": row["prediction"],
                        "gold_answers": row["gold_answers"]})
    return {"incorrect_comparison_count": len(incorrect), "category_counts": dict(counts), "details": details}


def pairwise(v3: Dict[str, Dict[str, Any]], v5: Dict[str, Dict[str, Any]], ids: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sample_count": 0, "metrics": {}, "rows": []}
    common = [sid for sid in ids if sid in v3 and sid in v5]
    out["sample_count"] = len(common)
    for metric in ("standard_em", "standard_f1"):
        categories = Counter()
        rows = []
        for sid in common:
            left, right = v3[sid], v5[sid]
            l_ok, r_ok = left[metric] == 1.0, right[metric] == 1.0
            if not l_ok and r_ok:
                category = "v3_wrong_v5_right"
            elif l_ok and not r_ok:
                category = "v3_right_v5_wrong"
            elif l_ok and r_ok:
                category = "both_right"
            else:
                category = "both_wrong"
            categories[category] += 1
            rows.append({"sample_id": sid, "type": left["type"],
                         "v3": left[metric], "v5": right[metric],
                         "delta_v5_minus_v3": right[metric] - left[metric],
                         "category": category})
        out["metrics"][metric] = {"categories": dict(categories),
                                   "mean_v3": sum(x["v3"] for x in rows) / len(rows) if rows else None,
                                   "mean_v5": sum(x["v5"] for x in rows) / len(rows) if rows else None,
                                   "mean_delta_v5_minus_v3": sum(x["delta_v5_minus_v3"] for x in rows) / len(rows) if rows else None,
                                   "positive_delta_count": sum(x["delta_v5_minus_v3"] > 0 for x in rows),
                                   "negative_delta_count": sum(x["delta_v5_minus_v3"] < 0 for x in rows),
                                   "unchanged_count": sum(x["delta_v5_minus_v3"] == 0 for x in rows)}
        if metric == "standard_em":
            out["rows"] = rows
    for kind in ("bridge", "comparison"):
        subset = [sid for sid in common if v3[sid]["type"] == kind]
        out.setdefault("by_type", {})[kind] = {"sample_count": len(subset)}
        for metric in ("standard_em", "standard_f1"):
            wins = sum(v3[sid][metric] != 1.0 and v5[sid][metric] == 1.0 for sid in subset)
            losses = sum(v3[sid][metric] == 1.0 and v5[sid][metric] != 1.0 for sid in subset)
            out["by_type"][kind][metric] = {
                "v3_wrong_v5_right": wins, "v3_right_v5_wrong": losses,
                "net_edge_gain": wins - losses,
            }
            deltas = [v5[sid][metric] - v3[sid][metric] for sid in subset]
            out["by_type"][kind][metric].update({
                "mean_delta_v5_minus_v3": sum(deltas) / len(deltas) if deltas else None,
                "positive_delta_count": sum(delta > 0 for delta in deltas),
                "negative_delta_count": sum(delta < 0 for delta in deltas),
                "unchanged_count": sum(delta == 0 for delta in deltas),
            })
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):.2f}%"


def markdown_summary(summary: Dict[str, Any], alignment: Dict[str, Any], paired: Dict[str, Any], attribution: Dict[str, Any]) -> str:
    lines = ["# GraphRAG Standard EM/F1 Offline Re-evaluation", "", "All values below are recomputed from existing prediction/trace JSONL files; original files were not modified.", "", "## Unified metrics"]
    lines += ["", "| Method / subset | N | Standard EM | Standard F1 | Relaxed accuracy | Relaxed F1 |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in summary.items():
        lines.append(f"| {key} | {value['sample_count']} | {pct(value['standard_em'])} | {pct(value['standard_f1'])} | {pct(value['relaxed_accuracy'])} | {pct(value['relaxed_f1'])} |")
    lines += ["", "## ID alignment", "", f"All three datasets contain the expected first-100 IDs: **{alignment.get('all_dataset_common_expected_100_percent')}**."]
    for name, value in alignment.items():
        if isinstance(value, dict) and name not in ("expected_ids",):
            lines.append(f"- {name}: rows={value.get('rows_read')}, unique={value.get('unique_sample_ids')}, matched={value.get('matched_expected_count')}, duplicates={len(value.get('duplicate_sample_ids', {}))}, exact_order={value.get('first_limit_exact_order_match')}")
    lines += ["", "## Paired v3 → v5", "", "| Metric / type | v3 wrong → v5 right | v3 right → v5 wrong | Net edge gain |", "|---|---:|---:|---:|"]
    for kind, value in paired.get("by_type", {}).items():
        for metric, stats in value.items():
            if metric in ("standard_em", "standard_f1"):
                lines.append(f"| {kind} / {metric} | {stats['v3_wrong_v5_right']} | {stats['v3_right_v5_wrong']} | {stats['net_edge_gain']} |")
    overall_f1 = paired.get("metrics", {}).get("standard_f1", {})
    lines += ["", f"Continuous paired Standard F1 delta (v5−v3): **{100 * overall_f1.get('mean_delta_v5_minus_v3', 0):.2f} pp**; positive={overall_f1.get('positive_delta_count', 0)}, negative={overall_f1.get('negative_delta_count', 0)}, unchanged={overall_f1.get('unchanged_count', 0)}."]
    for kind, value in paired.get("by_type", {}).items():
        f1_stats = value.get("standard_f1", {})
        lines.append(f"- {kind}: mean delta {100 * f1_stats.get('mean_delta_v5_minus_v3', 0):.2f} pp; positive={f1_stats.get('positive_delta_count', 0)}, negative={f1_stats.get('negative_delta_count', 0)}, unchanged={f1_stats.get('unchanged_count', 0)}")
    lines += ["", "## Comparison error attribution", ""]
    for method, value in attribution.items():
        lines.append(f"**{method}** (incorrect comparison questions: {value['incorrect_comparison_count']})")
        for category, count in value["category_counts"].items():
            lines.append(f"- {category}: {count}")
    lines += ["", "## Metric rule", "", "Standard EM/F1 is copied from `baseline/GraphRAG/graphrag/metrics.py`: lowercase; remove ASCII `string.punctuation`; remove standalone `a/an/the`; collapse whitespace; split on whitespace; Counter overlap precision/recall F1; for multiple gold answers take the lexicographic maximum `(EM,F1)`. Relaxed values are diagnostics only.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, default=QA_DEFAULT)
    parser.add_argument("--v3", type=Path, default=V3_DEFAULT)
    parser.add_argument("--v5", type=Path, default=V5_DEFAULT)
    parser.add_argument("--graphrag", type=Path, default=GR_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    qa_items = read_json(args.qa)
    if not isinstance(qa_items, list):
        raise ValueError(f"QA file must contain a list: {args.qa}")
    qa_by_id = qa_index(qa_items)
    raw = {"v3": read_jsonl(args.v3), "v5": read_jsonl(args.v5), "GraphRAG": read_jsonl(args.graphrag)}
    alignment = alignment_report(qa_items, raw, 100)
    out = args.out or (ROOT / "standard_eval_reports" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)

    indexed = {name: index_rows(rows) for name, rows in raw.items()}
    limits = {"v3_first100": 100, "v5_first100": 100, "v5_first120": 120, "GraphRAG_first100": 100, "GraphRAG_all": len(raw["GraphRAG"])}
    scored: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    for name, limit in limits.items():
        source = name.split("_first")[0].split("_all")[0]
        ids = [str(item.get("_id")) for item in qa_items[:limit]] if source != "GraphRAG" or name.endswith("first100") else [sample_id(row) for row in raw["GraphRAG"]]
        rows = [indexed[source][sid] for sid in ids if sid in indexed[source]]
        scored[name] = [score_row(row, qa_by_id.get(sample_id(row))) for row in rows]
        summary[name] = summarize(scored[name])
        write_json(out / f"{name}.json", {"summary": summary[name], "rows": scored[name]})

    v3_scores = {row["sample_id"]: row for row in scored["v3_first100"]}
    v5_scores = {row["sample_id"]: row for row in scored["v5_first100"]}
    common_ids = [str(item.get("_id")) for item in qa_items[:100] if str(item.get("_id")) in v3_scores and str(item.get("_id")) in v5_scores]
    paired = pairwise(v3_scores, v5_scores, common_ids)
    write_json(out / "paired_v3_v5.json", paired)
    write_csv(out / "paired_v3_v5.csv", paired.get("rows", []))
    attribution = {
        "v3": comparison_attribution(scored["v3_first100"], qa_by_id),
        "v5": comparison_attribution(scored["v5_first100"], qa_by_id),
        "GraphRAG": comparison_attribution(scored["GraphRAG_first100"], qa_by_id),
    }
    write_json(out / "comparison_error_attribution.json", attribution)
    write_json(out / "id_alignment.json", alignment)
    write_json(out / "summary.json", {"metrics": summary, "alignment": alignment, "paired_v3_v5": paired, "comparison_attribution": attribution})
    (out / "REPORT.md").write_text(markdown_summary(summary, alignment, paired, attribution), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "summary": summary, "alignment": alignment, "paired": paired.get("by_type"), "attribution_counts": {k: v["category_counts"] for k, v in attribution.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
