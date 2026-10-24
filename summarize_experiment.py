"""Summarize raw GBRAG timing JSONL without using an LLM."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


FIELDS = (
    "retrieval_seconds", "answer_generation_seconds", "nli_evaluation_seconds",
    "t5_context_metric_seconds", "result_write_seconds", "sample_wall_seconds",
)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(values, bootstrap_samples=2000, seed=20260721):
    values = [float(x) for x in values]
    if not values:
        return {key: 0.0 for key in ("mean", "median", "p90", "p95", "std", "ci95_low", "ci95_high")}
    rng = random.Random(seed)
    boot_means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples)]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "std": statistics.pstdev(values),
        "ci95_low": percentile(boot_means, 0.025),
        "ci95_high": percentile(boot_means, 0.975),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.timing_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {
        "experiment_id": rows[0].get("experiment_id") if rows else None,
        "sample_count": len(rows),
        "timing": {field: summarize([row[field] for row in rows if row.get(field) is not None]) for field in FIELDS},
    }
    wall_mean = result["timing"]["sample_wall_seconds"]["mean"]
    result["questions_per_hour"] = 3600.0 / wall_mean if wall_mean else 0.0
    output = args.output or args.timing_jsonl.with_suffix(".summary.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
