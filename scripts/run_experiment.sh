#!/usr/bin/env bash
# End-to-end example: build the graph and evaluate GBRAG on one dataset.
#
# Usage:
#   bash scripts/run_experiment.sh [GPU_IDS] [DATASET]
# Example:
#   bash scripts/run_experiment.sh 0 HotpotQA
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export GBRAG_CUDA_DEVICES="${1:-0}"
export GBRAG_DATASETS="${2:-GovReport_Geopolitics}"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

mkdir -p outputs

# 1) Build the granular-ball graph (skip if caches already exist).
GBRAG_STAGE=build python -u main.py

# 2) Run the QA benchmark with full metrics.
python -u main.py 2>&1 | tee "outputs/run_${GBRAG_DATASETS}.log"

# 3) (Optional) text-RAG baseline for the same dataset.
# GBRAG_RAG_MODE=dense GBRAG_RAG_TOP_K=10 python -u naive_rag.py

# 4) (Optional) timing summary with bootstrap confidence intervals.
# python summarize_experiment.py outputs/${GBRAG_DATASETS}/timing_metrics_DCB_v18_dual_pool.jsonl
