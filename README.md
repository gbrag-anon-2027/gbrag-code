# GBRAG: Granular-Ball Hierarchical Retrieval-Augmented Generation

Official implementation of **GBRAG**, a hierarchical GraphRAG framework that
organizes a corpus into a granular-ball hierarchy graph — A-level factual
propositions, E-level entity balls, and D-level domain balls, connected by
cross-ball boundary edges — and retrieves through the hierarchy with a
dual-pool evidence context assembly.

The framework is evaluated on multi-hop and long-document QA benchmarks and
is compared against flat text-RAG baselines (BM25 / dense top-k) under the
same generation and evaluation protocol.

## Overview

```
raw corpus chunks
      |
      v
[T5 propositionizer]  ----->  A-level propositions (atomic facts + entities/tags/time)
      |
      v
[offline graph builder]  --->  E balls (entity clusters)
                         --->  D balls (domain splits inside E balls)
                         --->  edges (boundary-bridge propositions between E balls)
                         --->  vector index (ChromaDB, E/D/A levels)
      |
      v
[online retrieval pipeline]  (Graph_Rag_pipeline)
  chunk gate -> entity extraction -> D-ball recall (+entity bonus)
  -> in-ball A-proposition scoring -> seed E balls -> edge-activated
  bounded shortest-path search -> dual-pool context assembly
  (ordered path segments + supplementary evidence pool + bridge propositions
   + E-level summaries + global fallback)
      |
      v
  LLM answer generation  ->  EM / F1 / NLI / ER / EP / CQT / latency metrics
```

## Repository structure

```
.
├── main.py                        # entry point: graph building + QA evaluation
├── graph_builder.py               # offline granular-ball graph construction
├── graph_retriever.py           # online retrieval pipeline (dual-pool context)
├── evaluator.py                   # metrics (EM/F1/NLI/ER/EP/CQT) + benchmark runner
├── experiment_config.py           # validated experiment / ablation registry
├── model.py                       # LLM / embedding / propositionizer loading
├── prompt.py                      # all LangChain prompt templates
├── make_pro.py                    # document -> proposition extraction (stage 1)
├── proposition_utils.py           # concurrent LLM metadata extraction, entity normalization
├── naive_rag.py                   # BM25 / dense text-RAG baselines
├── run_variants.py                # P1/P2 variant experiments (edge & clustering swaps)
├── standard_metrics.py            # GraphRAG-standard strict EM/F1
├── recompute_standard_metrics.py  # offline metric recomputation + error attribution
├── summarize_experiment.py        # timing summaries with bootstrap CIs
├── chunk_func.py                  # pre-chunked corpus loading / re-splitting
├── file_path.py                   # all path conventions
├── dataset_map.py                 # dataset name registry
├── edge_config.py                 # with-edges / no-edges switch
├── json_tool.py                   # JSONL helpers + QA benchmark loader
├── scripts/
│   └── run_experiment.sh          # end-to-end run example
├── requirements.txt
└── .env.example                   # annotated list of all environment variables
```

## Installation

```bash
# Python 3.10+ recommended.
python -m venv .venv && source .venv/bin/activate

# Install PyTorch for your CUDA version first, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Model preparation

Three model roles are required. All paths are configurable via environment
variables (see `.env.example`); defaults expect the following layout inside
the repository root:

| Role            | Default location                                     | Source |
|-----------------|------------------------------------------------------|--------|
| Embedding       | `all-MiniLM-L6-v2/`                                   | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Propositionizer | `chentong00/propositionizer-wiki-flan-t5-large/`      | [chentong00/propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large) |
| Generator LLM   | via Ollama (`qwen2.5:14b` by default) **or** local HF weights | see below |

```bash
# embedding model
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir all-MiniLM-L6-v2
# propositionizer (used both to extract propositions and to compute ER/EP/CQT)
huggingface-cli download chentong00/propositionizer-wiki-flan-t5-large \
    --local-dir chentong00/propositionizer-wiki-flan-t5-large
```

### Generator LLM backends

**Option A — Ollama (default, easiest).** Serve any Ollama-compatible chat
model and point the framework at it. The paper's experiments use an
instruction-tuned 7B/14B-class model with greedy decoding (`temperature=0`).

```bash
ollama pull qwen2.5:14b
OLLAMA_HOST=0.0.0.0:8080 ollama serve            # expose on port 8080
export GBRAG_LLM_BASE_URL=http://localhost:8080  # default already 8080
```

Any OpenAI-compatible endpoint works as well: run `ollama serve` (or vLLM,
TGI, LMDeploy, ...) and set `GBRAG_LLM_BASE_URL` to its URL. The default model
name is `qwen2.5:14b`; point `GBRAG_LLM_MODEL` at any tag served by your
endpoint (or serve your model under the same name,
`ollama cp your-model qwen2.5:14b`).

**Option B — local Transformers weights.**

```bash
export GBRAG_LLM_BACKEND=transformers
export GBRAG_LLM_MODEL_PATH=/path/to/your/hf-model   # must contain config.json
# optional VRAM controls:
# export GBRAG_LLM_DTYPE=float16
# export GBRAG_LLM_DEVICE_MAP=auto
# export GBRAG_LLM_MAX_MEM=16,16
```

A GPU is **required** for the propositionizer (stage 1); CPU fallback is
explicitly opt-in (`GBRAG_ALLOW_CPU=1`).

## Datasets

The processed benchmark files used in our experiments are **included** in this
repository under `dataset/` (1,000 QA items and the corresponding passage
corpus per dataset; ~20 MB in total). The raw sources are the public
benchmarks: HotpotQA (hotpotqa.github.io), 2WikiMultihopQA
(github.com/Alab-NII/2wikimultihopqa), PubMedQA
(github.com/pubmedqa/pubmedqa), and GovReport
(github.com/armancohan/long-summarization); the GovReport-Geopolitics subset
follows the pre-processing of prior long-document RAG work.

Expected layout:

```
dataset/
└── <DatasetName>/                      # must match dataset_map.py
    ├── corpus_<name>.json               # list of passage chunks (see below)
    ├── qa_benchmark_<name>.json        # list of QA items (see below)
    └── first_step_output/              # auto-created caches (not in repo)
```

Registered names (`dataset_map.py`): `GovReport_Geopolitics`, `HotpotQA`,
`2WikiMultiHopQA`, `PubMed_Medical`.

QA item fields used by the evaluator:

```json
{
  "_id": "sample-id",
  "question": "...",
  "answer": "short answer" ,
  "type": "bridge|comparison|...",
  "domain": "dataset name",
  "source_chunk_id": 0,
  "context_file": "optional",
  "supporting_sentences": ["gold evidence sentence 1", "..."]
}
```

`answer` may be a string or a list of acceptable answers;
`supporting_sentences` feeds the evidence-recall/precision metrics.

Corpus chunk fields:

```json
{
  "chunk_id": 0,
  "text": "passage text ...",
  "source_doc_id": "id of the source document",
  "source_doc_chunk_id": 0
}
```

To run on a different corpus, convert it into this layout and register the
file names in `dataset_map.py`.

## Running

```bash
# 0) environment (see .env.example for the full list)
export GBRAG_CUDA_DEVICES=0
export GBRAG_DATASETS=HotpotQA
export GBRAG_BATCH_SIZE=200

# 1) graph building only (proposition extraction + hierarchy + edges + index)
GBRAG_STAGE=build python main.py

# 2) QA evaluation (resumes automatically per sample_id)
python main.py
```

Outputs are written under `GBRAG_OUTPUT_ROOT` (default `outputs/`):

- `new_ball_predictions_<experiment_id>.jsonl` — per-sample prediction + metrics
- `path_activation_metrics_<experiment_id>.jsonl` — retrieval trace per sample
- `timing_metrics_<experiment_id>.jsonl` — per-stage latency
- `new_ball_metrics_summary_<experiment_id>.json` — running aggregate summary
- `qa_case_log_<experiment_id>.txt` — human-readable case log

Or simply `bash scripts/run_experiment.sh 0 HotpotQA`.

### Baselines

```bash
# dense / bm25 text-RAG with identical metrics
GBRAG_RAG_MODE=dense GBRAG_RAG_TOP_K=10 python naive_rag.py
GBRAG_RAG_MODE=bm25 python naive_rag.py
```

### Ablation and variant runs

```bash
# single-module ablations (see .env.example for the full switch list)
GBRAG_V18_NO_PATH=1 GBRAG_OUTPUT_ROOT=outputs_ablate_no_path python main.py

# flat-proposition retrieval baseline
GBRAG_V18_FLAT=1 GBRAG_OUTPUT_ROOT=outputs_flat python main.py

# P1/P2 variants
GBRAG_VARIANT=edge_entity     python run_variants.py
GBRAG_VARIANT=edge_similarity python run_variants.py
GBRAG_VARIANT=kmeans          python run_variants.py
```

### Analysis utilities

```bash
# timing summary with bootstrap 95% CIs
python summarize_experiment.py outputs/<Dataset>/timing_metrics_DCB_v18_dual_pool.jsonl

# offline metric recomputation / error attribution over saved predictions
python recompute_standard_metrics.py --help
```

## Environment variables

All behavior is controlled through `GBRAG_*` environment variables; the
annotated, complete list is in [`.env.example`](.env.example). Key groups:

- **Data / batching**: `GBRAG_DATASETS`, `GBRAG_BATCH_SIZE`, `GBRAG_OUTPUT_ROOT`, `GBRAG_STAGE`
- **LLM**: `GBRAG_LLM_BACKEND`, `GBRAG_LLM_BASE_URL`, `GBRAG_LLM_MODEL`, `GBRAG_LLM_MODEL_PATH`, `GBRAG_LLM_DTYPE`, `GBRAG_LLM_DEVICE_MAP`, `GBRAG_LLM_MAX_MEM`, `GBRAG_LLM_WORKERS`
- **Devices**: `GBRAG_CUDA_DEVICES`, `GBRAG_EMBEDDING_DEVICE`, `CONTEXT_METRIC_DEVICE`
- **Retrieval hyper-parameters**: `GBRAG_V17_TOP_B`, `GBRAG_V11_SCORE_KEEP`, `GBRAG_V11_FINAL_KEEP`, `GBRAG_V17_MAX_C_SEEDS`, `GBRAG_V11_MAX_PATH_HOPS`, `GBRAG_V11_MAX_PATH_COST`, `GBRAG_V17_MIN_EDGE_SIM`, ...
- **Single-module ablations**: `GBRAG_V18_NO_CHUNK_GATE`, `GBRAG_V18_NO_ENTITY_C`, `GBRAG_V18_NO_DOMAIN_BONUS`, `GBRAG_V18_NO_PATH`, `GBRAG_V18_NO_FALLBACK`, `GBRAG_V18_NO_SUPP`, `GBRAG_V18_SUPP_ONLY`, `GBRAG_V18_NO_BRIDGE`, `GBRAG_V18_NO_C_SUMMARY`, `GBRAG_V18_FLAT`

## Reproducibility notes

- Full-pipeline experiments use greedy decoding (`GBRAG_LLM_SAMPLE=0`,
  `temperature=0`); sampling is available only for ablations.
- Graph building and QA evaluation are fully cached and resumable; deleting
  `dataset/<name>/first_step_output/` forces a rebuild.
- The default experiment id is `DCB_v18_dual_pool` (see `experiment_config.py`).

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{gbrag2027,
  title     = {GBRAG: Granular-Ball Hierarchical Retrieval-Augmented Generation},
  author    = {Anonymous},
  booktitle = {Proceedings of The Web Conference (WWW)},
  year      = {2027}
}
```
