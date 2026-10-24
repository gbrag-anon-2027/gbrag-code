DATASET_KIND = [
    "GovReport_Geopolitics",
    "HotpotQA",
    "2WikiMultiHopQA",
    "PubMed_Medical",
]

DATASET_MAP = {
    "GovReport_Geopolitics": {
        "chunk_name": "corpus_govreport.json",
        "qa_name": "qa_benchmark_govreport.json",
    },
    "HotpotQA": {
        "chunk_name": "corpus_hotpotqa.json",
        "qa_name": "qa_benchmark_hotpotqa.json",
    },
    "2WikiMultiHopQA": {
        "chunk_name": "corpus_2wikimultihopqa.json",
        "qa_name": "qa_benchmark_2wikimultihopqa.json",
    },
    "PubMed_Medical": {
        "chunk_name": "corpus_pubmed.json",
        "qa_name": "qa_benchmark_pubmed.json",
    },
}
