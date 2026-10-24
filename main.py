import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GBRAG_CUDA_DEVICES", "0"))
os.environ.setdefault("BALL_MAIN_DEVICE", "cuda:0")
os.environ.setdefault("CONTEXT_METRIC_DEVICE", "cuda:0")

import json
from model import BallModel
from file_path import File_path
from dataset_map import DATASET_MAP
from graph_builder import build_full_graph
from make_pro import pro_abstract
from evaluator import run_qa_benchmark_with_metrics
from json_tool import load_qa_benchmark
from edge_config import EdgeMode, set_edge_mode
from experiment_config import get_experiment_config


def main():
    os.environ["GBRAG_EXPERIMENT_ID"] = "DCB_v18_dual_pool"
    batch_size = int(os.environ.get("GBRAG_BATCH_SIZE", "50"))
    datasets = [x.strip() for x in os.environ.get("GBRAG_DATASETS", "GovReport_Geopolitics").split(",") if x.strip()]
    output_root = os.environ.setdefault("GBRAG_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"))
    stage = os.environ.get("GBRAG_STAGE", "all").lower()
    config = get_experiment_config()
    current = os.path.abspath(__file__)
    fp = File_path(current)
    model = BallModel(current)
    embedding = model._get_embedding_model(); llm = model._get_llm_model()
    set_edge_mode(EdgeMode.WITH_EDGES)
    for dataset_name in datasets:
        if dataset_name not in DATASET_MAP: raise ValueError(dataset_name)
        props_path = fp._get_propositions_data_path(dataset_name)
        if stage in ("all", "build") and not os.path.exists(props_path):
            print(f"Proposition cache not found; extracting propositions for {dataset_name}...")
            pro_abstract(fp._get_dataset_path(), dataset_name, model, file_path=fp,
                         embedding_model=embedding, llm=llm)
        if not os.path.exists(props_path):
            raise FileNotFoundError(
                f"Missing proposition cache: {props_path}. Run "
                "GBRAG_STAGE=build python main.py first, or place an existing "
                "propositions_data_<dataset>.json in that directory."
            )
        with open(props_path, encoding="utf-8") as f: propositions = json.load(f)
        graph_dir = fp._get_graph_chroma_db_path_with_edges(dataset_name)
        C, B, D, edges, db = build_full_graph(propositions, embedding, llm, graph_dir)
        if stage == "build": continue
        qa_path = fp._get_qa_path(dataset_name)
        os.environ["GBRAG_RAW_CORPUS_PATH"] = fp._get_chunk_path(dataset_name)
        qa_items = load_qa_benchmark(qa_path)
        run_qa_benchmark_with_metrics(qa_items=qa_items, qa_benchmark_path=qa_path,
            C_balls=C, B_balls=B, edges=edges, Propositions=propositions,
            llm=llm, embedding_model=embedding, db_helper=db, edge_propagation=True,
            experiment_config=config, max_new_samples=batch_size)


if __name__ == "__main__": main()
