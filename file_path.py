from pathlib import Path
import os
from dataset_map import DATASET_MAP


# Directory names can be overridden with environment variables so that the
# repository layout stays flexible (e.g. models stored outside the repo).
EMBEDDING_MODEL_DIR = os.environ.get("GBRAG_EMBEDDING_MODEL_DIR", "all-MiniLM-L6-v2")
PROPOSITIONIZER_DIR = os.environ.get(
    "GBRAG_PROPOSITIONIZER_DIR",
    "chentong00/propositionizer-wiki-flan-t5-large",
)


class File_path():
    def __init__(self, current_file: str):
        self.current_file = current_file
        self.project_path = Path(self.current_file).parent.parent

    def _get_embedding_model_path(self):
        # Path of the sentence embedding model (a local directory or a model id).
        self.embedding_model_file = self.project_path / EMBEDDING_MODEL_DIR
        return self.embedding_model_file

    def _get_pro_model_path(self):
        # Path of the propositionizer model used to extract propositions.
        self.pro_model_file = self.project_path / PROPOSITIONIZER_DIR
        return self.pro_model_file

    def _get_dataset_path(self):
        # Path of the datasets root directory.
        self.dataset_file = self.project_path / "dataset"
        return self.dataset_file

    def _get_first_step_output_dir(self, dataset_name):
        """
        Return (and create if needed) the first_step_output directory for a dataset.
        """
        dataset_dir = self._get_dataset_path() / dataset_name
        output_dir = dataset_dir / "first_step_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_propositions_data_path(self, dataset_name: str) -> str:
        """
        Return the final proposition file path:
        .../dataset/{dataset_name}/first_step_output/propositions_data_{dataset_name}.json
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / f"propositions_data_{dataset_name}.json")

    def _get_pure_propositions_path(self, dataset_name: str) -> str:
        """
        Return the raw pure-proposition file path:
        .../dataset/{dataset_name}/first_step_output/pure_propositions_{dataset_name}.json
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / f"pure_propositions_{dataset_name}.json")

    def _get_chroma_db_path(self, dataset_name: str) -> str:
        """
        Return the dataset-specific chroma_db path (for A-level sentence-vector search).
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / "chroma_db")

    def _get_graph_chroma_db_path(self, dataset_name: str) -> str:
        """
        Return the dataset-specific graph_chroma_db path (for D/C/A granular-ball search).
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / "graph_chroma_db")

    def _get_chunk_path(self, dataset_name: str) -> str:
        """Resolve the chunk file name from DATASET_MAP automatically."""
        chunk_name = DATASET_MAP[dataset_name]["chunk_name"]
        dataset_dir = self._get_dataset_path() / dataset_name
        return str(dataset_dir / chunk_name)

    def _get_qa_path(self, dataset_name: str) -> str:
        """Resolve the QA benchmark file name from DATASET_MAP automatically."""
        qa_name = DATASET_MAP[dataset_name]["qa_name"]
        dataset_dir = self._get_dataset_path() / dataset_name
        return str(dataset_dir / qa_name)

    def _get_entity_map_path(self, dataset_name: str) -> str:
        """
        Return the entity-normalization mapping path:
        .../dataset/{dataset_name}/first_step_output/graph_chroma_db/entity_map.json
        """
        graph_chroma_dir = self._get_graph_chroma_db_path(dataset_name)
        return os.path.join(graph_chroma_dir, "entity_map.json")

    def _get_graph_chroma_db_path_with_edges(self, dataset_name: str) -> str:
        """
        graph_chroma_db path for the with-edges mode.
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / "graph_chroma_db_with_edges")

    def _get_graph_chroma_db_path_no_edges(self, dataset_name: str) -> str:
        """
        graph_chroma_db path for the no-edges mode.
        """
        output_dir = self._get_first_step_output_dir(dataset_name)
        return str(output_dir / "graph_chroma_db_no_edges")
