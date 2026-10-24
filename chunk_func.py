import json
from langchain_text_splitters import NLTKTextSplitter
from typing import List

def load_pre_chunked_dataset(chunk_path: str) -> List[dict]:
    with open(chunk_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    chunks = []
    for item in raw_chunks:
        chunk_text = item.get("text", "")
        chunk_id = item.get("chunk_id")

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            continue

        chunks.append(
            {
                "id": chunk_id,
                "chunks": [chunk_text.strip()],
            }
        )

    return chunks

def split_loaded_chunks_for_prop(chunks, chunk_size=1000, chunk_overlap=30):
    splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for item in chunks:
        full_text = " ".join(item["chunks"])
        sub_chunks = [t.strip() for t in splitter.split_text(full_text) if t.strip()]
        out.append({"id": item["id"], "chunks": sub_chunks or [full_text]})
    return out


def chunks_text(chunk_list: List[dict]) -> List[str]:
    chunk_str_list = []

    for chunk_item in chunk_list:
        full_text = " ".join([chunk.strip() for chunk in chunk_item["chunks"] if chunk.strip()])
        chunk_str_list.append(full_text)

    return chunk_str_list




