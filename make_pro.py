import gc
import os, re
import torch
import json
from file_path import File_path
import ast
from model import BallModel
from typing import List
from chunk_func import load_pre_chunked_dataset, split_loaded_chunks_for_prop
from json_tool import start_json_list_file, append_json_item, finish_json_list_file
from proposition_utils import llm_abstract_concurrent, normalize_entities_rule_based


def parse_t5_propositions(generated_text: str) -> List[str]:
    """
    Robustly parse the proposition list produced by the T5 propositionizer.
    Supports multiple formats: ["...", "..."], "- xxx" lines, and plain text lines.
    """
    results = []
    text = generated_text.strip()

    # Attempt 1: the whole output is a Python list literal.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except:
            pass

    # Attempt 2: parse line by line, handling "- " bullets or numeric prefixes.
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading markers, numeric prefixes, and quotes.
        line = re.sub(r"^[-•*]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip("\"'").strip()
        if line:
            results.append(line)

    return results


def chunk_to_proposition(
    chunks: List[dict],
    tokenizer,
    model,
    max_new_tokens: int,
    device: str,
    output_filename: str,
) -> List[dict]:
    propositions = []
    start_json_list_file(output_filename)
    is_first_item = True
    pro_set = set()

    for item in chunks:
        chunk_id = item["id"]
        current_group = {
            "id": chunk_id,
            "propositions": [],
        }

        for chunk_text in item["chunks"]:
            prompt = (
                "Extract all independent factual propositions from the following text: "
                f"{chunk_text}"
            )
            model_input = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            output = model.generate(
                **model_input,
                max_length=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            parsed_props = parse_t5_propositions(generated_text)


            for proposition_text in parsed_props:
                if proposition_text not in pro_set:
                    pro_set.add(proposition_text)
                    current_group["propositions"].append(
                        {
                            "text": proposition_text,
                            "source_chunk": chunk_text,
                        }
                    )

        propositions.append(current_group)
        is_first_item = append_json_item(output_filename, current_group, is_first_item)
        print(f"[OK] chunk_id {chunk_id}: pure propositions extracted and saved.")

    finish_json_list_file(output_filename)
    return propositions

def pro_abstract(
    dataset_path: str,
    dataset_name,
    model: BallModel,
    file_path: File_path,
    embedding_model=None,
    llm=None,

):
    # Reuse models created by main.py instead of loading duplicate copies.
    if embedding_model is None:
        embedding_model = model._get_embedding_model()
    if llm is None:
        llm = model._get_llm_model()
    print(f"\n{'=' * 20} Processing dataset: {dataset_name} {'=' * 20}")

    paths = {
        "dataset_dir": dataset_path,
        "chunk_path": file_path._get_chunk_path(dataset_name),
        "pure_props_path": file_path._get_pure_propositions_path(dataset_name),
        "propositions_data_path": file_path._get_propositions_data_path(dataset_name)
    }

    if not os.path.exists(paths["chunk_path"]):
        raise FileNotFoundError(f"Chunk file not found: {paths['chunk_path']}")

    print(f"[Local] Found chunk file {paths['chunk_path']}; loading raw chunks...")
    # Raw text chunks.
    chunks = load_pre_chunked_dataset(paths["chunk_path"])
    # Smaller chunks used for proposition extraction.
    prop_chunks = split_loaded_chunks_for_prop(chunks, 1000, 30)

    if os.path.exists(paths["pure_props_path"]):
        print(f"[Local] Found cache {paths['pure_props_path']}; skipping T5 extraction...")
    else:
        print("[T5] First run: extracting pure propositions with the local T5 model...")
        proposition_tool = model._get_pro_model()
        try:
            chunk_to_proposition(
                prop_chunks,
                proposition_tool["tokenizer"],
                proposition_tool["model"],
                1024,
                proposition_tool["device"],
                paths["pure_props_path"],
            )
        finally:
            # T5 is no longer needed after proposition extraction.
            del proposition_tool
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load pure propositions.
    with open(paths["pure_props_path"], "r", encoding="utf-8") as f:
        pure_propositions = json.load(f)
    # Flatten: expand the chunk-grouped propositions into a single list.
    flat_propositions = []
    for item in pure_propositions:
        flat_propositions.extend(item["propositions"])

    print(f"[Stats] {len(flat_propositions)} propositions await metadata extraction")
    #---------------------------------
    # (Metadata extraction cache handling is integrated below.)
    #---------------------------------
    entity_map_path = file_path._get_entity_map_path(dataset_name)

    if os.path.exists(paths["propositions_data_path"]):
        print(
            f"[Local] Found cache "
            f"{paths['propositions_data_path']}; skipping LLM metadata extraction..."
        )

        with open(
                paths["propositions_data_path"],
                "r",
                encoding="utf-8",
        ) as f:
            proposition_key_lists = json.load(f)

        # When both propositions_data and entity_map exist, proposition
        # metadata and entity normalization are already complete: reuse the
        # cache and skip re-normalization, de-duplication, and file overwrites.
        if os.path.exists(entity_map_path):
            print(
                "[Cache] Both propositions_data and entity_map exist; "
                "reusing the complete cache without re-normalization..."
            )
            return proposition_key_lists

        print(
            "[Warn] propositions_data exists but entity_map is missing; "
            "continuing with entity normalization..."
        )

    else:
        print("[LLM] First run: extracting proposition metadata...")

        proposition_key_lists = llm_abstract_concurrent(
            paths["pure_props_path"],
            llm,
            embedding_model,
            None,
            paths["propositions_data_path"],
            max_workers=int(os.environ.get("GBRAG_LLM_WORKERS", "4")),
        )
    # ---------------------------------

    # --- Entity normalization (cached) ---

    if os.path.exists(entity_map_path):
        print(f"[Local] Found entity-map cache {entity_map_path}; skipping normalization...")
        with open(entity_map_path, "r", encoding="utf-8") as f:
            reverse_map = json.load(f)  # {canonical_name: [variant list]}

        # Build the reverse lookup table (variant -> canonical name).
        map_dic_lookup = {}
        for canonical, variants in reverse_map.items():
            for variant in variants:
                map_dic_lookup[variant] = canonical

        # Apply the cached mapping to the entity_list of each proposition.
        for pro in proposition_key_lists:
            if pro["entity_list"] and isinstance(pro["entity_list"], list):
                new_entity_list = []
                for entity in pro["entity_list"]:
                    clean_entity = entity.strip()
                    map_name = map_dic_lookup.get(clean_entity, clean_entity)
                    new_entity_list.append(map_name)
                pro["entity_list"] = list(set(new_entity_list))
    else:
        print("[LLM] First run: normalizing entities...")
        proposition_key_lists, reverse_map = normalize_entities_rule_based(
            proposition_key_lists, embedding_model
        )

        # Save the normalization map (format: {canonical_name: [variant list]}).
        os.makedirs(os.path.dirname(entity_map_path), exist_ok=True)
        with open(entity_map_path, "w", encoding="utf-8") as f:
            json.dump(reverse_map, f, ensure_ascii=False, indent=2)
        print(f"[OK] Entity-normalization map saved to {entity_map_path}")

    # Save the normalized propositions (both on first run and after caching).
    with open(paths["propositions_data_path"], "w", encoding="utf-8") as f:
        json.dump(proposition_key_lists, f, ensure_ascii=False, indent=2)
    print(f"[OK] Entity normalization done; updated {paths['propositions_data_path']}")

    print(f"[OK] Proposition extraction finished: {paths['propositions_data_path']}")
    return proposition_key_lists
