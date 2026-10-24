import json

def start_json_list_file(output_filename: str):
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("[\n")


def append_json_item(output_filename: str, item: dict, is_first_item: bool) -> bool:
    with open(output_filename, "a", encoding="utf-8") as f:
        if not is_first_item:
            f.write(",\n")
        f.write(json.dumps(item, ensure_ascii=False, indent=4))
    return False

def finish_json_list_file(output_filename: str):
    with open(output_filename, "a", encoding="utf-8") as f:
        f.write("\n]")

def load_qa_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




