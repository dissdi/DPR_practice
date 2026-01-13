import csv
import json
from tqdm import tqdm
from pathlib import Path

tsv_path = "data/downloads/data/wikipedia_split/psgs_w100.tsv" # for full input psgs_w100.tsv
passages_path = 'data/corpus/passages.jsonl'

def tsv_to_jsonl(tsv_path: str, jsonl_path: str) -> None:
    tsv_path = str(Path(tsv_path))
    jsonl_path = str(Path(jsonl_path))

    print("open files")
    with open(tsv_path, "r", encoding="utf-8", newline="") as fin, \
         open(jsonl_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        required = {"id", "text", "title"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"TSV header must include {sorted(required)}. Got: {reader.fieldnames}")

        for row in tqdm(reader, desc="Converting TSV -> JSONL"):
            # Clean/normalize
            rid = row.get("id", "").strip()
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()

            if rid == "" and text == "" and title == "":
                continue  # skip empty lines

            obj = {"id": rid, "text": text, "title": title}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved: {jsonl_path}")

if __name__ == "__main__":
    tsv_to_jsonl(tsv_path, passages_path)
