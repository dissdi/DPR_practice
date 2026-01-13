import pickle
import json

passage_path = "data/corpus/passages.jsonl"
out_path = "data/corpus/passage_map.pkl"

def passage_text_map(passages_jsonl_path):
    passage_map = {}
    with open(passages_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            item = json.loads(line)
            pid = int(item["id"])
            text = item.get("text", "")
            title = item.get("title", "")
            passage_map[pid] = (title, text)
    return passage_map

def save_passage_map_pickle(passage_map, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(passage_map, f, protocol=pickle.HIGHEST_PROTOCOL)

pm = passage_text_map("data/corpus/passages.jsonl")
save_passage_map_pickle(pm, "data/corpus/passage_map.pkl")