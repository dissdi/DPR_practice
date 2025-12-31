import json

train_path = 'data/nq/train.jsonl'
passages_path = 'data/nq/passages.jsonl'

def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f: # read jsonl file
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data

train_data = load_jsonl(train_path)
passages = load_jsonl(passages_path)
# {"id": 0, "question": "who sings does he love me with reba", "positive_passage_ids": ["11828866"], "negative_passage_ids": ["14525568", "11828871", "11828872", "11828869", "11869749", "9446572", "12864394"]}
pid = train_data[0]["positive_passage_ids"][0]
pid_int = int(pid)
print(len(passages))
print(pid_int)
print(passages[pid_int])