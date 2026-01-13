import faiss
import json
import torch
import os
import numpy as np
from tqdm import tqdm
from model.DPRModel import DPRModel
    
def get_passage_list(passages_jsonl_path):
    passage_list = []
    with open(passages_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "")
            text = item.get("text", "")
            passage_list.append(f"{title}, {text}")
    return passage_list

def append_json_file(new_data, file_path):
    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        file_data = []
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = json.load(file)
            
    file_data.append(new_data)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(file_data, file, indent=4)
        
faiss_path = "data/corpus/embeddings/passages.faiss"
question_path = "data/corpus/dev.jsonl"
passages_path = "data/corpus/passages.jsonl"
checkpoint_path = "checkpoints/epoch_002"
device = torch.device("cuda")

print("Load model")
model = DPRModel()
model.load_checkpoint(checkpoint_path)
model.eval()
model.to(device)

print("Load faiss")
cpu_index = faiss.read_index(faiss_path)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

passage_list = get_passage_list(passages_path)

ks = [10, 20, 100]
max_length = 128
n_question = 0
hit10 = 0
hit20 = 0
hit100 = 0

print("Offline retrieving...")
# ----------------Offline-----------------
with open(question_path, 'r') as file:
    pbar = tqdm(file, desc="retrieving")
    for line in pbar:
        n_question += 1
        json_object = json.loads(line)
        q_text = json_object["question"]
        
        q_tokens = model.tokenize(q_text, None, max_length)
        q_tokens = {m: v.to(device) for m, v in q_tokens.items()}
        with torch.no_grad():
                q_vec = model.encode_questions(q_tokens)
                q_vec = q_vec.detach().cpu().numpy().astype(np.float32)
                q_vec = np.ascontiguousarray(q_vec)
                D, I = gpu_index.search(q_vec, 100)
        pid = int(json_object["positive_passage_ids"][0])
        for k in ks:
            for j in range(k):
                row_idx = int(I[0][j])
                if row_idx == pid:
                    match k:
                        case 10:
                            hit10 += 1
                        case 20:
                            hit20 += 1
                        case 100:
                            hit100 += 1
                    break
                
        if n_question % 100 == 0:
            pbar.set_postfix({"Recall 10":hit10/n_question, "Recall 20":hit20/n_question, "Recall 100":hit100/n_question})
            
print("Retrieving complete")

print("save statistics")
# inference statistic save
statistic = {
    "recall_10": hit10/n_question,
    "recall_20": hit20/n_question,
    "recall_100": hit100/n_question
}
print(statistic)
append_json_file(statistic, 'statistics/statistic.json')

# ----------------Online-------------------
# while(True):
#     q_text = input("Q> ")
#     if q_text=="break" or q_text=="exit":
#         break
#     q_tokens = model.tokenize(q_text, None, max_length)
#     q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
#     with torch.no_grad():
#         q_vec = model.encode_questions(q_tokens)
#         q_vec = q_vec.detach().cpu().numpy().astype(np.float32)
#         q_vec = np.ascontiguousarray(q_vec)
#         D, I = gpu_index.search(q_vec, k)
    
#     for j in range(len(I[0])):
#         score = float(D[0][j])
#         row_idx = int(I[0][j])
#         print(f"rank: {j+1}, score: {score}, row_idx: {row_idx}, text: {passage_list[row_idx][:200]}")