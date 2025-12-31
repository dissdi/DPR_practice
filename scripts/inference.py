import torch.nn as nn
from transformers import BertTokenizer, BertModel
import faiss
import json
import torch
import numpy as np

class DPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.p_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_checkpoint(self, epoch_dir):
        self.q_encoder = BertModel.from_pretrained(f"{epoch_dir}/q_encoder")
        self.p_encoder = BertModel.from_pretrained(f"{epoch_dir}/p_encoder")
        self.tokenizer = BertTokenizer.from_pretrained(f"{epoch_dir}/tokenizer")        
        
    def tokenize(self, text, MAX_LENGTH):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        return tokens
        
    def encode_questions(self, q_input):
        return self.q_encoder(**q_input).last_hidden_state[:, 0, :]

    def encode_passages(self, p_input):
        return self.p_encoder(**p_input).last_hidden_state[:, 0, :]

    def forward(self, q_emb, p_emb):
        return q_emb @ p_emb.T
    
def get_passage_list(passages_jsonl_path):
    passage_list = []
    with open(passages_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            title = item.get("title", "")
            text = item.get("text", "")
            passage_list.append(f"{title}, {text}")
    return passage_list

faiss_path = "data/corpus/embeddings/passages.faiss"
passages_path = "data/nq/passages.jsonl"
checkpoint_path = "checkpoints/epoch_002"
device = torch.device("cuda")

model = DPRModel()
model.load_checkpoint(checkpoint_path)
model.eval()
model.to(device)

cpu_index = faiss.read_index(faiss_path)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

passage_list = get_passage_list(passages_path)

k = 10
max_length = 128
while(True):
    q_text = input("Q> ")
    if q_text=="break" or q_text=="exit":
        break
    q_tokens = model.tokenize(q_text, max_length)
    q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
    with torch.no_grad():
        q_vec = model.encode_questions(q_tokens)
        q_vec = q_vec.detach().cpu().numpy().astype(np.float32)
        q_vec = np.ascontiguousarray(q_vec)
        D, I = gpu_index.search(q_vec, k)
    
    for j in range(len(I[0])):
        score = float(D[0][j])
        row_idx = int(I[0][j])
        print(f"rank: {j+1}, score: {score}, row_idx: {row_idx}, text: {passage_list[row_idx][:200]}")