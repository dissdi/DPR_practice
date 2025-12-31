import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import json
from tqdm import tqdm
from pathlib import Path

train_path = 'data/nq/train.jsonl'
passages_path = 'data/nq/passages.jsonl'
checkpoint_path = 'checkpoints'

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
    
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f: # read jsonl file
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def build_passage_text_map(passages_jsonl_path):
    passage_map = {}
    with open(passages_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pid = str(item["passage_id"])
            title = item.get("title", "")
            text = item.get("text", "")
            passage_map[pid] = f"{title} [SEP] {text}"
    return passage_map


train_data = load_jsonl(train_path)
passage_map = build_passage_text_map(passages_path)

model = DPRModel()
#model.load_checkpoint(checkpoint_path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model.train()
device = torch.device("cuda")
model.to(device)

batch = 16
epochs = 3
n_neg = 7 # number of negative
MAX_LENGTH = 128

for epoch in range(epochs):
    print(f"epoch {epoch}")
    pbar = tqdm(range(0, len(train_data), batch), desc="training")
    for i in pbar:
        optimizer.zero_grad()
        batch_samples = train_data[i:i+batch]
        questions = [sample["question"] for sample in batch_samples]
        pos_texts = [passage_map[sample["positive_passage_ids"][0]] for sample in batch_samples]
        neg_texts = []
        for sample in batch_samples:
            for n_pid in sample["negative_passage_ids"][:n_neg]:
                neg_texts.append(passage_map[n_pid])
        passages = pos_texts + neg_texts
        
        # checker
        # if i == 0:
        #     print(f"i={i} B={len(batch_samples)} Q={len(questions)} P={len(pos_texts)} N={len(neg_texts)} All={len(passages)}")        
        #     input("Enter to continue")
        
        q_tokens = model.tokenize(questions, MAX_LENGTH)
        p_tokens = model.tokenize(passages, MAX_LENGTH)
        q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
        p_tokens = {k: v.to(device) for k, v in p_tokens.items()}
        
        q_input = model.encode_questions(q_tokens)
        p_input = model.encode_passages(p_tokens)
        score = model.forward(q_input, p_input)
        labels = torch.arange(len(batch_samples), dtype=torch.long, device=device)
        
        loss = criterion(score, labels)
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        
    # checkpoint
    epoch_dir = Path("checkpoints") / f"epoch_{epoch:03d}"
    q_dir = epoch_dir / "q_encoder"
    p_dir = epoch_dir / "p_encoder"
    tok_dir = epoch_dir / "tokenizer"
    
    q_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    
    model.q_encoder.save_pretrained(str(q_dir))
    model.p_encoder.save_pretrained(str(p_dir))
    model.tokenizer.save_pretrained(str(tok_dir))    