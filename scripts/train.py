import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from pathlib import Path
import pickle

from model.DPRModel import DPRModel
from transformers.utils import logging
logging.set_verbosity_error()

train_path = 'data/corpus/train.jsonl'
passages_path = 'data/corpus/passages.jsonl'
map_path = 'data/corpus/passage_map.pkl'
checkpoint_path = 'checkpoints'
    
device = torch.device("cuda")    

def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f: # read jsonl file
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_passage_map(map_path):
    with open(map_path, "rb") as f:
        return pickle.load(f)

print("load train jsonl")
train_data = load_jsonl(train_path)

print("load passage map")
passage_map = load_passage_map(map_path)

print("load model")
model = DPRModel()
#model.load_checkpoint(checkpoint_path)
model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


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
        
        # queries
        questions = [sample["question"] for sample in batch_samples]
        
        # postive passages (title/text pair)
        pos_titles, pos_texts = [], []
        for sample in batch_samples:
            pos_pid = int(sample["positive_passage_ids"][0])
            t, x = passage_map[pos_pid]
            pos_titles.append(t)
            pos_texts.append(x)
        
        # negative passages (title/text pair)
        neg_titles, neg_texts = [], []
        for sample in batch_samples:
            for neg_pid in sample["negative_passage_ids"][:n_neg]:
                neg_pid = int(neg_pid)
                t, x = passage_map[neg_pid]
                neg_titles.append(t)
                neg_texts.append(x)
        
        passage_titles = pos_titles + neg_titles
        passage_texts = pos_texts + neg_texts
        
        # tokenize
        q_tokens = model.tokenize(questions, None, MAX_LENGTH)
        p_tokens = model.tokenize(passage_titles, passage_texts, MAX_LENGTH)
        
        q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
        p_tokens = {k: v.to(device) for k, v in p_tokens.items()}
        
        # encode
        q_input = model.encode_questions(q_tokens)
        p_input = model.encode_passages(p_tokens)
        
        score = model.forward(q_input, p_input)
        labels = torch.arange(len(batch_samples), dtype=torch.long, device=device)
        
        loss = criterion(score, labels)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
        
    # checkpoint save
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