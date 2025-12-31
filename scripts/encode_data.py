import json
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from tqdm import tqdm

from transformers.utils import logging
logging.set_verbosity_error() # remove warning message


passage_read_path = 'data/nq/passages.jsonl'
query_read_path = 'data/nq/train.jsonl'

passage_write_path = 'data/corpus/embeddings/passages_emb.npy'
query_write_path = 'data/corpus/embeddings/queries_emb.npy'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()


def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f: # read jsonl file
        for line in f:
            json_object = json.loads(line)
            data.append(json_object)
    return data
        
passage_data = load_jsonl(passage_read_path)
query_data = load_jsonl(query_read_path)

title = [item.get("title","") for item in passage_data]
text = [item.get("text","") for item in passage_data]
question = [item.get("question","") for item in query_data]
MAX_LENGTH = 256
batch = 64

device = "cuda"
model.to(device)
model.eval()

passage_out = []
query_out = []


print("passage encoding...")
with torch.no_grad():
    for i in tqdm(range(0, len(title), batch), desc="passages"):
        bt = title[i:i+batch]
        bx = text[i:i+batch]
        passage_input = tokenizer(  # tokenize parsed data
            bt, bx,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in passage_input.items()}
        out = model(**inputs)
        passage_out.append(out.pooler_output.detach().cpu().numpy().astype(np.float32))
print("passage is encoded")


with torch.no_grad():
    print("query encoding...")
    for i in tqdm(range(0, len(question), batch), desc="queries"):
        bq = question[i:i+batch]
        query_input = tokenizer(
            bq,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in query_input.items()}
        out = model(**inputs)
        query_out.append(out.pooler_output.detach().cpu().numpy().astype(np.float32))
print("query is encoded")


passage_emb = np.concatenate(passage_out, axis=0)
query_emb   = np.concatenate(query_out, axis=0)

np.save(passage_write_path, passage_emb)  # write embeddings to write path
np.save(query_write_path, query_emb)