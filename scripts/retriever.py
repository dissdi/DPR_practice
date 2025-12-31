import numpy as np
import faiss
import json
import torch
from transformers import BertTokenizer, BertModel

cfg = {
        "paths": {
            "faiss_path": "data/corpus/embeddings/passages.faiss",
            "passages_path": "data/nq/passages.jsonl",
            "query_emb_path": "data/corpus/embeddings/queries_emb.npy",
        },
        "retrieval": {
            "k": 10,
            "use_gpu_faiss": False,  
            "gpu_device": 0,
        },
        "encoding": {
            "realtime": False,        # if it's true ->  it is running on realtime
            "model_name": "bert-base-uncased",
            "max_length": 256,
            "device": "cuda",         # "cuda" / "cpu"
        },
        "output": {
            "save_results": False,
            "results_path": "data/retrieval/topk_results.jsonl",
            "show_text_chars": 300,   # output text length
        }
    }

tokenizer = BertTokenizer.from_pretrained(cfg["encoding"]["model_name"])
model = BertModel.from_pretrained(cfg["encoding"]["model_name"])
device = cfg["encoding"]["device"]
model.to(device)
model.eval()

def set_input(faiss_path, passages_path, query_emb_path, k, max_length, realtime):
    cfg["paths"]["faiss_path"] = faiss_path
    cfg["paths"]["passages_path"] = passages_path
    cfg["paths"]["query_emb_path"] = query_emb_path
    cfg["retrieval"]["k"] = k
    cfg["encoding"]["max_length"] = max_length
    cfg["encoding"]["realtime"] = realtime
    
def encode_query(query_text):
    with torch.no_grad():
        query_out = tokenizer(
            [query_text],
            padding="max_length",
            max_length=cfg["encoding"]["max_length"],
            truncation=True,
            return_tensors="pt"
        )
        query_out = {k: v.to(device) for k, v in query_out.items()}
        out = model(**query_out)
        qemb = out.pooler_output.detach().cpu().numpy().astype(np.float32) 
    return qemb

def load_queries(query_text=None): # return embedded query
    if cfg["encoding"]["realtime"]:
        return encode_query(query_text)
    return np.load(cfg["paths"]["query_emb_path"]).astype(np.float32, order="C")

def load_retriever():
    index = faiss.read_index(cfg["paths"]["faiss_path"])
    passages = []
    with open(cfg["paths"]["passages_path"], 'r', encoding='utf-8') as f: # read jsonl file
        for line in f:
            passages.append(json.loads(line))
    return passages, index

def search(index, qemb):
    return index.search(qemb, cfg["retrieval"]["k"])

def print_topk(k, passages, I, D):
    print("Query:", query_text)
    print()
    
    for rank in range(k):
        pid = int(I[0][rank])
        score = float(D[0][rank])

        title = passages[pid].get("title", "")
        text  = passages[pid].get("text", "")

        snippet_len = cfg["output"]["show_text_chars"]
        snippet = text[:snippet_len].replace("\n", " ")

        print(f"[{rank+1}] pid={pid}  score={score:.4f}")
        print("Title:", title)
        print("Text :", snippet)
        print("-" * 80)

if __name__=="__main__":
    query_text = "Who wrote the novel 1984?"
    cfg["encoding"]["realtime"] = True
    
    qemb = load_queries(query_text) # if query_text is not None -> its realtime
    passages, index = load_retriever()
    
    print(len(passages)==index.ntotal)
    # D, I = search(index, qemb)
    # print_topk(3, passages, I, D)