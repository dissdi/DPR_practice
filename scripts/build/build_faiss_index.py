import numpy as np
import faiss
import json
import csv

passage_read_path = 'data/corpus/embeddings/passages_emb.npy'
write_path = 'data/corpus/embeddings/passages.faiss'
passage_jsonl_path = r'data\downloads\data\wikipedia_split\psgs_w100.tsv'

passages = np.load(passage_read_path)
pids = []
with open(passage_jsonl_path, 'r', newline='', encoding='utf-8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    first = next(reader, None)
    if first is not None:
        if first and first[0].isdigit():
            pids.append(int(first[0]))
    for row in reader:
        pids.append(int(row[0]))
pids = np.asarray(pids, dtype=np.int64)

d = passages.shape[1] # database dimension
nb = passages.shape[0] # database size


passages = np.asarray(passages, dtype=np.float32, order="C")

print(d, nb)
quantizer  = faiss.IndexFlatIP(d)
cpu_index = faiss.IndexIDMap2(quantizer)
cpu_index.add_with_ids(passages, pids)

# print("FAISS test")
# k = 10 
# print("FAISS search start...")
# D, I = gpu_index.search(queries, k)
# print()

print("save to file")
faiss.write_index(cpu_index, write_path)