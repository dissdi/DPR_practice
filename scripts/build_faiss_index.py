import numpy as np
import faiss

# due to environment issue, run on another conda environment

passage_read_path = 'data/corpus/embeddings/passages_emb.npy'
query_read_path = 'data/corpus/embeddings/queries_emb.npy'
write_path = 'data/corpus/embeddings/passages.faiss'

passages = np.load(passage_read_path)
queries = np.load(query_read_path)

d = passages.shape[1] # database dimension
nb = passages.shape[0] # database size
nq = queries.shape[0] # num of queries


passages = np.asarray(passages, dtype=np.float32, order="C")
queries  = np.asarray(queries,  dtype=np.float32, order="C")

print(d, nb, nq)
res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)   # set res on gpu
gpu_index.add(passages)
print(f"Number of vectors in index: {gpu_index.ntotal}")

# print("FAISS test")
# k = 10 
# print("FAISS search start...")
# D, I = gpu_index.search(queries, k)
# print()

print("save on file")
cpu_index = faiss.index_gpu_to_cpu(gpu_index)   # index res should be loaded on cpu to do write job
faiss.write_index(cpu_index, write_path)