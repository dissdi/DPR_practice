import json
from model.DPRModel import DPRModel
import numpy as np
import torch
from tqdm import tqdm
from transformers.utils import logging
import os
logging.set_verbosity_error() # remove warning message

passage_read_path = 'data/corpus/passages.jsonl'
query_read_path = 'data/corpus/train.jsonl'

passage_write_path = 'data/corpus/embeddings/passages_emb.npy'
query_write_path = 'data/corpus/embeddings/queries_emb.npy'

checkpoint_path = "checkpoints/epoch_002"
device = torch.device("cuda")

model = DPRModel()
model.load_checkpoint(checkpoint_path)
model.to(device)
model.eval()

MAX_LENGTH = 128
BATCH_SIZE = 64

def count_lines(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def iter_passage_batches(file_obj, batch_size: int):
    titles, texts = [], []
    for line in file_obj:
        if not line.strip():
            continue
        obj = json.loads(line)
        titles.append(obj.get("title", ""))
        texts.append(obj.get("text", ""))
        if len(titles) == batch_size:
            yield titles, texts
            titles, texts = [], []
    if titles:
        yield titles, texts

def iter_query_batches(file_obj, batch_size: int):
    questions = []
    for line in file_obj:
        if not line.strip():
            continue
        obj = json.loads(line)
        questions.append(obj.get("question", ""))
        if len(questions) == batch_size:
            yield questions
            questions = []
    if questions:
        yield questions

# --------- encode & save (memmap) ---------
def encode_passages_to_npy(model: DPRModel, in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    N = count_lines(in_path)
    if N == 0:
        raise RuntimeError(f"No lines found in {in_path}")
    print(f"[PASSAGES] N={N:,} -> {out_path}")

    with open(in_path, 'r', encoding='utf-8') as f, torch.no_grad():
        gen = iter_passage_batches(f, BATCH_SIZE)

        # 1) 첫 배치로 embed dim 확정 + 바로 기록 시작
        first = next(gen, None)
        if first is None:
            raise RuntimeError("Passage file is empty.")
        bt, bx = first

        passage_input = model.tokenize(bt, bx, MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in passage_input.items()}
        out = model.encode_passages(inputs).detach().cpu().numpy().astype(np.float32)

        dim = out.shape[1]
        mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, dim))

        idx = 0
        bs = out.shape[0]
        mm[idx:idx+bs] = out
        idx += bs

        pbar = tqdm(total=N, desc="passages")
        pbar.update(bs)

        # 2) 나머지 배치들 스트리밍 인코딩 -> memmap에 바로 저장
        for bt, bx in gen:
            passage_input = model.tokenize(bt, bx, MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in passage_input.items()}
            out = model.encode_passages(inputs).detach().cpu().numpy().astype(np.float32)

            bs = out.shape[0]
            mm[idx:idx+bs] = out
            idx += bs
            pbar.update(bs)

        pbar.close()
        mm.flush()

    # 안전 체크: 실제로 N개 채웠는지
    if idx != N:
        print(f"[WARN] wrote {idx:,} rows, expected {N:,}. (빈 줄/깨진 줄이 있었을 수 있어요.)")
    print("[PASSAGES] done")


def encode_queries_to_npy(model: DPRModel, in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    N = count_lines(in_path)
    if N == 0:
        raise RuntimeError(f"No lines found in {in_path}")
    print(f"[QUERIES] N={N:,} -> {out_path}")

    with open(in_path, 'r', encoding='utf-8') as f, torch.no_grad():
        gen = iter_query_batches(f, BATCH_SIZE)

        # 1) 첫 배치로 embed dim 확정 + 바로 기록 시작
        first = next(gen, None)
        if first is None:
            raise RuntimeError("Query file is empty.")
        bq = first

        query_input = model.tokenize(bq, None, MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in query_input.items()}
        out = model.encode_questions(inputs).detach().cpu().numpy().astype(np.float32)

        dim = out.shape[1]
        mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, dim))

        idx = 0
        bs = out.shape[0]
        mm[idx:idx+bs] = out
        idx += bs

        pbar = tqdm(total=N, desc="queries")
        pbar.update(bs)

        # 2) 나머지 배치 스트리밍 인코딩 -> memmap에 바로 저장
        for bq in gen:
            query_input = model.tokenize(bq, None, MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in query_input.items()}
            out = model.encode_questions(inputs).detach().cpu().numpy().astype(np.float32)

            bs = out.shape[0]
            mm[idx:idx+bs] = out
            idx += bs
            pbar.update(bs)

        pbar.close()
        mm.flush()

    if idx != N:
        print(f"[WARN] wrote {idx:,} rows, expected {N:,}. (빈 줄/깨진 줄이 있었을 수 있어요.)")
    print("[QUERIES] done")


# --------- main ---------
def main():
    model = DPRModel()
    model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    encode_passages_to_npy(model, passage_read_path, passage_write_path)
    encode_queries_to_npy(model, query_read_path, query_write_path)


if __name__ == "__main__":
    main()