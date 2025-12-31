import json

path = "data/downloads/data/retriever/nq-dev.json"  

queries_path = 'data/nq/processed/queries_dev.jsonl'
passages_path = 'data/corpus/passages.jsonl'

def build_passage_corpus(input_path, out_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus_by_id = {}
    
    for item in data:
        pos = item.get("positive_ctxs")
        neg = item.get("negative_ctxs")
        hard = item.get("hard_negative_ctxs")
        
        for ctx in (pos + neg + hard):
            pid = ctx.get("passage_id")
            title = ctx.get("title")
            text = ctx.get("text")

            if not pid or title is None or text is None:
                continue

            if pid in corpus_by_id:
                continue

            corpus_by_id[pid] = {
                "passage_id": pid,
                "title": title,
                "text": text,
            }
            
    with open(out_dir, "w", encoding="utf-8") as f:
        for pid, p in corpus_by_id.items():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved {len(corpus_by_id)} passages -> {out_dir}")



if __name__ == "__main__":
    build_passage_corpus(
        input_path=path,      # 네 입력 JSON 배열 파일
        out_dir="data/nq/passages.jsonl" # 네 출력 코퍼스
    )