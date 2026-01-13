import json
from pathlib import Path

def build_question_jsonl(
    input_json_path: str,
    output_train_jsonl_path: str,
    num_neg: int = 7,
    include_answers: bool = False,
    skip_if_insufficient_neg: bool = True,
) -> None:
    in_path = Path(input_json_path)
    out_path = Path(output_train_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) JSON 배열 로드
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # list[dict]

    written = 0
    skipped_no_pos = 0
    skipped_no_neg = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(data):
            q = sample.get("question")
            if not q:
                continue

            pos_list = sample.get("positive_ctxs") or []
            neg_list = sample.get("negative_ctxs") or []
            hard_list = sample.get("hard_negative_ctxs") or []

            # 2) Positive 선택: passage_id 기준 dedup 후, score 최대(없으면 첫 번째)
            best_pos_pid = None
            best_score = None
            seen_pos = set()

            for ctx in pos_list:
                pid = ctx.get("passage_id")
                if not pid or pid in seen_pos:
                    continue
                seen_pos.add(pid)

                score = ctx.get("score")
                # score가 있으면 최대 score 우선, 없으면 첫 번째를 임시로 잡아둠
                if best_pos_pid is None:
                    best_pos_pid = pid
                    best_score = score
                else:
                    if score is not None and (best_score is None or score > best_score):
                        best_pos_pid = pid
                        best_score = score

            if best_pos_pid is None:
                skipped_no_pos += 1
                continue

            # 3) Negative 후보 구성: hard 먼저, 부족하면 neg로 채움 (pos pid 제외, pid dedup)
            neg_pids = []
            seen_neg = set([best_pos_pid])

            for ctx in (hard_list + neg_list):
                pid = ctx.get("passage_id")
                if not pid or pid in seen_neg:
                    continue
                seen_neg.add(pid)
                neg_pids.append(pid)
                if len(neg_pids) >= num_neg:
                    break

            if len(neg_pids) < num_neg and skip_if_insufficient_neg:
                skipped_no_neg += 1
                continue

            # 4) 한 줄(row) 만들기
            row = {
                "id": idx,
                "question": q,
                "positive_passage_ids": [best_pos_pid],
                "negative_passage_ids": neg_pids[:num_neg],
            }
            if include_answers:
                row["answers"] = sample.get("answers") or []

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved train rows: {written} -> {out_path}")
    print(f"Skipped (no positive): {skipped_no_pos}")
    print(f"Skipped (insufficient negatives): {skipped_no_neg}")

if __name__ == "__main__":
    print("build train corpus")
    build_question_jsonl(
        input_json_path="data/downloads/data/retriever/nq-train.json",      
        output_train_jsonl_path="data/corpus/train.jsonl",
        num_neg=7,
        include_answers=False,
        skip_if_insufficient_neg=True,
    )
    print("build dev corpus")
    build_question_jsonl(
        input_json_path="data/downloads/data/retriever/nq-dev.json",      
        output_train_jsonl_path="data/corpus/dev.jsonl",
        num_neg=7,
        include_answers=False,
        skip_if_insufficient_neg=True,
    )
