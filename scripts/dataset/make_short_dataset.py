import os
import random

IN_PATH = r"data\downloads\data\wikipedia_split\psgs_w100.tsv"
OUT_NAME = "psgs_short.tsv"
K = 1_000_000
SEED = 20260113

def reservoir_sample_indices(n_stream, k, rng):
    sample = []
    for i in n_stream:
        if i < k:
            sample.append(i)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = i
    return sample

def main():
    out_dir = os.path.dirname(IN_PATH) or "."
    out_path = os.path.join(out_dir, OUT_NAME)

    rng = random.Random(SEED)

    # ---------- 1st pass: count & sample indices ----------
    print(f"[1/2] Sampling indices from: {IN_PATH}")
    with open(IN_PATH, "r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            raise RuntimeError("Input TSV is empty (no header).")

        # stream indices for data lines
        def index_stream():
            i = 0
            for _line in f:
                yield i
                i += 1

        sampled = reservoir_sample_indices(index_stream(), K, rng)
        total_rows = len(sampled) if sampled else 0

    if total_rows == 0:
        raise RuntimeError("No data rows found after header.")

    sampled_set = set(sampled)
    print(f"[INFO] Selected rows: {len(sampled_set):,} (seed={SEED})")

    # ---------- 2nd pass: write selected rows ----------
    print(f"[2/2] Writing to: {out_path}")
    written = 0
    with open(IN_PATH, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:

        header2 = fin.readline()
        fout.write(header2)

        idx = 0
        for line in fin:
            if idx in sampled_set:
                fout.write(line)
                written += 1
                # optional: speed up by removing already-written indices
                sampled_set.remove(idx)
                if written % 100000 == 0:
                    print(f"  ...written {written:,}")
                if not sampled_set:  # all picked rows written
                    break
            idx += 1

    print(f"[DONE] Wrote {written:,} rows (+ header) -> {out_path}")

if __name__ == "__main__":
    main()
