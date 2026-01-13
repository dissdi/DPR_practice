import pandas as pd

passages_path = 'data/corpus/passages.jsonl'
data = []
with open(passages_path, 'r', encoding='utf-8') as f: # read jsonl file
    i = 0
    for line in f:
        print(line)
        if i == 10:
            break
        i += 1
