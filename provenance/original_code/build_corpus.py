"""Plain-text newspaper corpus for the trigram baseline, the vulnerability statistic and
the near-miss probe. Lines sharing an 8-word sequence with an evaluation sentence are dropped.

    python code/build_corpus.py biglam/hmd_newspapers corpora/hmd.txt
    python code/build_corpus.py PleIAs/US-PD-Newspapers corpora/uspd.txt
"""
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 50_000_000
K = 8


def normalise(text):
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def shingles(words):
    return {" ".join(words[i:i + K]) for i in range(max(1, len(words) - K + 1))}


def main(dataset, out):
    excluded = set()
    for f in ("benchmark.csv", "external.csv"):
        for sentence in pd.read_csv(ROOT / "data" / f).sentence:
            excluded |= shingles(normalise(sentence).split())
    stream = load_dataset(dataset, split="train", streaming=True).shuffle(seed=20260825, buffer_size=1000)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with open(out, "w") as fh:
        for example in stream:
            for line in str(example["text"]).splitlines():
                line = normalise(line)
                if line and not shingles(line.split()) & excluded:
                    fh.write(line + "\n")
                    n_tokens += line.count(" ") + 1
            if n_tokens >= MAX_TOKENS:
                break


if __name__ == "__main__":
    main(*sys.argv[1:3])
