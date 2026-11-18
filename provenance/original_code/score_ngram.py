"""In-domain trigram baseline.

For keyword k with left context (a, b) and right context (c, d):
    susp = -[log P(k | a b) + log P(c | b k) + log P(d | k c)]
with stupid backoff (alpha = 0.4) and an add-one unigram floor. Only the n-grams that
the evaluation windows need are counted, so the corpus is streamed once.

    python code/score_ngram.py data/benchmark.csv corpora/hmd.txt scores/benchmark
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from common import load

TOKEN = re.compile(r"\w+")
ALPHA = 0.4
MAX_VOCAB = 2_000_000


def windows(sentence, spans):
    for s, e in spans:
        left = TOKEN.findall(sentence[:s].lower())[-2:]
        right = TOKEN.findall(sentence[e:].lower())[:2]
        yield ["<s>"] * (2 - len(left)) + left + [sentence[s:e].lower()] + right + ["</s>"] * (2 - len(right))


def logp(w1, w2, w3, C, V, N):
    if C[(w1, w2, w3)] and C[(w1, w2)]:
        return math.log(C[(w1, w2, w3)] / C[(w1, w2)])
    if C[(w2, w3)] and C[(w2,)]:
        return math.log(ALPHA) + math.log(C[(w2, w3)] / C[(w2,)])
    return 2 * math.log(ALPHA) + math.log((C[(w3,)] + 1) / (N + V))


def main(data, corpus, out_dir):
    df = load(data)
    hits = [(rid, list(windows(s, sp))) for rid, s, sp in zip(df.id, df.sentence, df.spans)]
    needed = set()
    for _, ws in hits:
        for w in ws:
            for i in range(2, 5):  # trigrams ending at k, c and d
                t = tuple(w[i - 2:i + 1])
                needed |= {t, t[:2], t[1:], t[1:2], t[2:]}

    C, N, vocab = Counter(), 0, set()
    with open(corpus) as fh:
        for line in fh:
            t = ["<s>", "<s>"] + TOKEN.findall(line) + ["</s>", "</s>"]
            N += len(t) - 4
            for i in range(len(t)):
                for n in (1, 2, 3):
                    g = tuple(t[i:i + n])
                    if len(g) == n and g in needed:
                        C[g] += 1
                if len(vocab) < MAX_VOCAB:
                    vocab.add(t[i])
    V = len(vocab)

    rows = []
    for rid, ws in hits:
        scores = [-(logp(*w[0:3], C, V, N) + logp(*w[1:4], C, V, N) + logp(*w[2:5], C, V, N)) for w in ws]
        rows.append({"id": rid, "susp": max(scores) if scores else None})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "ngram-hmd.ngram.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
