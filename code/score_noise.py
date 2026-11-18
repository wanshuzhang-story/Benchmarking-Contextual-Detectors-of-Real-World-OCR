"""Lexical-noise baseline: share of the sentence's alphabetic tokens (two or more letters,
keyword excluded) that are not English words in wordfreq.

    python code/score_noise.py data/benchmark_context_satire_2202.csv scores/rebuilt
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

from common import load

TOKEN = re.compile(r"[A-Za-z]{2,}")


@lru_cache(maxsize=None)
def is_nonword(token):
    return zipf_frequency(token, "en") == 0


def noise(sentence, spans):
    for s, e in sorted(spans, reverse=True):
        sentence = sentence[:s] + " " + sentence[e:]
    tokens = [t.lower() for t in TOKEN.findall(sentence)]
    return sum(map(is_nonword, tokens)) / max(1, len(tokens))


def main(data, out_dir):
    df = load(data)
    df["susp"] = [noise(s, sp) for s, sp in zip(df.sentence, df.spans)]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df[["id", "susp"]].to_csv(Path(out_dir) / "noise.lex.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
