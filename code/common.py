import re

import pandas as pd

# Surface forms matched for each query.
QUERY_PATTERNS = {
    "hoax": r"hoax(?:es|ed|ing|ers?)?",
    "misinformation": r"misinformations?",
    "parody": r"parod(?:y|ies|ied|ying)",
    "satire": r"satires?",
    "propaganda": r"propagandas?",
    "rumour": r"rumou?r(?:s|ed|ing)?",
}


def keyword_spans(sentence, query):
    pattern = rf"\b{QUERY_PATTERNS[query]}\b"
    return [m.span() for m in re.finditer(pattern, sentence, flags=re.IGNORECASE)]


def load(path):
    """Read a hit table and locate the query word(s) in each sentence."""
    df = pd.read_csv(path, dtype={"id": str})
    df["spans"] = [keyword_spans(s, q) for s, q in zip(df.sentence, df["query"])]
    return df


def binary(df, label="label", proper_noun=None):
    """Keep genuine (y=0) and ocr_sub (y=1) hits; proper nouns optionally as y=proper_noun."""
    y = {"genuine": 0, "ocr_sub": 1}
    if proper_noun is not None:
        y["proper_noun"] = proper_noun
    out = df[df[label].isin(y)].copy()
    out["y"] = out[label].map(y)
    return out


def window(sentence, start, end, max_chars=600):
    """Cut long sentences to max_chars around the keyword."""
    if len(sentence) <= max_chars:
        return sentence, start, end
    half = (max_chars - (end - start)) // 2
    a = max(0, start - half)
    b = min(len(sentence), end + half)
    return sentence[a:b], start - a, end - a
