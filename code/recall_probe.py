"""Near-miss probe for hits lost to OCR: tokens within edit distance 1 of a query (2 for
queries of nine or more letters) that are not among the 300,000 most frequent English words
in wordfreq, per exact query token. Includes long-s readings such as "fatire". The ratio is
an upper bound on lost hits.

    python code/recall_probe.py UK=corpora/hmd.txt US=corpora/uspd.txt
"""
import re
import sys
from collections import Counter

from wordfreq import top_n_list

from common import QUERY_PATTERNS
from vulnerability import levenshtein


def main(corpora):
    words = set(top_n_list("en", 300_000))
    for spec in corpora:
        corpus, path = spec.split("=", 1)
        with open(path) as fh:
            freq = Counter(w for line in fh for w in re.findall(r"[a-z]+", line))
        exact = near = 0
        for query, pattern in QUERY_PATTERNS.items():
            is_query = re.compile(rf"^{pattern}$").match
            d = 2 if len(query) >= 9 else 1
            exact += sum(c for w, c in freq.items() if is_query(w))
            near += sum(c for w, c in freq.items() if not is_query(w) and w not in words
                        and abs(len(w) - len(query)) <= d and levenshtein(w, query, cap=d) <= d)
        print(f"{corpus}: {near / max(1, exact):.3f} near-miss tokens per exact query token")


if __name__ == "__main__":
    main(sys.argv[1:])
