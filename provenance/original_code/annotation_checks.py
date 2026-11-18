"""Blind page re-check of external hits and blind second annotation of benchmark hits.

    python code/annotation_checks.py

Writes results/annotation_checks.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def page_recheck(m):
    r = pd.read_csv(DATA / "page_recheck.csv")
    m["recheck.n"] = len(r)
    m["recheck.unresolved"] = int((r.verdict == "uncertain").sum())
    for group, g in r.groupby("selection"):
        m[f"recheck.{group}.n"] = len(g)
        m[f"recheck.{group}.sub"] = int((g.verdict == "ocr_sub").sum())


def second_annotator(m):
    """Three strata of 100 hits (benchmark substitutions, genuine hits flagged by all three LLM
    judges, all other hits). Population-level figures weight each hit by stratum size / 100."""
    d = pd.read_csv(DATA / "second_annotator.csv", dtype={"id": str})
    d = d.merge(pd.read_csv(DATA / "benchmark.csv", dtype={"id": str})[["id", "label"]],
                on="id", suffixes=("", "_benchmark"))
    d["w"] = d.stratum_size / d.groupby("stratum").id.transform("size")
    sub, sub_benchmark = d.label == "ocr_sub", d.label_benchmark == "ocr_sub"
    m["second.n"] = len(d)
    for stratum, g in d.groupby("stratum"):
        m[f"second.{stratum}.sub_rate"] = (g.label == "ocr_sub").mean()
    m["second.kappa_weighted"] = cohen_kappa_score(sub_benchmark, sub, sample_weight=d.w)
    m["second.prevalence_weighted"] = np.average(sub, weights=d.w)

    judges = {}
    for f in sorted((ROOT / "annotations").glob("*.csv")):
        judges[f.stem] = pd.read_csv(f, dtype={"id": str}).set_index("id").label.reindex(d.id).values == "ocr_sub"
    majority = np.sum(list(judges.values()), axis=0) >= 2
    m["second.kappa_llm_majority"] = cohen_kappa_score(majority, sub, sample_weight=d.w)

    # detectors scored against the second annotator's labels (genuine vs ocr_sub only)
    keep = d.label.isin(["genuine", "ocr_sub"]).values
    for f in sorted((ROOT / "scores/benchmark").glob("*.csv")):
        s = pd.read_csv(f, dtype={"id": str}).set_index("id").susp.reindex(d.id).values.astype(float)
        ok = keep & np.isfinite(s)
        m[f"second.{f.stem}.auc"] = roc_auc_score(sub[ok], s[ok], sample_weight=d.w[ok])


def main():
    m = {}
    page_recheck(m)
    second_annotator(m)
    m = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in m.items()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/annotation_checks.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
