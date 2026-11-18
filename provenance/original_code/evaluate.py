"""Detector evaluation on the benchmark and the external page-verified set.

    python code/evaluate.py

Reads data/, scores/ and annotations/, writes results/metrics.json and
results/pr_curves.pdf. Each file scores/<set>/<detector>.csv holds one
suspicion score per hit (column `susp`, higher = more likely a substitution).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import binary, load

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260825)
B = 2000  # bootstrap replicates
PRIMARY = "bert-base.pll_orig"
NOISE = "noise.lex"


def auc(y, s):
    """Mann-Whitney AUC (equal to roc_auc_score, faster inside the bootstrap)."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(y):
    """Resample positives and negatives separately."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(B):
        yield np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])


def with_ci(y, s, fn):
    values = [fn(y[i], s[i]) for i in bootstrap(y)]
    return fn(y, s), np.percentile(values, 2.5), np.percentile(values, 97.5)


def paired_delta(y, s1, s2):
    values = [auc(y[i], s1[i]) - auc(y[i], s2[i]) for i in bootstrap(y)]
    return auc(y, s1) - auc(y, s2), np.percentile(values, 2.5), np.percentile(values, 97.5)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def recall_at_fpr(y, s, fpr=0.10, folds=5):
    """Threshold set on the training folds at the given false-positive rate, applied to the held-out fold."""
    tp = fp = 0
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(s, y):
        threshold = np.quantile(s[tr][y[tr] == 0], 1 - fpr)
        flagged = s[te] > threshold
        tp += (flagged & (y[te] == 1)).sum()
        fp += (flagged & (y[te] == 0)).sum()
    return tp / y.sum(), tp / max(1, tp + fp)


def fill(s):
    s = np.asarray(s, dtype=float)
    return np.where(np.isfinite(s), s, np.nanmedian(s))


def load_scores(directory, ids):
    return {f.stem: pd.read_csv(f, dtype={"id": str}).set_index("id")["susp"].reindex(ids)
            for f in sorted(directory.glob("*.csv"))}


def detector_metrics(hits, scores, m, prefix):
    y = hits.y.values
    k = int(round(0.05 * len(y)))
    m[f"{prefix}.n"], m[f"{prefix}.npos"], m[f"{prefix}.k"] = len(y), int(y.sum()), k
    reference = fill(scores[PRIMARY].reindex(hits.id))
    for name, sc in scores.items():
        s = fill(sc.reindex(hits.id))
        key = f"{prefix}.{name}"
        m[f"{key}.auc"], m[f"{key}.auc_lo"], m[f"{key}.auc_hi"] = with_ci(y, s, auc)
        m[f"{key}.ap"] = with_ci(y, s, average_precision_score)[0]
        top = np.argsort(-s, kind="stable")[:k]
        m[f"{key}.p_at_k"] = y[top].mean()
        m[f"{key}.recall_fpr10"], m[f"{key}.precision_fpr10"] = recall_at_fpr(y, s)
        if name != PRIMARY:
            m[f"{key}.dauc"], m[f"{key}.dauc_lo"], m[f"{key}.dauc_hi"] = paired_delta(y, s, reference)


def combined(bench, scores):
    """Logistic regression on [BERT-base PLL, lexical noise]: cross-fitted on the benchmark,
    plus a model fitted on the whole benchmark for transfer."""
    X = fill(np.column_stack([scores[PRIMARY].reindex(bench.id), scores[NOISE].reindex(bench.id)]))
    y = bench.y.values
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        s[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).decision_function(X[te])
    return pd.Series(s, index=bench.id.values), LogisticRegression(max_iter=1000).fit(X, y)


def judge_agreement(bench, m):
    """LLM judges as annotators: kappa with the benchmark labels and recovery of the printed word."""
    subs = (bench.label == "ocr_sub").values
    for f in sorted((ROOT / "annotations").glob("*.csv")):
        d = pd.read_csv(f, dtype={"id": str}).set_index("id").reindex(bench.id)
        ok = d.label.notna().values
        m[f"agree.{f.stem}.kappa2"] = cohen_kappa_score(subs[ok], (d.label.values == "ocr_sub")[ok])
        guess = d.printed_word.astype(str).str.lower().values[subs]
        m[f"recover.{f.stem}.hit1"] = np.mean(guess == bench.source_word.str.lower().values[subs])


def margin_recovery(bench, directory, m):
    """Is the printed word the best (hit1) or among the ten best (hit10) alternatives of the margin scorer?"""
    subs = (bench.label == "ocr_sub").values
    source = bench.source_word.str.lower().values[subs]
    for f in sorted(directory.glob("*.margin.csv")):
        d = pd.read_csv(f, dtype={"id": str}).set_index("id").reindex(bench.id)
        alts = d.alternatives.fillna("").str.lower().str.split().values[subs]
        m[f"recover.{f.stem}.hit1"] = np.mean([len(a) > 0 and a[0] == w for a, w in zip(alts, source)])
        m[f"recover.{f.stem}.hit10"] = np.mean([w in a[:10] for a, w in zip(alts, source)])


def sample_rates(ext, m):
    """Substitution rate per corpus x query sample, with Wilson intervals."""
    for (corpus, query), g in ext.groupby(["corpus", "query"]):
        n, k, u = len(g), int((g.label == "ocr_sub").sum()), int((g.label == "uncertain").sum())
        key = f"sample.{corpus.lower()}.{query}"
        m[f"{key}.n"], m[f"{key}.sub"], m[f"{key}.unresolved"] = n, k, u
        m[f"{key}.rate"] = k / n
        m[f"{key}.rate_upper"] = (k + u) / n
        m[f"{key}.ci"] = "census" if n >= g.population.iloc[0] else list(wilson(k, n))


def benchmark_overlap(ext, bench, m):
    """External hits that are also benchmark hits: contextual label vs page verdict."""
    j = ext[ext.bench_id.notna()].merge(bench[["id", "label"]], left_on="bench_id", right_on="id",
                                        suffixes=("", "_context"))
    j = j[j.label != "uncertain"]
    ctx, page = j.label_context == "ocr_sub", j.label == "ocr_sub"
    m["overlap.n"], m["overlap.context_sub"] = len(j), int(ctx.sum())
    m["overlap.page_sub"], m["overlap.both"] = int(page.sum()), int((ctx & page).sum())


def external_metrics(ext, scores, m):
    """Pooled AUC is confounded by query identity (base rates and typical scores differ by query), so we
    also report AUC after z-normalising scores within each corpus x query sample (auc_z) and the
    within-sample AUC averaged over samples, weighted by their number of substitutions (auc_within)."""
    detector_metrics(ext, scores, m, "ext")
    for corpus, g in ext.groupby("corpus"):
        m[f"ext.{corpus.lower()}.npos"] = int(g.y.sum())
    cell = ext.corpus + "|" + ext["query"]
    y = ext.y.values
    for name, sc in scores.items():
        v = pd.Series(fill(sc.reindex(ext.id)), index=ext.index)
        z = v.groupby(cell).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        m[f"ext.{name}.auc_z"] = roc_auc_score(y, z)
        boot = [auc(y[i], z.values[i]) for i in bootstrap(y)]
        m[f"ext.{name}.auc_z_lo"], m[f"ext.{name}.auc_z_hi"] = np.percentile(boot, [2.5, 97.5])
        num = den = 0.0
        for _, g in ext.groupby(cell):
            if 0 < g.y.sum() < len(g):
                num += g.y.sum() * roc_auc_score(g.y, v[g.index])
                den += g.y.sum()
        m[f"ext.{name}.auc_within"] = num / den
        m[f"ext.{name}.found_top50"] = int(y[np.argsort(-v.values, kind="stable")[:50]].sum())


def pr_figure(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.33, 1.85), sharey=True)
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    colors = ["#1f4e79", "#c0504d", "#4f8a3c", "#7f7f7f", "#b8860b"]
    for ax, (title, y, curves) in zip(axes, panels):
        for i, (label, s) in enumerate(curves.items()):
            precision, recall, _ = precision_recall_curve(y, fill(s))
            ax.plot(recall, precision, linestyle=styles[i], color=colors[i], lw=1, label=label)
        ax.axhline(y.mean(), color="black", lw=0.5, alpha=0.4)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("Precision")
    fig.tight_layout(pad=0.4, rect=(0.02, 0.17, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=5.6,
               frameon=False, handlelength=2.2, columnspacing=1.2)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-label", default="label",
                    help="label column of data/external.csv: label (after re-check) or label_initial")
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    m = {}

    bench_all = load(ROOT / "data/benchmark.csv")
    bench = binary(bench_all)
    scores = load_scores(ROOT / "scores/benchmark", bench_all.id)
    scores["combined"], combined_model = combined(bench, scores)
    m["n_detectors"] = len(scores) + 1  # plus the OCR-quality gate itself
    detector_metrics(bench, scores, m, "bench")
    # sensitivity: proper nouns counted as negatives
    detector_metrics(binary(bench_all, proper_noun=0), scores, m, "bench_pn")
    m["sensitivity.proper_noun.max_dauc"] = max(abs(m[f"bench_pn.{d}.auc"] - m[f"bench.{d}.auc"]) for d in scores)
    judge_agreement(bench_all, m)
    margin_recovery(bench_all, ROOT / "scores/benchmark", m)

    ext_all = load(ROOT / "data/external.csv")
    ext_all["label"] = ext_all[args.external_label]
    sample_rates(ext_all, m)
    benchmark_overlap(ext_all, bench_all, m)
    ext = binary(ext_all, proper_noun=0)
    ext_scores = load_scores(ROOT / "scores/external", ext_all.id)
    X = fill(np.column_stack([ext_scores[PRIMARY], ext_scores[NOISE]]))
    ext_scores["combined"] = pd.Series(combined_model.decision_function(X), index=ext_all.id.values)
    external_metrics(ext, ext_scores, m)

    curves = {"Claude Sonnet 5.5 (LLM)": "claude-sonnet.llm", "RoBERTa-large PLL": "roberta-large.pll_orig",
              "BERT-base PLL": PRIMARY, "lexical noise": NOISE, "in-domain trigram": "ngram-hmd.ngram"}
    pr_figure([(title, hits.y.values, {k: sc[v].reindex(hits.id).values for k, v in curves.items()})
               for title, hits, sc in [("UK benchmark", bench, scores),
                                       ("External (page-verified)", ext, ext_scores)]],
              out / "pr_curves.pdf")

    m = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in m.items()}
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=int))
    rows = [(d, m[f"bench.{d}.auc"], m[f"ext.{d}.auc_z"]) for d in scores]
    print(f"{'detector':28} {'bench AUC':>9} {'ext AUC_z':>9}")
    for d, a, b in sorted(rows, key=lambda r: -r[1]):
        print(f"{d:28} {a:9.3f} {b:9.3f}")


if __name__ == "__main__":
    main()
