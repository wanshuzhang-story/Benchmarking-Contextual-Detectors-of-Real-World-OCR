"""Reproduce detector comparisons from SAVED per-hit scores and separate original/rechecked labels.

    python code/evaluate.py --bootstrap 2000 --seed 20260825
    python code/evaluate.py --external-label label_initial --output-suffix _initial_sensitivity

The UK satire reference labels and `inferred_source_word` are CONTEXT INFERENCES,
not independent newspaper image transcriptions. External labels are paper-reported
page-audit judgments, with model-enriched recheck selection.

Combined model: 5-fold out-of-fold on the 2,161 original binary benchmark rows.
External transfer fit EXCLUDES every benchmark row linked by any external bench_id.
Constant quality gate is included in score matrices, not passed off as a detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import binary, load

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = 'bert-base.pll_orig'
NOISE = 'noise.lex'


def fill(s):
    """Non-finite scores are imputed from each score COLUMN, never median-filled for names."""
    a = np.asarray(s, dtype=float).copy()
    if a.ndim == 1:
        good = np.isfinite(a)
        if not good.any():
            raise ValueError('No finite scores in detector')
        a[~good] = np.median(a[good])
    else:
        for col in range(a.shape[1]):
            a[:, col] = fill(a[:, col])
    return a


def load_scores(directory, ids):
    ans = {}
    for f in sorted(Path(directory).glob('*.csv')):
        d = pd.read_csv(f, dtype={'id': str})
        if d.id.duplicated().any():
            raise ValueError(f'Duplicate detector IDs: {f}')
        s = d.set_index('id')['susp'].reindex(ids)
        ans[f.stem] = s
    return ans


def auc(y, scores):
    return roc_auc_score(y, scores)


class PairedStratifiedBootstrap:
    """One common set of resampling weights for all detectors and paired delta CIs.

    Multinomial count draws are distributionally identical to sampling each
    class with replacement (and much faster than repeated pandas/SL evaluation).
    """
    def __init__(self, labels, n_replicates, seed):
        self.y = np.asarray(labels, dtype=np.uint8)
        self.B = n_replicates
        self.npos = int(self.y.sum())
        self.nneg = len(self.y) - self.npos
        assert self.npos > 0 and self.nneg > 0
        rg = np.random.default_rng(seed)
        self.weights = np.zeros((n_replicates, len(self.y)), dtype=np.int16)
        for idx in [np.flatnonzero(self.y == 1), np.flatnonzero(self.y == 0)]:
            self.weights[:, idx] = rg.multinomial(len(idx), np.full(len(idx), 1/len(idx)), size=n_replicates).astype(np.int16)

    def score_distributions(self, scores):
        """AUC and average precision for the paired class-stratified draws, ties exact."""
        s = fill(scores)
        order = np.argsort(-s, kind='stable')
        sorted_s = s[order]
        ends = np.r_[np.flatnonzero(sorted_s[:-1] != sorted_s[1:]), len(s) - 1]
        w = self.weights[:, order].astype(np.int32)
        tp = np.cumsum(w * self.y[order][None, :], axis=1)[:, ends]
        fp = np.cumsum(w * (1 - self.y[order])[None, :], axis=1)[:, ends]
        tp_grp = np.diff(np.pad(tp, ((0,0),(1,0))), axis=1)
        fp_grp = np.diff(np.pad(fp, ((0,0),(1,0))), axis=1)
        aucs = np.sum(tp_grp * (self.nneg - fp) + 0.5 * tp_grp * fp_grp, axis=1) / (self.npos * self.nneg)
        aps = np.sum(tp_grp * np.divide(tp, tp + fp, out=np.zeros_like(tp,dtype=float), where=(tp + fp)!=0), axis=1) / self.npos
        return aucs, aps

    @staticmethod
    def ci(x):
        return [float(v) for v in np.percentile(x, [2.5,97.5])]


def recall_at_fpr(y, s, fpr=.10, folds=5):
    tp=fp=0
    for tr,te in StratifiedKFold(folds,shuffle=True,random_state=0).split(s,y):
        threshold = np.quantile(s[tr][y[tr]==0], 1-fpr)
        flagged = s[te] > threshold
        tp += int((flagged & (y[te]==1)).sum())
        fp += int((flagged & (y[te]==0)).sum())
    return tp/max(1,int(y.sum())),tp/max(1,tp+fp)


def combined_predict(benchmark_all, bench_scores, external, ext_scores):
    """Crossfit binary benchmark; external transfer excludes all overlaps (not only positives)."""
    bench = binary(benchmark_all)
    X=fill(np.column_stack([bench_scores[PRIMARY].reindex(bench.id),bench_scores[NOISE].reindex(bench.id)]))
    y=bench.y.values
    oof=np.zeros(len(bench))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        oof[te] = LogisticRegression(max_iter=1000).fit(X[tr],y[tr]).decision_function(X[te])
    # Genuine model scores for the 20 proper-name records. Do not assign an artificial median.
    benchmark_full_fit=LogisticRegression(max_iter=1000).fit(X,y)
    remaining = benchmark_all[~benchmark_all.id.isin(bench.id)]
    Xextra=fill(np.column_stack([bench_scores[PRIMARY].reindex(remaining.id),bench_scores[NOISE].reindex(remaining.id)]))
    full = pd.concat([pd.Series(oof,index=bench.id.values),
                      pd.Series(benchmark_full_fit.decision_function(Xextra),index=remaining.id.values)])
    full=full.reindex(benchmark_all.id)
    overlapping = set(external.bench_id.dropna().astype(str))
    keep = ~bench.id.isin(overlapping)
    excluded=int((~keep).sum())
    if excluded != 91:
        raise RuntimeError(f'Unexpected external/benchmark transfer overlap: {excluded} (expected 91)')
    trans=LogisticRegression(max_iter=1000).fit(X[keep],y[keep])
    EX=fill(np.column_stack([ext_scores[PRIMARY].reindex(external.id),ext_scores[NOISE].reindex(external.id)]))
    es=pd.Series(trans.decision_function(EX), index=external.id.values)
    return full,es,excluded


def detector_metrics(hits, scores, m, prefix, bs):
    y=hits.y.to_numpy(dtype=np.uint8)
    k=int(round(.05*len(y)))
    m.update({f'{prefix}.n':len(y),f'{prefix}.npos':int(y.sum()),f'{prefix}.k':k})
    ref=fill(scores[PRIMARY].reindex(hits.id))
    ref_dist,_=bs.score_distributions(ref)
    for name,sr in scores.items():
        s=fill(sr.reindex(hits.id))
        da,dp=bs.score_distributions(s)
        key=f'{prefix}.{name}'
        lo,hi=bs.ci(da); alo,ahi=bs.ci(dp)
        m.update({f'{key}.auc':float(auc(y,s)),f'{key}.auc_lo':lo,f'{key}.auc_hi':hi,
                  f'{key}.ap':float(average_precision_score(y,s)),
                  f'{key}.ap_lo':alo,f'{key}.ap_hi':ahi,
                  f'{key}.p_at_k':float(y[np.argsort(-s,kind='stable')[:k]].mean())})
        m[f'{key}.recall_fpr10'],m[f'{key}.precision_fpr10']=[float(x) for x in recall_at_fpr(y,s)]
        if name!=PRIMARY:
            dlo,dhi=bs.ci(da-ref_dist)
            m.update({f'{key}.dauc':float(m[f'{key}.auc']-m[f'{prefix}.{PRIMARY}.auc']) if f'{prefix}.{PRIMARY}.auc' in m else float(auc(y,s)-auc(y,ref)),
                      f'{key}.dauc_lo':dlo,f'{key}.dauc_hi':dhi})
    # Include admission gate only as the nondiscriminating fixed-score control.
    m[f'{prefix}.admission_gate.auc']=0.5
    m[f'{prefix}.admission_gate.ap']=float(y.mean())


def judge_agreement(bench,m):
    subs=(bench.label=='ocr_sub').to_numpy()
    for f in sorted((ROOT/'annotations').glob('*.csv')):
        d=pd.read_csv(f,dtype={'id':str}).set_index('id').reindex(bench.id)
        ok=d.label.notna().to_numpy()
        m[f'agree.{f.stem}.kappa2']=float(cohen_kappa_score(subs[ok],(d.label.to_numpy()=='ocr_sub')[ok]))
        guesses=d.printed_word.fillna('').astype(str).str.casefold().to_numpy()[subs]
        inferred=bench.inferred_source_word.fillna('').astype(str).str.casefold().to_numpy()[subs]
        m[f'recover.{f.stem}.hit1']=float(np.mean(guesses==inferred))


def margin_recovery(bench,m):
    subs=(bench.label=='ocr_sub').to_numpy()
    inferred=bench.inferred_source_word.fillna('').astype(str).str.casefold().to_numpy()[subs]
    for f in sorted((ROOT/'scores/benchmark').glob('*.margin.csv')):
        d=pd.read_csv(f,dtype={'id':str}).set_index('id').reindex(bench.id)
        alt=d.alternatives.fillna('').astype(str).str.casefold().str.split().to_numpy()[subs]
        m[f'recover.{f.stem}.hit1']=float(np.mean([len(a)>0 and a[0]==w for a,w in zip(alt,inferred)]))
        m[f'recover.{f.stem}.hit10']=float(np.mean([w in a[:10] for a,w in zip(alt,inferred)]))


def wilson(k,n,z=1.96):
    import math
    if not n: return [float('nan')]*2
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.0,c-h),min(1.0,c+h)]


def sample_rates(external,m):
    for (corp,query),g in external.groupby(['corpus','query']):
        n=len(g);k=int((g.label=='ocr_sub').sum());u=int((g.label=='uncertain').sum());key=f'sample.{corp.lower()}.{query}'
        m.update({f'{key}.n':n,f'{key}.sub':k,f'{key}.unresolved':u,
                  f'{key}.rate':k/n, f'{key}.rate_lower':k/n,
                  f'{key}.rate_resolved':k/(n-u), f'{key}.rate_upper':(k+u)/n})
        m[f'{key}.ci_lower_bound_assumption']=list(wilson(k,n))
        m[f'{key}.ci_resolved_only']=list(wilson(k,n-u)) if n < int(g.population.iloc[0]) else 'census'
        m[f'{key}.ci']='census' if n >= int(g.population.iloc[0]) else list(wilson(k,n))


def benchmark_overlap(ext,bench,m):
    j=ext[ext.bench_id.notna()].merge(bench[['id','label']],left_on='bench_id',right_on='id',suffixes=('','_context'))
    j=j[j.label!='uncertain']
    ctx=j.label_context=='ocr_sub';page=j.label=='ocr_sub'
    m.update({'overlap.n':len(j),'overlap.context_sub':int(ctx.sum()),
              'overlap.page_sub':int(page.sum()),'overlap.both':int((ctx&page).sum())})


def external_metrics(ext,scores,m,bs):
    detector_metrics(ext,scores,m,'ext',bs)
    for corp,g in ext.groupby('corpus'):m[f'ext.{corp.lower()}.npos']=int(g.y.sum())
    cell=(ext.corpus+'|'+ext['query']).to_numpy()
    y=ext.y.to_numpy(dtype=np.uint8)
    for name,sr in scores.items():
        s=fill(sr.reindex(ext.id))
        v=pd.Series(s,index=ext.index)
        z=v.groupby(pd.Series(cell,index=ext.index)).transform(lambda x:(x-x.mean())/(x.std()+1e-9)).to_numpy()
        dist,_=bs.score_distributions(z)
        m[f'ext.{name}.auc_z']=float(auc(y,z))
        m[f'ext.{name}.auc_z_lo'],m[f'ext.{name}.auc_z_hi']=bs.ci(dist)
        num=den=0.0
        for _,g in ext.groupby(['corpus','query']):
            if 0<int(g.y.sum())<len(g):
                num+=int(g.y.sum())*roc_auc_score(g.y,v.loc[g.index]);den+=int(g.y.sum())
        m[f'ext.{name}.auc_within']=float(num/den) if den else None
        m[f'ext.{name}.found_top50']=int(y[np.argsort(-s,kind='stable')[:50]].sum())


def pr_figure(bench,bscores,ext,escores,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    picks={'Claude Sonnet (LLM)':'claude-sonnet.llm','RoBERTa-large PLL':'roberta-large.pll_orig',
           'BERT-base PLL':PRIMARY,'lexical noise':NOISE,'in-domain trigram':'ngram-hmd.ngram'}
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True)
    for ax,(title,hits,scorefiles) in zip(axes,[('UK satire (context labels)',bench,bscores),
                                                ('UK+US (reported audit labels)',ext,escores)]):
        for label,name in picks.items():
            s=fill(scorefiles[name].reindex(hits.id))
            p,r,_=precision_recall_curve(hits.y.values,s)
            ax.plot(r,p,lw=1,label=label)
        ax.axhline(hits.y.mean(),color='gray',alpha=.6,linestyle=':')
        ax.set(xlabel='Recall',title=title,xlim=(0,1),ylim=(0,1.01))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Precision')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.07),ncol=3,fontsize=7)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(path,bbox_inches='tight');plt.close(fig)


def prepare_scores(bench_all,external_all):
    bench_scores=load_scores(ROOT/'scores/benchmark',bench_all.id)
    ext_scores=load_scores(ROOT/'scores/external',external_all.id)
    if len(bench_scores)!=22 or len(ext_scores)!=22:
        raise RuntimeError('Missing original per-row saved detector-score files')
    full,ext,excluded=combined_predict(bench_all,bench_scores,external_all,ext_scores)
    bench_scores['combined']=full
    ext_scores['combined']=ext
    return bench_scores,ext_scores,excluded


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--external-label',choices=['label','label_initial'],default='label')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20260825)
    ap.add_argument('--output-suffix',default='')
    args=ap.parse_args()
    if args.bootstrap < 20: raise ValueError('Require at least 20 bootstrap draws')
    m={'run.bootstrap_replicates':args.bootstrap,'run.bootstrap_seed':args.seed,
       'run.bootstrap_method':'paired_class_stratified_multinomial_with_replacement',
       'run.external_label_column':args.external_label,
       'run.benchmark_reference_label_basis':'CONTEXT_INFERRED_NOT_IMAGE_CONFIRMED',
       'run.external_label_basis':'PAPER_REPORTED_AUDIT_WITH_MODEL_ENRICHED_RECHECK'}
    bench_all=load(ROOT/'data/benchmark_context_satire_2202.csv')
    ext_all=load(ROOT/'data/external.csv')
    scores,ext_scores,excluded=prepare_scores(bench_all,ext_all)
    m['run.external_transfer_excluded_overlapping_binary_benchmark_rows']=excluded
    m['n_detectors']=len(scores)+1
    bench=binary(bench_all)
    detector_metrics(bench,scores,m,'bench',PairedStratifiedBootstrap(bench.y,args.bootstrap,args.seed))
    bench_pn=binary(bench_all,proper_noun=0)
    detector_metrics(bench_pn,scores,m,'bench_pn',PairedStratifiedBootstrap(bench_pn.y,args.bootstrap,args.seed+1))
    m['sensitivity.proper_noun.max_dauc']=float(max(abs(m[f'bench_pn.{d}.auc']-m[f'bench.{d}.auc']) for d in scores))
    judge_agreement(bench_all,m)
    margin_recovery(bench_all,m)
    ext_all['label']=ext_all[args.external_label]
    sample_rates(ext_all,m)
    benchmark_overlap(ext_all,bench_all,m)
    ext=binary(ext_all,proper_noun=0)
    external_metrics(ext,ext_scores,m,PairedStratifiedBootstrap(ext.y,args.bootstrap,args.seed+2))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    suff=args.output_suffix or ('_initial_sensitivity' if args.external_label=='label_initial' else '')
    (out/f'metrics{suff}.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
    pr_figure(bench,scores,ext,ext_scores,out/f'pr_curves{suff}.pdf')
    print(f"Benchmark {m['bench.n']} / {m['bench.npos']}; external {m['ext.n']} / {m['ext.npos']}; overlap excluded {excluded}")
    for name in ('bert-base.pll_orig','roberta-large.pll_orig','claude-sonnet.llm','combined'):
        print(name,'benchmark AUC',round(m[f'bench.{name}.auc'],6),
              'external raw AUC',round(m[f'ext.{name}.auc'],6),
              'external z-AUC',round(m[f'ext.{name}.auc_z'],6))

if __name__=='__main__':main()
