"""Annotation and external re-check statistics. Context-only second annotation != page-image truth.

The 300 cases were deliberately selected from three strata, so unweighted and
design-weighted kappas answer different questions. The page re-check was blind
in presentation, but selected partly by detector outputs (model-enriched).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'

def main():
 m={'second.annotation_basis':'CONTEXT_ONLY','second.selection':'TARGETED_THREE_STRATA',
    'page_recheck.selection':'BLIND_PRESENTATION_BUT_MODEL_ENRICHED'}
 r=pd.read_csv(DATA/'page_recheck.csv')
 m['recheck.n']=len(r);m['recheck.unresolved']=int((r.verdict=='uncertain').sum())
 for group,g in r.groupby('selection'):
  m[f'recheck.{group}.n']=len(g)
  m[f'recheck.{group}.sub']=int((g.verdict=='ocr_sub').sum())
 d=pd.read_csv(DATA/'second_annotator.csv',dtype={'id':str})
 b=pd.read_csv(DATA/'benchmark_context_satire_2202.csv',dtype={'id':str})[['id','label']]
 d=d.merge(b,on='id',suffixes=('','_benchmark'),validate='one_to_one')
 assert len(d)==300
 d['w']=d.stratum_size/d.groupby('stratum').id.transform('size')
 y=(d.label=='ocr_sub').values
 ctx=(d.label_benchmark=='ocr_sub').values
 m['second.n']=len(d)
 for group,g in d.groupby('stratum'):
  m[f'second.{group}.sub_rate']=float((g.label=='ocr_sub').mean())
 m['second.kappa_context_vs_second_unweighted']=float(cohen_kappa_score(ctx,y))
 m['second.kappa_context_vs_second_design_weighted']=float(cohen_kappa_score(ctx,y,sample_weight=d.w))
 # Retain historic name as explicit alias for downstream scripts.
 m['second.kappa_weighted']=m['second.kappa_context_vs_second_design_weighted']
 m['second.prevalence_design_weighted']=float(np.average(y,weights=d.w))
 vote=[]
 for f in sorted((ROOT/'annotations').glob('*.csv')):
  z=pd.read_csv(f,dtype={'id':str}).set_index('id').label.reindex(d.id)
  assert not z.isna().any(),f
  vote.append((z.values=='ocr_sub').astype(int))
 majority=(np.sum(vote,axis=0)>=2)
 m['second.kappa_llm_majority_unweighted']=float(cohen_kappa_score(majority,y))
 m['second.kappa_llm_majority_design_weighted']=float(cohen_kappa_score(majority,y,sample_weight=d.w))
 m['second.kappa_llm_majority']=m['second.kappa_llm_majority_design_weighted']
 keep=d.label.isin(['genuine','ocr_sub']).values
 for f in sorted((ROOT/'scores/benchmark').glob('*.csv')):
  sc=pd.read_csv(f,dtype={'id':str}).set_index('id').susp.reindex(d.id).values.astype(float)
  good=keep & np.isfinite(sc)
  m[f'second.{f.stem}.auc']=float(roc_auc_score(y[good],sc[good],sample_weight=d.w[good]))
 (ROOT/'results').mkdir(exist_ok=True)
 (ROOT/'results/annotation_checks.json').write_text(json.dumps(m,indent=2)+'\n')
 print('300 context-only reannotations; LLM-majority kappa unweighted=',round(m['second.kappa_llm_majority_unweighted'],6),
       'design-weighted=',round(m['second.kappa_llm_majority_design_weighted'],6))

if __name__=='__main__':main()
