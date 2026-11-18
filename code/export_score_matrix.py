"""Wide, ID-aligned matrices: 22 saved detector configurations plus 2 derived controls.
Run after installation of requirements-analysis.txt, no model downloads needed.
"""
from pathlib import Path
import pandas as pd
from common import load, binary
from evaluate import ROOT,prepare_scores,fill

bench_all=load(ROOT/'data/benchmark_context_satire_2202.csv')
ext_all=load(ROOT/'data/external.csv')
bs,es,exc=prepare_scores(bench_all,ext_all)
ext_final=binary(ext_all,proper_noun=0)
for label,hits,scores,fn in [('benchmark',binary(bench_all),bs,'benchmark_all_24_configuration_scores_2161.csv'),
                             ('external',ext_final,es,'external_all_24_configuration_scores_1084.csv')]:
    out=hits[['id','query','label']].reset_index(drop=True).copy()
    if 'corpus' in hits:out.insert(1,'corpus',hits.corpus.to_numpy())
    for name,s in sorted(scores.items()):out[name]=fill(s.reindex(hits.id))
    out['admission_gate_constant']=1.0
    assert out.id.is_unique and len(out.columns)-(['id','query','label','corpus'].__contains__('corpus') if False else (4 if 'corpus' in out else 3))==24
    (ROOT/'results').mkdir(exist_ok=True)
    out.to_csv(ROOT/'results'/fn,index=False)
    print(label,len(out),'configurations',24,'excluded',exc)
