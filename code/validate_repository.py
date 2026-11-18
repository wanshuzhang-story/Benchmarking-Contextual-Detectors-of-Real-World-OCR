"""Fast integrity audit: no model downloads or paid API calls."""
from pathlib import Path
import json
import pandas as pd
from common import load
ROOT=Path(__file__).resolve().parent.parent
b=pd.read_csv(ROOT/'data/benchmark_context_satire_2202.csv',dtype={'id':str})
full=pd.read_csv(ROOT/'data/benchmark.csv',dtype={'id':str})
p=pd.read_csv(ROOT/'data/uk_primary_population_3083.csv',dtype={'benchmark_row_id':str})
e=pd.read_csv(ROOT/'data/external.csv',dtype={'id':str})
r=pd.read_csv(ROOT/'data/page_recheck.csv',dtype={'id':str})
a=pd.read_csv(ROOT/'data/external_image_provenance_1093.csv',dtype={'id':str})
assert len(p)==3083 and p.benchmark_row_id.is_unique
assert len(full)==3083 and full.id.is_unique and full.label.notna().sum()==2202 and full.inferred_source_word.notna().sum()==206
assert p['query'].value_counts().to_dict()=={'satire':2202,'parody':684,'misinformation':83,'hoax':114}
assert len(b)==2202 and b.id.is_unique and 'source_word' not in b.columns and 'inferred_source_word' in b.columns
assert b.label.value_counts().to_dict()=={'genuine':1975,'ocr_sub':186,'proper_noun':20,'uncertain':21}
assert b[b.label=='ocr_sub'].inferred_source_word.notna().all()
assert p.original_contextual_label.notna().sum()==2202
assert p.original_inferred_source_word.notna().sum()==186
assert p['inferred_source_word'].notna().sum()==206
assert (p.inferred_word_basis=='LATER_CONTEXT_REANNOTATION').sum()==10
assert (p.inferred_word_basis=='LATER_FULL_2026_CONTEXT_REANNOTATION').sum()==10
assert (p.inferred_word_basis=='ORIGINAL_CONTEXT_INFERENCE').sum()==186
assert len(e)==len(a)==1093 and len(r)==111 and set(a.id)==set(e.id)
assert sum(e.label=='ocr_sub')==49 and sum(e.label=='uncertain')==9
assert sum(e.label_initial=='ocr_sub')==22 and sum(e.label_initial=='uncertain')==2
assert sum(a.review_note_contradiction)==5
assert len(pd.read_csv(ROOT/'data/uk_context_reannotation_v2_FULL_3083.csv'))==3083
assert len(pd.read_csv(ROOT/'data/uk_satire_third_person_image_check_60.csv'))==60
assert len(pd.read_csv(ROOT/'data/image_word_records_available_two_designs_126.csv'))==126
for group,rows in [('benchmark',2202),('external',1093)]:
    fs=list((ROOT/'scores'/group).glob('*.csv'))
    assert len(fs)==22,(group,len(fs))
    for f in fs:
        d=pd.read_csv(f,dtype={'id':str})
        assert len(d)==rows and d.id.is_unique and 'susp' in d.columns,(f,len(d))
for f,rows in [('results/benchmark_all_24_configuration_scores_2161.csv',2161),
               ('results/external_all_24_configuration_scores_1084.csv',1084)]:
    path=ROOT/f
    if path.exists():
        assert len(pd.read_csv(path))==rows
if (ROOT/'results/metrics.json').exists():
    m=json.loads((ROOT/'results/metrics.json').read_text())
    assert m['n_detectors']==24 and m['bench.n']==2161 and m['ext.n']==1084
    assert m['run.external_transfer_excluded_overlapping_binary_benchmark_rows']==91
print('PASS: complete 3083 row population, original label separation, inferred-source-word provenance,\n      saved scores 22x2, external audit counts, recheck flags, output integrity.')
