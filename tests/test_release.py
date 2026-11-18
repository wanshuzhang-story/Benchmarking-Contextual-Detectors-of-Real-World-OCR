import json,hashlib,unittest
from pathlib import Path
import pandas as pd
import openpyxl
R=Path(__file__).resolve().parents[1]

class TestCompleteRelease(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.full=pd.read_csv(R/'data/benchmark.csv',dtype={'id':str})
  cls.old=pd.read_csv(R/'data/benchmark_context_satire_2202.csv',dtype={'id':str})
  cls.pop=pd.read_csv(R/'data/uk_primary_population_3083.csv',dtype={'benchmark_row_id':str})
  cls.ext=pd.read_csv(R/'data/external.csv',dtype={'id':str})
  cls.m=json.loads((R/'results/metrics.json').read_text())

 def test_full_rows_historical_separation(self):
  self.assertEqual(len(self.full),3083)
  self.assertEqual(self.full['query'].value_counts().to_dict(),{'satire':2202,'parody':684,'hoax':114,'misinformation':83})
  self.assertEqual(self.full.label.notna().sum(),2202)
  self.assertEqual(self.full[self.full['query']!='satire'].label.notna().sum(),0)
  self.assertEqual(self.old.label.value_counts().to_dict(),{'genuine':1975,'ocr_sub':186,'proper_noun':20,'uncertain':21})
  self.assertEqual(self.old[self.old.label=='ocr_sub'].inferred_source_word.notna().sum(),186)

 def test_word_inference_versions(self):
  self.assertEqual(self.full.inferred_source_word.notna().sum(),206)
  self.assertEqual(self.pop.original_inferred_source_word.notna().sum(),186)
  self.assertEqual((self.pop.inferred_word_basis=='LATER_CONTEXT_REANNOTATION').sum(),10)
  self.assertEqual((self.pop.inferred_word_basis=='LATER_FULL_2026_CONTEXT_REANNOTATION').sum(),10)
  self.assertTrue(self.full[self.full.inferred_source_word.notna()].inferred_word_basis.notna().all())
  self.assertTrue(self.full[self.full.inferred_source_word.notna()].inferred_word_confidence.notna().all())

 def test_semantic_crosscheck_881(self):
  d=pd.read_csv(R/'data/non_satire_semantic_crosscheck_881.csv')
  self.assertEqual(len(d),881)
  self.assertEqual(d['query'].value_counts().to_dict(),{'parody':684,'hoax':114,'misinformation':83})
  self.assertTrue(d.semantic_source_sentence_exact_match.all())
  self.assertEqual(d.inferred_word_crosscheck_status.str.startswith('WITHHELD').sum(),4)
  self.assertEqual(self.full[self.full['query']!='satire'].inferred_source_word.notna().sum(),20)
  for id in ('UK_HOAX_0093','UK_PARODY_0017','UK_PARODY_0024','UK_PARODY_0641'):
   row=d[d.benchmark_row_id==id].iloc[0]
   self.assertTrue(pd.isna(row.inferred_source_word))
   self.assertTrue(pd.notna(row.word_inference_prior_to_semantic_check))
  self.assertTrue((R/'data/non_satire_word_candidate_decisions.csv').exists())

 def test_versioned_human(self):
  d=pd.read_csv(R/'data/uk_context_reannotation_v2_FULL_3083.csv')
  self.assertEqual(len(d),3083)
  self.assertEqual((d.reannotation_interpreted_category=='SUSPECTED_OCR_SUBSTITUTION').sum(),72)
  self.assertEqual((d.reannotation_interpreted_category=='UNRECORDED').sum(),81)
  self.assertEqual((d.original_vs_reannotation=='DISAGREE').sum(),339)
  self.assertEqual(len(pd.read_csv(R/'data/uk_context_reannotation_comparison_3083.csv')),3083)

 def test_uk_image_check_separate(self):
  d=pd.read_csv(R/'data/uk_satire_third_person_image_check_60.csv')
  self.assertEqual(len(d),60)
  self.assertEqual(d.third_printed_word.notna().sum(),46)
  self.assertEqual(d.third_image_label.value_counts().to_dict(),
                   {'OCR_SUBSTITUTION':33,'TARGET_PRINTED':13,'UNRESOLVED':13,'AMBIGUOUS':1})
  self.assertEqual(len(pd.read_csv(R/'data/image_word_records_available_two_designs_126.csv')),126)

 def test_external_paper_labels_and_provenance(self):
  self.assertEqual(len(self.ext),1093)
  self.assertEqual((self.ext.label=='ocr_sub').sum(),49)
  self.assertEqual((self.ext.label=='uncertain').sum(),9)
  self.assertEqual((self.ext.label_initial=='ocr_sub').sum(),22)
  self.assertEqual((self.ext.label_initial=='uncertain').sum(),2)
  prov=pd.read_csv(R/'data/external_image_provenance_1093.csv')
  self.assertEqual(len(prov),1093)
  self.assertEqual(prov.has_reported_printed_word.sum(),80)
  self.assertEqual(prov.review_note_contradiction.sum(),5)
  self.assertEqual(len(pd.read_csv(R/'data/query_vulnerability_11_cells_inputs.csv')),11)

 def test_no_original_score_changed(self):
  h=json.loads((R/'provenance/original_score_sha256.json').read_text())
  self.assertEqual(len(h),44)
  for path,hexhash in h.items():
   with self.subTest(path=path):self.assertEqual(hashlib.sha256((R/path).read_bytes()).hexdigest(),hexhash)

 def test_reproduced_metrics(self):
  m=self.m
  self.assertEqual(m['n_detectors'],24)
  self.assertEqual(m['bench.n'],2161)
  self.assertEqual(m['ext.n'],1084)
  self.assertEqual(m['ext.npos'],49)
  self.assertEqual(m['run.bootstrap_replicates'],2000)
  self.assertEqual(m['run.external_transfer_excluded_overlapping_binary_benchmark_rows'],91)
  self.assertAlmostEqual(m['bench.bert-base.pll_orig.auc'],.8960950047638492,places=12)
  self.assertAlmostEqual(m['bench.roberta-large.pll_orig.auc'],.9101823873689942,places=12)
  self.assertAlmostEqual(m['ext.roberta-large.pll_orig.auc_z'],.9031055900621119,places=12)
  self.assertAlmostEqual(m['ext.claude-sonnet.llm.auc_z'],.9698807059055506,places=12)
  x=json.loads((R/'results/annotation_checks.json').read_text())
  self.assertAlmostEqual(x['second.kappa_llm_majority_unweighted'],.868235,places=5)
  self.assertAlmostEqual(x['second.kappa_llm_majority_design_weighted'],.782011,places=5)
  old=json.loads((R/'provenance/uploaded_metrics_reference.json').read_text())
  for key,val in m.items():
   if (key.endswith('.auc') or key.endswith('.auc_z') or key.endswith('.ap') or key.endswith('.found_top50') or key.endswith('.p_at_k')) and key in old:
    self.assertAlmostEqual(val,old[key],places=8,msg=key)

 def test_matrix_alignment(self):
  for name,n in [('benchmark_all_24_configuration_scores_2161.csv',2161),('external_all_24_configuration_scores_1084.csv',1084)]:
   t=pd.read_csv(R/'results'/name)
   self.assertEqual(len(t),n)
   meta=['id','query','label']+(['corpus'] if 'corpus' in t.columns else [])
   self.assertEqual(len(t.columns)-len(meta),24)
   self.assertTrue(t.id.is_unique)
   self.assertEqual(t.admission_gate_constant.nunique(),1)

 def test_excel_complete_and_named_inference(self):
  p=R/'data/UK_Benchmark_Complete_3083_with_Inferred_Source_Words.xlsx'
  w=openpyxl.load_workbook(p,read_only=True,data_only=True)
  s=w['Benchmark_3083'];s.reset_dimensions(); it=s.values; header=next(it)
  self.assertIn('inferred_source_word',header)
  self.assertEqual(len(list(it)),3083)
  w.close()

if __name__=='__main__':unittest.main()
