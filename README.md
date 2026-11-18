# Contextual Detectors of Real-Word OCR Substitutions — anonymous release

**Release scope:** full recoverable UK benchmark population (3,083 rows), original UK satire evaluation subset (2,202 rows, 2,161 binary), available later human context annotations, 60-case third-person UK image review, cross-corpus page-audit judgments (1,093 sampled), 44 preserved original score files, revised analysis code, and recalculated outputs.

**Read before reuse:** `inferred_source_word` is a word **inferred from OCR context**. It is **not** a transcription of the original newspaper image. The old 881 non-satire UK *historical* row-level detector labels are unavailable, even though the 881 underlying rows and later annotation versions are present. Do not treat later inferred words as the historical label version.

## Where the actual complete benchmark is

- **`data/benchmark.csv` — 3,083 rows. This is the canonical complete UK benchmark population, including `id`, `query`, `sentence`, `label`, `inferred_source_word`, and explicit label/word provenance columns.** Original `label` is set for the 2,202 satire rows and intentionally blank for the 881 non-satire rows. It is *not* a 3,083-row gold-classification test set.
- **`data/UK_Benchmark_Complete_3083_with_Inferred_Source_Words.xlsx` — 3,083-row reviewer-friendly Excel workbook**, with same row order, prominent inference fields, later annotation versions, date/source metadata and evidence notes.
- `data/uk_primary_population_3083.csv` — the full original reconstructed master columns plus versioned review fields and historical `benchmark_row_id`.
- `data/benchmark_context_satire_2202.csv` — **the original 2,202-row, score-compatible context-only UK satire subset**. This is what the reproduced benchmark statistics use. The former misleading `source_word` header was renamed `inferred_source_word`, preserving all values, apart from output CSV formatting.

**Counts:** 114 hoax + 83 misinformation + 684 parody + 2,202 satire = 3,083. The original satire row labels contain 1,975 genuine, 186 suspected OCR substitutions, 20 proper names and 21 uncertain. **All 186 original substitution rows retain their context-inferred replacement word.** The canonical full file now carries 20 non-satire, later context-inferred single-token words (10 hoax and 10 parody), after four uncertain guesses were withheld during review of previous semantic-extraction files; an additional misinformation *information/this information* disagreement remains in versioned fields rather than falsely adjudicated. Thus **206 primary `inferred_source_word` entries** are nonblank, each with provenance. Alternative/rejected/tentative proposals are preserved in their own fields. Neither 206 nor the 72 suspected in the later review is a count of image-confirmed substitutions.

## Files and evidence layers

| Relative path | Records | Meaning |
|---|---:|---|
| `data/benchmark.csv` | 3,083 | Complete UK population + original row labels where available; primary context-inferred words and provenance |
| `data/benchmark_context_satire_2202.csv` | 2,202 | Old score-compatible context annotation (2,161 binary) |
| `data/uk_context_reannotation_v2_FULL_3083.csv` | 3,083 | Later *different* human context-annotation version (72 suspected substitutions, 31 names, 137 uncertain, 81 unrecorded). Do not backfill original 881 labels. |
| `data/uk_context_reannotation_comparison_3083.csv` | 3,083 | Side-by-side original, preliminary, later full review and selected image-readings |
| `data/uk_hoax_semantic_annotations_114.csv` | 114 | Semantic/content extraction annotations; not OCR-image gold |
| `data/second_annotator.csv` | 300 | Context-only, selected three-stratum sample, not random |
| `data/uk_satire_third_person_image_check_60.csv` | 60 | Separately selected third-person page readings: 33 substitutions, 13 target printed, 1 ambiguous, 13 unresolved; 46 reported source words |
| `data/external.csv` | 1,093 | Query-stratified UK+US paper-audit labels and source links; 1,084 final evaluable, 49 reported substitutions; 9 unresolved |
| `data/external_image_provenance_1093.csv` | 1,093 | Row-aligned available page word / recheck note / provenance tier; five contradictory “not found” notes flagged |
| `data/external_reported_source_words_available.csv` | 80 | Word strings actually recorded in the supplied 1,093-row audit file (not uniformly independent verification) |
| `data/image_word_records_available_two_designs_126.csv` | 126 | 80 audit-recorded words + 46 third-person UK words from distinct selection designs; **do not pool as prevalence** |
| `data/page_recheck.csv` | 111 | Blinded presentation but model-enriched selection: 54 LLM, 30 encoder, 7 conflict, 20 control |
| `data/query_vulnerability_11_cells_inputs.csv` | 11 | Audit rate bounds; historical exposure-frequency inputs blank because archived corpora are missing |
| `source_workbooks/` | 5 XLSX + 1 CSV | Available provenance workbooks, with creator/last-modifier metadata anonymized, cells preserved; no page images redistributed |
| `scores/benchmark/`, `scores/external/` | 22 + 22 score files | Original saved per-hit model scores, **preserved byte for byte** |
| `results/*_all_24_configuration_scores_*.csv` | 2,161 / 1,084 | Wide aligned matrices: 22 saved scores plus newly derived logistic combination and fixed admission gate |
| `results/metrics.json` | — | Recomputed final-label AUC/AP with 2,000 paired bootstrap replicates |
| `results/metrics_initial_sensitivity.json` | — | Results using initial labels: 1,091 evaluable, 22 positive; for verification-selection sensitivity |
| `results/annotation_checks.json` | — | Unweighted and design-weighted Cohen's kappa for context-only annotation |

## Reproduce without model downloads or API access

Python with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `openpyxl`:

```bash
python -m pip install -r requirements-analysis.txt
python code/validate_repository.py
python code/evaluate.py --bootstrap 2000 --seed 20260825
python code/evaluate.py --external-label label_initial --output-suffix _initial_sensitivity --bootstrap 2000 --seed 20260825
python code/annotation_checks.py
python code/export_score_matrix.py
python code/vulnerability.py --rates-only
python -m unittest discover -s tests -v
```

The saved metrics remain separate after the second call because of `--output-suffix`. To regenerate the historic **score values themselves** requires the models and exact historic API/corpus settings, many of which are not documented by the provided archive. The released full 3,083-row CSV is for data access; `data/benchmark_context_satire_2202.csv` is the precise dataset to use with the 2,202 saved per-hit score files.

**Reproduced key figures:** UK satire BERT-base PLL AUC 0.896095; RoBERTa-large PLL AUC 0.910182; external query-normalized RoBERTa-large PLL AUC 0.903106 and Claude 0.969881; leakage-free combined external raw AUC 0.869407 (91 overlapping original binary benchmark rows excluded). All are **relative to released labels** rather than guaranteed uniformly independently page-verified truth. The 300-case second annotator vs. LLM-majority kappa is 0.868235 unweighted and 0.782011 design-weighted.

## Optional new corpus / model scoring

New corpus building defaults to 30 million tokens and writes a versioned checksum manifest. It **does not establish** that historic scores used 30 million rather than the original code's 50-million-token default.

```bash
python code/build_corpus.py biglam/hmd_newspapers corpora/hmd.txt --max-tokens 30000000 --revision PINNED_DATASET_REVISION
python code/build_corpus.py PleIAs/US-PD-Newspapers corpora/uspd.txt --max-tokens 30000000 --revision PINNED_DATASET_REVISION
python code/score_ngram.py data/external.csv corpora/hmd.txt scores/rebuilt --corpus UK --output-name ngram-uk.csv
python code/score_ngram.py data/external.csv corpora/uspd.txt scores/rebuilt --corpus US --output-name ngram-us.csv
python code/merge_ngram.py scores/rebuilt/ngram-uk.csv scores/rebuilt/ngram-us.csv scores/rebuilt/ngram-hmd.ngram.csv
python code/vulnerability.py UK=corpora/hmd.txt US=corpora/uspd.txt
```

`code/score_lm.py`, `score_noise.py`, `score_ngram.py`, `llm_judge.py` and `recall_probe.py` are present for optional new runs; do not overwrite the preserved historical `scores/` files. `llm_judge.py --temperature` is an explicit **new-run** option, not evidence that the old API requests specified temperature 0.

## Important missing evidence

The archive is complete **for the recoverable records and saved scores available to this reconstruction**, but not all materials previously described in the separate uploaded README were supplied. In particular, the 11 original per-query page-audit source workbooks, the separate 100-case targeted US satire image-check table, the historical 30M/50M corpus snapshots and token-frequency exposures, exact historic API request configurations, and the 881 old non-satire labels were unavailable. The newer full 3,083-row annotation comparison and 60-case image check have now been recovered from the user's existing file library. **No absent image labels or frequencies were fabricated.** Five source-localization notes explicitly say “not found in BNA” despite conclusive paper verdicts: see the flagged rows and `docs/LIMITATIONS.md`.

## Rights, anonymity, manuscript

Code is MIT licensed (`LICENSE`). Labels and analysis tables may be used as research outputs subject to the rights of underlying providers. **Do not redistribute British Newspaper Archive scans or viewer material as CC0.** The package contains source locators, not copied page images. Workbook creator/last-modifier metadata were changed to `Anonymous`. No API tokens are included. Follow institutional/double-blind release guidance before publication.

For precise five-page manuscript changes, see **`MANUSCRIPT_EXACT_CHANGES.md`** (the original submitted TeX was not provided, so the manuscript PDF itself was not edited). See `docs/LIMITATIONS.md` and `provenance/source_manifest.json` for evidence provenance.

## Manual semantic-source alignment (2026-11-18)

The three earlier `*_UK_manually_verified` uploads are **semantic extraction annotation** files, not page-image OCR labels. All 881 non-satire sentences match exactly. We have preserved original human source annotations in `data/non_satire_semantic_crosscheck_881.csv`, the complete candidate decision table in `data/non_satire_word_candidate_decisions.csv`, and the XLSX sheets `NonSatire_Semantic_Check` and `Word_Review`. Four uncertain primary guesses were withheld but retained in prior-guess columns. See `docs/NON_SATIRE_SEMANTIC_CROSSCHECK.md`; all 186 historical satire guesses remain untouched.
