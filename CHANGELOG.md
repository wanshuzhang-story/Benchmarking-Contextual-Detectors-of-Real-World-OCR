# Anonymous repository rebuild — 2026-11-18

- Replaced the **canonical `data/benchmark.csv`** with the full 3,083-row UK population and a provenance-qualified `inferred_source_word` column. Preserved score-compatible 2,202-row original satire data separately in `data/benchmark_context_satire_2202.csv`. No unknown historical row-level labels were manufactured.
- Restored 186/186 original satire source-word **inferences**; integrated 13 nonconflicting later parody inferences and 11 nonconflicting later hoax inferences (210 populated primary fields). Kept tentative guesses, multiword suggestions, and disagreements in side-by-side version columns.
- Added the available full 3,083-row later context-reannotation workbook and its CSV, with clear separation from historical label versions. Added a 60-case third-person UK satire image-check CSV and metadata-anonymized source workbook (46 recorded source words). No image was newly reopened.
- Included original provided 114 hoax semantic annotations and two later reviewed workbooks (83 misinformation, 684 parody), plus reconstructed 3,083 workbook.
- Preserved all original 44 per-hit detector score files. Added ID-aligned 24-configuration wide matrices, derived combined scores and a constant quality-gate control.
- Removed supervised-model transfer leakage by excluding 91 overlapping binary benchmark rows. Computed genuine combined-model predictions for proper-name sensitivity instead of median filling.
- Recalculated final and initial-label metrics with 2,000 paired, stratified multinomial-equivalent bootstrap resamples and added AP confidence intervals.
- Published two kappa estimators with explicit weighting, recheck-source tiers, contradictory note flags, and unresolved-rate lower/upper bounds.
- Added corpus-specific rebuilding options, 30M new-run default, SHA256 corpus manifests and optional LLM temperature while explicitly not claiming historic settings from new defaults.
- Preserved the exact uploaded original code snapshots in `provenance/original_code/` to document changes.
- Added exact manuscript correction guide and limitations, without changing the original 5-page PDF or inventing source/corpus evidence.

**Missing historical evidence was not fabricated:** 881 old non-satire row labels, 11 original page-audit workbooks, 100 US targeted image checks, historic corpus snapshots/exposures and exact API configurations. See `docs/LIMITATIONS.md`.

- 2026-11-18 semantic-source cross-check: aligned all 881 non-satire rows to three earlier manually verified semantic-content datasets; the renamed parody spreadsheet had an incorrect `.csv` extension. Distinguished semantic extraction validity from image OCR verification. Withheld four disputable primary non-satire word guesses with earlier values preserved, reducing contextual words 210 -> 206, and added 881-row and candidate audit tables. Model evaluations and original satire annotations unchanged.
