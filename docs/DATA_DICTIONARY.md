# Dataset dictionary and version policy

## Primary, complete population

**Canonical:** `data/benchmark.csv` (3,083 rows). `id` is `uk-sat-N` for 2,202 historical satire cases (matches saved scores), and a stable `benchmark_row_id` identifier for the 881 non-satire cases. `query` is the search concept, not an extracted semantic referent. `sentence` is an OCR sentence. `label` is the **original** contextual OCR label when available: `genuine`, `ocr_sub`, `proper_noun`, or `uncertain`. It is missing, not `genuine`, for 881 historic non-satire rows because original case-level labels are unavailable. Do not impute the original per-row 0/0/5 aggregate.

`inferred_source_word` is a proposed OCR-context replacement and **is never automatically equivalent to a newspaper-image transcription**. 186 entries come from the original satire context annotations. A further 10 non-disputed entries come from the supplied later parody context review and 10 from a later full hoax review. **206 entries in total**. Four previous non-satire suggestions were withheld after cross-checking the earlier semantic-extraction files; earlier values are retained in the cross-check audit. No historical original labels were inferred from these later fields. `inferred_word_basis` identifies the origin. `inferred_word_confidence=UNVERIFIED_CONTEXTUAL` explicitly disclaims source-image verification.

`later_review_source_word_candidate_raw` preserves tentative strings such as `parade?`, `UNCERTAIN`, and `surname (proper noun)` without interpreting them as certain printed words. `later_full_reannotation_guessed_word` preserves the original value of a different, more extensive later context review, including the **two-word** suggestion `this information`. `inferred_word_conflict_flag` marks values that differ across versions. The raw reviewer data take precedence over guessing. Do not force multiword or source-name proposals into a one-token substitution statistic. The historic case `uk-sat-2172` differs only in capitalization (`salic` vs `Salic`) across inputs; saved benchmark comparison uses the original CSV spelling and the master preserves its original capitalization.

## Context reviews and separate image checks

`data/uk_context_reannotation_v2_FULL_3083.csv` has exactly 3,083 case matches, 72 later *suspected* substitutions, 31 name labels, 137 uncertain, 2,762 `GENUINE`, and 81 unrecorded. This is a later annotation version: **not 3,083 image-verification judgments**, not the original 191 benchmark positives. Of the 2,202 satire records with two contextual versions, 1,863 match and 339 differ under the version comparison's label mapping.

`data/uk_hoax_semantic_annotations_114.csv` targets *semantic/content extraction* (`Original_Content`, `Manual_Valid`, etc.). It must not be used as an OCR-image gold standard.

`data/second_annotator.csv` is a selected 300-case, three-stratum **OCR-context-only** second annotation. The unweighted vs design-weighted kappa scores are different by design; use both with precise labels.

`data/uk_satire_third_person_image_check_60.csv` is a **targeted, nonrandom** third-person check, originally recorded in the `Satire_image_check_1.xlsx` workbook and linked to the full later comparison. Its 60 records include 46 source-word entries: 33 substitution verdicts and 13 target-printed verdicts. There are 13 unresolved and one ambiguous. These 60 do not have the same sample denominator or independence assumptions as the external 11-cell random audit.

## External audit and reported printed words

`data/external.csv` has 1,093 query-stratified UK+US hits with `label_initial` before the 111-case recheck and final `label` after it. `uncertain` is never silently treated as a confirmed non-substitution. Final: 1,084 evaluable / 49 paper-reported substitutions / 9 unresolved. Initial: 1,091 evaluable / 22 recorded substitutions / 2 unresolved. Recheck selection was enriched by model flags (54 LLM + 30 encoder, plus 7 conflicts + 20 controls), so its final detector comparisons may inherit score-correlated verification bias.

`data/external_image_provenance_1093.csv` records source locator, reported printed word when saved, recheck notes, and evidential tiers, not fresh 2026 page-image reading. There are 80 nonmissing `printed_word` records in the directly supplied external CSV, of which some are correct printed query tokens rather than substitutions. The available third-person UK data add 46 reported word readings. `data/image_word_records_available_two_designs_126.csv` concatenates them **for provenance navigation only; never pool their labels for prevalence or detector AUC**. The 100 targeted US satire verification records mentioned in old prose were not supplied and are not invented.

Five source-note contradictions are flagged: `UK-hoax-83`, `UK-hoax-89`, `UK-parody-81`, `UK-satire-10`, and `UK-rumour-13`. Each has a conclusive published label but a recheck note `not found in BNA`. This is a provenance conflict requiring actual scan review; no silent correction was made.

## Reproducible evaluation boundaries

The only historical score-compatible UK dataset is `data/benchmark_context_satire_2202.csv`. The saved 22 detector score files in each of `scores/benchmark/` and `scores/external/` must not be misrepresented as 3,083-row inference files. The two additional columns `combined` and `admission_gate_constant` in wide results are **derived** from those scores, not independently stored original model inferences. The combined external logistic model excludes 91 overlapping binary benchmark cases from fitting.

Query-vulnerability exposures and exact corpus settings from historical runs are absent. The 11-cell CSV contains actual sampled reported rates with unresolved bounds, but leaves the unpublished historic exposure fields empty. Any newly rebuilt exposures must be tagged **new experiment**.
