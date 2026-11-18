# Exact manuscript corrections for the supplied five-page WSDM PDF

**Source inspected:** `Benchmarking_Contextual_Detectors_of_Real_Word_OCR_Substitutions_in_Historical_Newspaper_Search(3).pdf` (5 pages). Locations refer to the *printed PDF page* and printed line numbering; line wrapping may differ when the corresponding LaTeX source is edited. The actual source TeX was not supplied, so **these are copy-paste replacements, not claims that the PDF has been modified**.

**Non-negotiable interpretation:** a full **3,083-row UK benchmark population** exists, but historic detector *row-level* reference labels exist only for 2,202 satire rows. Its 186 original substitution words are context-inferred, not page-image transcriptions. A separate later 3,083-row context reannotation exists and has 72 suspected substitutions, which cannot be substituted for historic labels. The external 1,093 sampled rows contain 1,084 resolved after 9 uncertain, with 49 **paper-reported** OCR substitutions (37 UK, 12 US); most `genuine` reports lack a recorded transcribed page word and rechecks were model-enriched.

## Priority 1: exact replacement texts (must fix before submission)

### 1. Page 1 — replace the entire Abstract (approximately printed lines 8–35)

The existing abstract calls 1,084 hits “page-verified random hits,” asserts ρ=0.79 as reproduced, and equates the original contextual annotations with true substitutions. Replace the abstract with:

> Keyword search over historical newspaper OCR can return false hits when a different valid word is recognized as the query term. We compare 24 detector configurations for prioritizing manual review, including lexical-noise and trigram baselines, causal and masked language models, alternative scoring functions, and LLM judges. On 2,161 UK satire hits originally annotated from OCR context (186 suspected substitutions), RoBERTa-large pseudo-log-likelihood attains AUC 0.910 and BERT-base 0.896 against the contextual labels. For cross-corpus transfer, we examine 1,093 query-stratified UK and US audit samples. Nine remain unresolved, leaving 1,084 evaluable cases with 49 paper-reported substitutions. Query-normalized AUC reaches 0.903 for RoBERTa-large and 0.970 for the best LLM judge. A blind but model-enriched image re-check increased recorded positive verdicts from 22 to 49; its selection mechanism and incomplete source-word documentation limit interpretation. We release the complete UK row population, versioned annotations, evidence notes, detector scores, and code.

This uses only results reproduced in the released files, removes the unreplicated vulnerability rho, and does not call all 1,084 external records independently page-word-verified.

### 2. Page 1 — Introduction, example sentence (approximately printed lines 66–70)

**Find:** `yet the page reads “entire”.`

**Replace with:** `yet the surrounding text suggests “entire” as a possible original reading. The benchmark's proposed source words were inferred from OCR context, not transcribed independently from page images.`

Likewise, in the abstract's example “an satire part” **for** “an active part,” use “contextually suggests” if retaining that example; do not present the wording as an independently read scan unless the relevant page is provided and matched.

### 3. Page 1 — Contributions paragraph (approximately printed lines 82–92)

**Find:** `Zero-shot transfer to 1,091 page-verified random hits from two corpora`

**Replace the entire contribution sentence/numbered clause with:**

> (2) Zero-shot transfer to 1,084 evaluable cases drawn from 1,093 query-stratified UK and US newspaper hits, with query-normalized and within-cell metrics and source-evidence qualifications.

**Find:** `(3) A blind, detector-guided page re-check that more than doubles the confirmed substitutions in the audits and quantifies the recall of contextual labelling.`

**Replace with:**

> (3) A blind, detector-guided re-check of 111 selected cases that increases paper-reported substitution judgments from 22 to 49 and reveals both context-annotation disagreement and verification-selection bias.

**Find:** `(4) An annotation-free predictor of query vulnerability, tested against the audits.`

**Replace with:**

> (4) A provenance-aware data release separating historical contextual labels, subsequent context reannotations, reported page readings, unresolved cases, and reproducible per-hit detector scores.

The predictor claim can return **only if** the actual historic UK and US frequency corpora and exposure computations are supplied and reproduced. Remove the unconditional ρ=0.79 claim from the abstract and conclusions in the meantime.

### 4. Page 2 — Table 1 caption and UK benchmark column (approximately printed lines 117–138)

**Find:** `Table 1: Evaluation data. Benchmark: contextual labels ... Audits: simple random samples of 100 hits, page-verified after the blind re-check`

**Replace caption/footnote with:**

> **Table 1: Evaluation samples and paper-reported audit labels.** The UK benchmark contains a complete 3,083-row four-query population; its original per-row contextual OCR labels are available only for 2,202 satire records. The non-satire figures are previously reported aggregate counts, not reconstructed row-level image labels. UK and US audit cells were sampled independently by corpus and query (100 hits per cell, except a 93-hit UK misinformation census). The original audit includes bulk verdicts without a separately recorded printed source word; the later blind re-check was selected partly by detector flags. Nine final audit cases are unresolved and excluded from the 1,084-case detector comparison. For unresolved audit cells, the tabulated Wilson intervals use the selected sample denominator and treat unresolved cases as non-substitutions for a **lower-bound rate**, not a resolved-only prevalence estimate.

**Keep** the historical aggregate table values hoax 0, misinformation 0, parody 5 and satire 186 as **originally suspected** counts; do **not** assign specific original labels to the 881 non-satire rows. The full later context review is a separate dataset and cannot revise this table without explicitly labeling a new annotation version.

**UK satire row clarification:** 19/100 sampled lower-bound reported fraction = 19.0%; 19/95 resolved-only = 20.0%; 24/100 upper bound if all five unresolved were substitutions = 24.0%. The current [12.5%, 27.8%] Wilson interval is for **19/100 under the missing-as-negative assumption**, not 19/95. UK hoax (2 unresolved) and UK rumour (2 unresolved) also require the same distinction.

### 5. Page 2 — “Within-query benchmark” paragraph (approximately printed lines 141–151)

**Find:** `The labels are 1,975 genuine, 186 substitutions, 20 proper nouns, and 21 uncertain.`

**Replace with:**

> The original satire annotations comprise 1,975 contextually genuine hits, 186 suspected substitutions, 20 proper-name ambiguities, and 21 uncertain cases. Their proposed source words are inferred from sentence context rather than independent page transcription. The binary comparison uses the 2,161 contextually genuine or suspected-substitution cases (8.6% originally suspected).

**Find:** `[name the admission metric in one clause]`

**Replace the placeholder only after checking the source pipeline.** The released rows have `impresso_ocr_score=1.0` and `completeness_score=100`, but these columns do **not**, by themselves, prove which upstream admission gate was run. Suggested conditional wording when confirmed: `In the released set, the recorded impresso OCR score is 1.0 for every hit (completeness score 100).` Never ship the literal `[name the admission metric ...]` placeholder.

### 6. Page 2 — “External page-verified set” paragraph (approximately printed lines 152–169)

**Find:** `Every sampled hit was checked against the page image ... Most verdicts were first recorded in bulk; after the blind re-check of §5, and excluding 7 pages that could not be resolved, the set has 1084 hits with 49 substitutions`

**Replace this contiguous section with:**

> The audit sampled 1,093 hits by corpus and query (593 UK and 500 US). Initial paper-reported judgments recorded 22 substitutions and two unresolved cases; many genuine verdicts were checked or recorded in bulk without saving the printed word. A subsequent blind but model-enriched re-check of 111 selected cases produced 27 additional reported substitutions and seven further unresolved cases. The final comparison therefore uses 1,084 evaluable hits and 49 reported substitutions (37 UK, 12 US); nine cases are unresolved. Source locators, transcribed words where saved, re-check notes, and contradictory verdict flags are released separately. This external set is a paper-audit comparison, not 1,084 individually documented printed-word transcriptions.

The submitted PDF's `excluding 7` is inconsistent with the full dataset because two cases were already unresolved before the re-check. Also qualify `Printed source words include ...` as `Recorded printed-word reports include ...` (not all 1,084 have a saved printed word).

**Find:** `In a separate check, page images confirmed 99 of 100 hits sampled from 256 contextually flagged US satire candidates ...`

**Conditional:** The underlying 100-case verification source table was not supplied to the current repository. Either add it, along with its sampling identifiers and image-verdict provenance, or remove the numerical 99/100 claim from a reproducibility-focused version. The released 60-case UK image check is a *different* sample and does not substitute for this missing US dataset.

### 7. Page 2 — Detectors: trigrams, LLMs, supervised model (approximately printed lines 178–198)

**Find:** `In-domain trigram: stupid backoff ... on 30M held-out tokens of HMD ... or US-PD`

**Replace with (until original corpora/manifest recovered):**

> In-domain trigram: stupid backoff on newspaper text filtered for overlapping evaluation 8-grams. Historic corpus snapshots and exact token caps were not retained; we provide scripts for a new, separately versioned 30-million-token rebuild. The reported saved scores are preserved, but this rebuild is not proof of the historical corpus size.

**Find:** `at temperature 0 (minimal reasoning effort for reasoning models).`

**Replace with:** `with the inference options recorded in the supplied client code (minimal reasoning effort where supported, or disabled thinking for Anthropic); explicit temperature settings were not recorded for the historical requests.`

**Find:** `Combination: logistic regression on BERT-base PLL and lexical noise, cross-fitted over 5 folds and frozen for the external set.`

**Replace with:** `Combination: logistic regression on BERT-base PLL and lexical noise, evaluated by five-fold out-of-fold predictions on the benchmark. For external transfer, the fitted model excludes 91 benchmark cases overlapping the external audit (external raw AUC 0.8694).`

**Find:** `Except for the combination, every detector is zero-shot, so all evaluation is query-held-out by construction.`

**Replace with:** `Except for the supervised combination, detectors use saved zero-shot scores without task-specific fitting. The within-satire evaluation holds query identity constant; the external comparison tests additional queries and a second corpus, but its paper-audit labels were partly rechecked based on detector flags.`

### 8. Page 3 — Table 2 caption (approximately printed lines 233–238)

**Find:** `External: 1084 page-verified random hits ... 49 substitutions after the re-check ... zero-shot.`

**Replace with:** `External: 1,084 evaluable cases drawn from 1,093 query-stratified samples, 49 paper-reported substitutions after a blind, model-enriched re-check; results are relative to these labels, not uniformly archived source-word transcriptions. The supervised combined detector excludes training overlaps; it is not zero-shot.`

The code now uses the SAME 2,000 paired, class-stratified bootstrap draws for each detector comparison and includes average-precision percentile CIs. **Some confidence limits in Table 2 differ by ~0.001–0.003 from the uploaded PDF; update them together rather than mixing methods.** See `results/table2_reproduced_values.csv` for every displayed detector's exact AUC, CI, AP, P@5%, R@10%, external AUC, q-norm AUC, weighted within-cell AUC, and top-50 count.

Main reproduced comparison lines: BERT-base PLL 0.896 [0.871, 0.919]; RoBERTa-large PLL 0.910 [0.888, 0.930]; Claude Sonnet 0.911 [0.889, 0.932]. External RoBERTa-large q-normalized AUC 0.903; Claude 0.970. None of these require invented historical labels.

### 9. Page 3 — Results: “printed word” assertions (approximately printed lines 279–286 and 335–343)

**Find:** `Historical pretraining does help with a different task, proposing the printed word.`

**Replace with:** `On the original context-annotated satire benchmark, historical pretraining can help propose the annotator's inferred replacement word; this is not a test of recovery of independently transcribed page words.`

**Find:** `An interface can therefore show the likely reading next to a flag.`

**Replace with:** `An interface could display the contextually suggested reading as a hypothesis to be checked against the page image.`

**Find:** `They also name the printed word better: claude-sonnet-5-5’s guess matches the inferred source word for 52.2% ...`

**Replace with:** `On the context-annotated satire data, Claude's proposed replacement agrees with the original annotator's context-inferred word for 52.2% of suspected cases, versus 16.1% for LwM-BERT's top-ranked alternative. These are agreement-with-inference scores, not independently validated printed-word restoration rates.`

When discussing “39 of 49” top-50 hits, keep the numerical ranking **conditional on the model-enriched paper-audit labels**; it is not an unbiased recall estimate against an independent random image-grounded gold set.

### 10. Page 4 — page re-check interpretation (approximately printed lines 355–374)

**Find:** `Single-pass contextual labels are thus precise but miss roughly half of the substitutions, so benchmark AUCs are likely underestimates.`

**Replace with:** `Among the 90 audit cases also present in the context-labeled benchmark, the page-audit record reports 19 substitutions, while the original context version flagged 10, of which nine agree. This illustrates disagreement and possible missed candidates within a small, partly model-rechecked overlap; it does not establish population-level recall or the direction of AUC bias.`

**Find:** `the clean controls and the unchanged ranking under the original labels suggest the effect is small, but triaged audits should be read as lower bounds.`

**Replace with:** `Because re-check selection was partly based on LLM and encoder suspicion, final labels may preferentially discover errors flagged by those models. The original-label sensitivity is therefore reported separately; blinded presentation does not remove selection bias, and neither comparison is an independent uniformly transcribed image-gold test.`

The old wording **“more than doubles the confirmed substitutions”** should consistently become **“increased the number of paper-reported substitutions from 22 to 49.”** Do not call every positive conclusively image-verified without resolving the five contradictory source notes.

### 11. Page 4 — 300-case second annotator (approximately printed lines 374–384)

**Find:** `population-weighted κ = 0.37 with the original labels, but 0.87 with the LLM majority`

**Replace with:**

> The 300 second-annotator judgments were made from OCR context alone and were sampled from three deliberately enriched strata (100 original suspected substitutions, 100 LLM-flagged original negatives, and 100 others). Against the original context labels, design-weighted Cohen's κ is approximately 0.369. Against the LLM-majority context judgments, κ is 0.868 unweighted and 0.782 design-weighted. These statistics concern contextual agreement, not the printed word on a page. Any design-weighted prevalence estimate is exploratory and not independent image-grounded prevalence.

The existing 0.87 is **unweighted**, while 0.37 is **design-weighted**; they must not be directly contrasted without naming the estimators.

### 12. Page 4 — “Anticipating vulnerable queries” (approximately printed lines 386–398)

**Find:** `Across the 11 audited corpus–query cells, Spearman ρ = 0.79 (p = 0.004); with decay rates 1 and 3, ρ = 0.79 and ρ = 0.84.`

**Recommended action:** **Remove the entire unconditional ρ paragraph** from the five-page manuscript and remove the claim from the abstract/conclusion until the exact original 30M/50M corpus frequency snapshot and eleven exposure inputs are provided and the result reproduced. The supplied archive has none of those exposures. A new builder default of 30M cannot verify what the historical experiment used. If retained purely as exploratory, explicitly say `exploratory, historical frequency inputs and corpus snapshot unavailable in the release; correlation not independently reproduced` and remove the headline-contribution framing.

### 13. Page 4 — Limitations and Conclusion (approximately printed lines 400–423)

**Find:** `Across 24 detector configurations, a held-out page-verified set, and a label audit, bidirectional contextual scoring reliably ranks false keyword hits ...`

**Replace the opening of the final claims with:**

> Against the original context-only UK satire labels, bidirectional masked-language-model scores rank suspected false keyword hits above lexical and causal baselines. Performance relative to the external UK/US paper-audit labels is promising, especially for LLM judges, but the audited negative evidence is often recorded in bulk and the blind re-check was selected partly from model flags. We therefore recommend using scores to prioritize manual image review rather than automatically deleting search results. Further evaluation needs randomly selected, independently transcribed image labels and adjudication of contradictory page locators.

**Insert in limitations:** `The full UK row population includes 881 non-satire records without recoverable historical row-level OCR labels; later contextual reannotations cannot replace them. Some recorded substitutions involve multiword phrases, proper names, merged columns or segmentation rather than a straightforward one-token real-word replacement.`

### 14. Page 4 — Ethical Considerations (approximately printed lines 425–435)

**Find:** `All texts are public-domain or openly licensed (CC0) historical newspapers.`

**Replace with:** `The analysis uses historical newspaper OCR and metadata distributed by the respective source providers under their stated terms. The release contains textual research annotations, scores and source locators but no British Newspaper Archive page images or viewer captures; access to such images remains subject to the provider's terms.`

Keep the ranking-not-deletion caution and the discussion of epistemic error. Do not blanket-license digitized BNA page scans as CC0.

### 15. Page 1 — publishing metadata and anonymous URL (approximately printed lines 45–58, 115)

- The ACM Reference Format currently says **“In Proceedings of Proceedings of the Twentieth ...”**. Delete the duplicated `Proceedings of` and verify the official `acmart` conference short title.
- Replace the footnote's `[create anonymous repo]` placeholder with the actual double-blind repository address **after** uploading this ZIP. Do not invent a URL; the current ZIP is a local artifact, not a deployed anonymous repository.
- The `10.1145/nnnnnnn.nnnnnnn` DOI and generic ISBN are template placeholders. Apply conference instructions; do not present these as published bibliographic details.

## Global language replacements to search in the LaTeX manuscript

| Current wording | Use instead, unless an individually documented page transcription supports it |
|---|---|
| `page-verified random hits` | `evaluable sampled hits with paper-reported audit labels` |
| `186 substitutions` (UK satire benchmark) | `186 originally suspected substitutions from contextual labels` |
| `191 substitutions` (all four historic query totals) | `191 originally reported context-suspected substitutions across four queries` |
| `source_word` for UK benchmark | `inferred_source_word` |
| `correctly predicts the printed word` (benchmark) | `agrees with the context-inferred replacement word` |
| `confirmed substitutions` (bulk/recheck audit) | `paper-reported substitution verdicts`, until conflicting source notes resolved |
| `1,091 page-verified` (final external) | `1,093 sampled, nine unresolved, 1,084 evaluable` |
| `excluding seven pages` (final external) | `nine unresolved cases (two initial, seven newly unresolved)` |
| `temperature 0` | `historic temperature not explicitly recorded` |
| `trained on 30M tokens` (historic score run) | `historic corpus token cap unverified; 30M is the new rebuild setting` |
| `ρ=0.79` as reproducible primary conclusion | remove or qualify as unreproduced exploratory result |
| `0.87` agreement without weighting label | `κ=0.868 unweighted; κ=0.782 design-weighted` for LLM-majority vs second annotator |

## Evidence still needed from coauthors

- Confirm the precise pipeline admission quality gate (recorded `impresso_ocr_score=1` and `completeness_score=100` are visible, but may not document the preceding filter).
- Resolve page identities and evidence for `UK-hoax-83`, `UK-hoax-89`, `UK-parody-81`, `UK-satire-10`, `UK-rumour-13`; inspect proper-name and segmentation case `UK-satire-72`.
- Supply the missing 100-case US satire source-image verification table if keeping the 99/100 statement.
- Supply original 881 non-satire row labels only if actually recovered; do not backfill from the later 3,083 context review.
- Supply historical corpus snapshots/revisions, exposure values, 30M/50M documentation and API request logs if retaining the vulnerable-query correlation and temperature=0 wording.

## Exact numerical consistency checks

See `results/table2_reproduced_values.csv` for all detector rows and confidence intervals. The model scores and point AUCs were reproduced from the original saved score files. The rebuilt 2,000-replicate bootstrap uses shared resample weights for paired comparisons. It may differ slightly from confidence limits printed in the prior PDF, so **choose one bootstrap implementation** and regenerate all intervals consistently.
