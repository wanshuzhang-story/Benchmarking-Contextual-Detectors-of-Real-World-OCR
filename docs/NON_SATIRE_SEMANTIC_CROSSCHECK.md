# Cross-check of 881 non-satire rows against earlier manually verified semantic-extraction files

## Scope and meaning
All 114 hoax, 83 misinformation and 684 parody rows match the benchmark exactly by row index and sentence, including the parody workbook which arrived with a misleading `.csv` extension but is an XLSX file. The historical source fields `Original_Content`, `Original_Success`, `Manual_Valid`, `Manual_Content`, `Manual_Function`, and `Note` assess *semantic extraction of referents or functions*, not original newspaper-page OCR verification. Therefore a `Manual_Valid=True` is not evidence that the queried keyword was correctly printed, nor is `Manual_Valid=exclude` proof of a specific OCR replacement. The original non-satire OCR gold labels remain absent.

## Primary source-word inference decisions
Four previously entered context-only guesses have been **withheld from the primary inferred_source_word field**, not deleted. They remain preserved in `word_inference_prior_to_semantic_check` and original later-review columns:

| Benchmark ID | Old inferred word | Why it is withheld |
|---|---|---|
| UK_HOAX_0093 | houses | Historical extraction note discusses home/domestic referent; does not support the exact word houses. Other words remain plausible in garbled trade context. |
| UK_PARODY_0017 | presided | Preliminary guess conflicts with a later complete review that marks the row UNCERTAIN. |
| UK_PARODY_0024 | Paradise | The word is a poor semantic fit for the advertisement "at his Parodies for the Deaf"; context sources repeat the guess without a scan reading. |
| UK_PARODY_0641 | paradise | Earlier proposed guess conflicts with a later full review's GENUINE category. Neither is independent page verification. |

**Updated primary count: 206 nonblank contextual word inferences**: 186 unchanged original satire suggestions, 10 later hoax suggestions and 10 later parody suggestions. All non-satire primary guesses remain contextual, not independently image-verified. The earlier conflicting misinformation word `information` versus `this information` remains unadjudicated and blank in the primary column. Newly introduced cross-check columns retain original manual semantic annotations and a traceable reason for withholding.

The three older manually verified sources do **not** prove that any of the remaining guesses are right or wrong, and cannot establish the originally reported five per-row parody OCR labels. Images would be needed for actual source-word adjudication. No manuscript detector metrics or historical original labels were changed.
