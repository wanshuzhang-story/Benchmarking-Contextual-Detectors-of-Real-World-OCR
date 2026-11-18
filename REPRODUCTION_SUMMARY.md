# Verified reproduction summary — 18 November 2026

Existing per-row saved scores were preserved. Local analysis was rerun with `python code/evaluate.py --bootstrap 2000 --seed 20260825`, then repeated against `label_initial` with a separate output name. Full annotation kappas, wide matrices and integrity checks were regenerated. No large model was downloaded, newspaper image was newly reopened, or unavailable frequency reconstructed.

| Statistic | Reproduced |
|---|---:|
| Complete UK four-query row population | 3,083 |
| Score-compatible original UK satire context set | 2,202; 2,161 binary; 186 suspected |
| Original satire inferred replacement words retained | 186/186 |
| Nonblank, nonconflicting primary inferred words in 3,083-row file | 206 |
| Later full context review | 3,083 with 72 later suspected substitutions (distinct labels) |
| External sampled, finally evaluable, reported positives | 1,093; 1,084; 49 |
| Initial-label evaluable and positives | 1,091; 22 |
| External cases with a recorded source word in supplied CSV | 80 |
| Targeted third-person UK satire review | 60; 46 source-word records |
| Preserved original per-hit score files | 22 UK satire + 22 external |
| Derived configurations (not new API inference) | combined logistic and constant quality gate |
| Excluded overlap from external supervised training | 91 labeled binary benchmark examples |
| Original satire context BERT-base PLL AUC | 0.896095 |
| Original satire context RoBERTa-large PLL AUC | 0.910182 |
| External, corpus-query-normalized RoBERTa-large PLL AUC | 0.903106 |
| External, corpus-query-normalized Claude AUC | 0.969881 |
| Leakage-free combined external raw AUC | 0.869407 |
| LLM majority versus second context annotator kappa, unweighted | 0.868235 |
| LLM majority versus second context annotator kappa, design-weighted | 0.782011 |
| Bootstrap | 2,000 paired class-stratified with replacement, multinomial equivalent |

Original-label sensitivity is in `results/metrics_initial_sensitivity.json`; the final audit result is in `results/metrics.json`. The original score rank files are unchanged. The originally quoted full-image grounding, 3,083 historical case labels, historic 30M corpus cap, ρ=0.79 and API temperature 0 **cannot** be proved from the supplied archives. The five-page PDF has **not** been edited; see `MANUSCRIPT_EXACT_CHANGES.md`.
