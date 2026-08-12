# Pairwise Significance Matrix — MEIRA (FIRE 2026)

> Loaded from the archived multi-seed run (`results/k10_s10/experiments.json`, 10 seeds). Simulated evaluation-harness data — see `model_sim.py`.

Metric for the matrices: **MAP** (paired t-test, two-sided). p-values are **holm-corrected** for the family of all 28 pairwise tests per dataset. `*` p<0.05, `**` p<0.01, `***` p<0.001.

## FIRE-AgentIR-2026

Significant pairs at α=0.05: **28/28 raw** → **28/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 0.0068** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | <0.0001*** | 0.0068** | <0.0001*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0002*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | <0.0001*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | <0.0001*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**MAP (mean±std across seeds):** BM25=0.732±0.020, TF-IDF=0.728±0.021, Dense-IR=0.858±0.020, ColBERT-like=0.909±0.017, MEIRA-no-memory=0.900±0.019, MEIRA-no-decay=0.915±0.015, MEIRA-no-xai=0.929±0.013, MEIRA-full=0.954±0.011

**MEIRA-full vs each model (corrected p-values):**

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| BM25 | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-memory | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-xai | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |

## FIRE-CrossLingIR-2026

Significant pairs at α=0.05: **28/28 raw** → **28/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 0.0247* | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | 0.0019** | 0.0019** | 0.0007*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0005*** | 0.0005*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | 0.0019** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | 0.0002*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**MAP (mean±std across seeds):** BM25=0.747±0.017, TF-IDF=0.744±0.019, Dense-IR=0.855±0.023, ColBERT-like=0.905±0.016, MEIRA-no-memory=0.891±0.021, MEIRA-no-decay=0.911±0.015, MEIRA-no-xai=0.922±0.011, MEIRA-full=0.947±0.011

**MEIRA-full vs each model (corrected p-values):**

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| BM25 | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-memory | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-xai | <0.0001*** | 0.0003*** | 0.0002*** | 0.0009*** |
