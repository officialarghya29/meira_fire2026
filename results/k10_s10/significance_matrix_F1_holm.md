# Pairwise Significance Matrix — MEIRA (FIRE 2026)

> Loaded from the archived multi-seed run (`results/k10_s10/experiments.json`, 10 seeds). Simulated evaluation-harness data — see `model_sim.py`.

Metric for the matrices: **F1** (paired t-test, two-sided). p-values are **holm-corrected** for the family of all 28 pairwise tests per dataset. `*` p<0.05, `**` p<0.01, `***` p<0.001.

## FIRE-AgentIR-2026

Significant pairs at α=0.05: **28/28 raw** → **28/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | <0.0001*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | <0.0001*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**F1 (mean±std across seeds):** BM25=0.491±0.016, TF-IDF=0.480±0.017, Dense-IR=0.650±0.017, ColBERT-like=0.740±0.013, MEIRA-no-memory=0.716±0.014, MEIRA-no-decay=0.748±0.012, MEIRA-no-xai=0.777±0.013, MEIRA-full=0.826±0.014

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
| BM25 | — | 0.0003*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | <0.0001*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | <0.0001*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**F1 (mean±std across seeds):** BM25=0.464±0.011, TF-IDF=0.454±0.008, Dense-IR=0.591±0.026, ColBERT-like=0.680±0.029, MEIRA-no-memory=0.655±0.030, MEIRA-no-decay=0.694±0.027, MEIRA-no-xai=0.726±0.029, MEIRA-full=0.780±0.030

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
