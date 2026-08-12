# Pairwise Significance Matrix — MEIRA (FIRE 2026)

> Loaded from the archived multi-seed run (`results/k10_s10/experiments.json`, 10 seeds). Simulated evaluation-harness data — see `model_sim.py`.

Metric for the matrices: **MRR** (paired t-test, two-sided). p-values are **holm-corrected** for the family of all 28 pairwise tests per dataset. `*` p<0.05, `**` p<0.01, `***` p<0.001.

## FIRE-AgentIR-2026

Significant pairs at α=0.05: **27/28 raw** → **27/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 0.3351 | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | 0.0005*** | 0.0472* | <0.0001*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0009*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | <0.0001*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | <0.0001*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**MRR (mean±std across seeds):** BM25=0.506±0.017, TF-IDF=0.504±0.018, Dense-IR=0.588±0.019, ColBERT-like=0.617±0.021, MEIRA-no-memory=0.612±0.021, MEIRA-no-decay=0.620±0.019, MEIRA-no-xai=0.628±0.020, MEIRA-full=0.644±0.020

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

Significant pairs at α=0.05: **27/28 raw** → **27/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 0.1554 | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | 0.0032** | 0.0045** | 0.0009*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0007*** | 0.0004*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | 0.0009*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | 0.0009*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**MRR (mean±std across seeds):** BM25=0.423±0.017, TF-IDF=0.422±0.018, Dense-IR=0.482±0.018, ColBERT-like=0.509±0.015, MEIRA-no-memory=0.501±0.017, MEIRA-no-decay=0.511±0.015, MEIRA-no-xai=0.517±0.014, MEIRA-full=0.529±0.012

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
