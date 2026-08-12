# Pairwise Significance Matrix — MEIRA (FIRE 2026)

> Loaded from the archived multi-seed run (`results/k10_s10/experiments.json`, 10 seeds). Simulated evaluation-harness data — see `model_sim.py`.

Metric for the matrices: **nDCG@10** (paired t-test, two-sided). p-values are **holm-corrected** for the family of all 28 pairwise tests per dataset. `*` p<0.05, `**` p<0.01, `***` p<0.001.

## FIRE-AgentIR-2026

Significant pairs at α=0.05: **28/28 raw** → **28/28** after holm correction.

| Model | BM25 | TF-IDF | Dense-IR | ColBERT-like | MEIRA-no-memory | MEIRA-no-decay | MEIRA-no-xai | MEIRA-full |
|---|---|---|---|---|---|---|---|---|
| BM25 | — | 0.0141* | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | <0.0001*** | 0.0141* | <0.0001*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0003*** | <0.0001*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | <0.0001*** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | <0.0001*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**nDCG@10 (mean±std across seeds):** BM25=0.810±0.015, TF-IDF=0.807±0.016, Dense-IR=0.901±0.015, ColBERT-like=0.937±0.012, MEIRA-no-memory=0.930±0.014, MEIRA-no-decay=0.941±0.011, MEIRA-no-xai=0.951±0.010, MEIRA-full=0.969±0.008

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
| BM25 | — | 0.0458* | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| TF-IDF |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| Dense-IR |  |  | — | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** | <0.0001*** |
| ColBERT-like |  |  |  | — | 0.0023** | 0.0023** | 0.0007*** | <0.0001*** |
| MEIRA-no-memory |  |  |  |  | — | 0.0005*** | 0.0005*** | <0.0001*** |
| MEIRA-no-decay |  |  |  |  |  | — | 0.0013** | <0.0001*** |
| MEIRA-no-xai |  |  |  |  |  |  | — | 0.0003*** |
| MEIRA-full |  |  |  |  |  |  |  | — |

**nDCG@10 (mean±std across seeds):** BM25=0.817±0.013, TF-IDF=0.815±0.014, Dense-IR=0.896±0.017, ColBERT-like=0.932±0.012, MEIRA-no-memory=0.922±0.015, MEIRA-no-decay=0.937±0.011, MEIRA-no-xai=0.945±0.008, MEIRA-full=0.962±0.008

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
