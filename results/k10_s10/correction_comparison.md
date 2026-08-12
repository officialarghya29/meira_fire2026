# Correction Robustness — Holm vs Bonferroni (FIRE 2026)

> Source: `results/k10_s10/significance_matrix_{metric}_{holm|bonferroni}.json` 
> (paired t-tests, family of all 28 pairwise comparisons per dataset × metric). Simulated evaluation-harness data — see `model_sim.py`.

**Reading the tables:** a pair is **lost** when it is significant at α=0.05 under Holm-Bonferroni but no longer under the stricter Bonferroni correction (×28). Holm is always at least as powerful as Bonferroni, so `raw ≥ Holm ≤ Bonferroni` per pair.

## 1. Significant pairs at α=0.05 (of 28)

| Metric | Dataset | raw | Holm | Bonferroni | lost |
|---|---|---|---|---|---|
| F1 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| F1 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| nDCG@10 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| nDCG@10 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| MAP | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| MAP | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| MRR | FIRE-AgentIR-2026 | 27/28 | 27/28 | 26/28 | 1 |
| MRR | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 26/28 | 1 |

## 2. Pairs lost under Bonferroni (significant under Holm)

| Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |
|---|---|---|---|---|---|---|
| nDCG@10 | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0105 | 0.0141 | 0.2927 | 3.22 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0071 | 0.0141 | 0.1979 | 3.47 |
| nDCG@10 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0458 | 0.0458 | 1.0000 | 2.32 |
| MAP | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0034 | 0.0068 | 0.0946 | 3.95 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | 3.77 |
| MAP | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0247 | 0.0247 | 0.6907 | 2.69 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0236 | 0.0472 | 0.6605 | 2.72 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0023 | 0.0045 | 0.0635 | 4.21 |

## 3. Full per-pair appendix (raw / Holm / Bonferroni p-values)

| Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |
|---|---|---|---|---|---|---|
| F1 | FIRE-AgentIR-2026 | BM25 > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 10.68 |
| F1 | FIRE-AgentIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 117.71 |
| F1 | FIRE-AgentIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 105.29 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 129.50 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 101.26 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 85.40 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 86.70 |
| F1 | FIRE-AgentIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 175.57 |
| F1 | FIRE-AgentIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 87.08 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 104.53 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 85.17 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 74.63 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 77.24 |
| F1 | FIRE-AgentIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 29.69 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 27.59 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 30.86 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 30.90 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 39.15 |
| F1 | FIRE-AgentIR-2026 | ColBERT-like > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 22.75 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 19.50 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 21.96 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 33.27 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 26.72 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 27.00 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 33.04 |
| F1 | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 18.45 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 31.53 |
| F1 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | <0.0001 | <0.0001 | 26.77 |
| F1 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0003 | 0.0003 | 0.0077 | 5.76 |
| F1 | FIRE-CrossLingIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 22.63 |
| F1 | FIRE-CrossLingIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 33.90 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 27.63 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 38.56 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 39.08 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 44.07 |
| F1 | FIRE-CrossLingIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 19.45 |
| F1 | FIRE-CrossLingIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 29.70 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 24.25 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 33.08 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 34.94 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 39.63 |
| F1 | FIRE-CrossLingIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 28.76 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 21.25 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 37.27 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 26.80 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 29.89 |
| F1 | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 11.58 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 11.00 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 14.96 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 20.45 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 15.48 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 14.75 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 20.47 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 11.90 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 20.43 |
| F1 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | <0.0001 | <0.0001 | 21.41 |
| nDCG@10 | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0105 | 0.0141 | 0.2927 | 3.22 |
| nDCG@10 | FIRE-AgentIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 21.56 |
| nDCG@10 | FIRE-AgentIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 28.80 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 24.67 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 27.88 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 29.97 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 34.38 |
| nDCG@10 | FIRE-AgentIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 22.69 |
| nDCG@10 | FIRE-AgentIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 29.47 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 25.30 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 28.06 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 30.25 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 34.45 |
| nDCG@10 | FIRE-AgentIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 20.88 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 18.34 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 18.59 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 19.99 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 23.27 |
| nDCG@10 | FIRE-AgentIR-2026 | ColBERT-like > MEIRA-no-memory | <0.0001 | <0.0001 | 0.0004 | 8.46 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0071 | 0.0141 | 0.1979 | 3.47 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 10.84 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 20.12 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | 0.0003 | 0.0024 | 6.74 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 11.22 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 18.09 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 14.83 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 17.79 |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | <0.0001 | <0.0001 | 13.41 |
| nDCG@10 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0458 | 0.0458 | 1.0000 | 2.32 |
| nDCG@10 | FIRE-CrossLingIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 21.49 |
| nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 48.43 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 42.23 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 64.11 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 53.21 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 37.04 |
| nDCG@10 | FIRE-CrossLingIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 21.75 |
| nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 51.33 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 40.04 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 66.06 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 51.44 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 36.45 |
| nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 10.52 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 13.37 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.90 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.33 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 14.66 |
| nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0023 | 0.0266 | 4.82 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0008 | 0.0023 | 0.0215 | 4.97 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > ColBERT-like | 0.0001 | 0.0007 | 0.0037 | 6.35 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 13.06 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | 0.0005 | 0.0020 | 6.89 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | 0.0005 | 0.0020 | 6.89 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 11.34 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | 0.0003 | 0.0013 | 0.0093 | 5.61 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 10.15 |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | 0.0003 | 0.0012 | 7.40 |
| MAP | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0034 | 0.0068 | 0.0946 | 3.95 |
| MAP | FIRE-AgentIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 22.04 |
| MAP | FIRE-AgentIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 30.69 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 25.59 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 29.34 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 31.89 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 37.08 |
| MAP | FIRE-AgentIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 22.69 |
| MAP | FIRE-AgentIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 30.63 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 25.76 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 28.80 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 31.32 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 36.27 |
| MAP | FIRE-AgentIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 23.15 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 19.83 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 19.84 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 20.96 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 24.89 |
| MAP | FIRE-AgentIR-2026 | ColBERT-like > MEIRA-no-memory | <0.0001 | <0.0001 | 0.0003 | 8.60 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | 3.77 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 11.27 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 21.14 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | 0.0002 | 0.0016 | 7.07 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 11.48 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 18.91 |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 14.96 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 18.47 |
| MAP | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | <0.0001 | <0.0001 | 14.31 |
| MAP | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0247 | 0.0247 | 0.6907 | 2.69 |
| MAP | FIRE-CrossLingIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 21.42 |
| MAP | FIRE-CrossLingIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 48.51 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 42.53 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 64.30 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 53.86 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 38.05 |
| MAP | FIRE-CrossLingIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 21.47 |
| MAP | FIRE-CrossLingIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 51.43 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 40.37 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 68.62 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 54.06 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 37.55 |
| MAP | FIRE-CrossLingIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 10.77 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 13.84 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 12.10 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.18 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 14.57 |
| MAP | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0019 | 0.0247 | 4.87 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0005 | 0.0019 | 0.0148 | 5.25 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > ColBERT-like | 0.0001 | 0.0007 | 0.0042 | 6.26 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 13.23 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > MEIRA-no-memory | <0.0001 | 0.0005 | 0.0019 | 6.94 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | 0.0005 | 0.0025 | 6.70 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 11.17 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | 0.0005 | 0.0019 | 0.0135 | 5.32 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 10.44 |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | 0.0002 | 0.0007 | 7.82 |
| MRR | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.3351 | 0.3351 | 1.0000 | 1.02 |
| MRR | FIRE-AgentIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 19.79 |
| MRR | FIRE-AgentIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 22.97 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 20.91 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 23.15 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 23.71 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 25.80 |
| MRR | FIRE-AgentIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 21.94 |
| MRR | FIRE-AgentIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 24.44 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 22.18 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 24.50 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 25.27 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 26.90 |
| MRR | FIRE-AgentIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 15.08 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 14.11 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 15.39 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 16.51 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 18.05 |
| MRR | FIRE-AgentIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0001 | 0.0005 | 0.0035 | 6.39 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0236 | 0.0472 | 0.6605 | 2.72 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > ColBERT-like | <0.0001 | <0.0001 | 0.0002 | 9.41 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 16.21 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > MEIRA-no-memory | 0.0003 | 0.0009 | 0.0080 | 5.72 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 10.21 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 14.86 |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-xai > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 12.06 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | <0.0001 | 14.31 |
| MRR | FIRE-AgentIR-2026 | MEIRA-full > MEIRA-no-xai | <0.0001 | <0.0001 | <0.0001 | 10.61 |
| MRR | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.1554 | 0.1554 | 1.0000 | 1.55 |
| MRR | FIRE-CrossLingIR-2026 | Dense-IR > BM25 | <0.0001 | <0.0001 | <0.0001 | 22.63 |
| MRR | FIRE-CrossLingIR-2026 | ColBERT-like > BM25 | <0.0001 | <0.0001 | <0.0001 | 50.09 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-memory > BM25 | <0.0001 | <0.0001 | <0.0001 | 44.62 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > BM25 | <0.0001 | <0.0001 | <0.0001 | 60.23 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > BM25 | <0.0001 | <0.0001 | <0.0001 | 49.31 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > BM25 | <0.0001 | <0.0001 | <0.0001 | 38.35 |
| MRR | FIRE-CrossLingIR-2026 | Dense-IR > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 23.26 |
| MRR | FIRE-CrossLingIR-2026 | ColBERT-like > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 53.93 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-memory > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 42.16 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 59.32 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 46.30 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > TF-IDF | <0.0001 | <0.0001 | <0.0001 | 38.20 |
| MRR | FIRE-CrossLingIR-2026 | ColBERT-like > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 10.03 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-memory > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.97 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.32 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 11.53 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > Dense-IR | <0.0001 | <0.0001 | <0.0001 | 15.48 |
| MRR | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0011 | 0.0032 | 0.0298 | 4.74 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0023 | 0.0045 | 0.0635 | 4.21 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > ColBERT-like | 0.0002 | 0.0009 | 0.0043 | 6.22 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > ColBERT-like | <0.0001 | <0.0001 | <0.0001 | 11.86 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > MEIRA-no-memory | 0.0001 | 0.0007 | 0.0028 | 6.58 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-memory | <0.0001 | 0.0004 | 0.0014 | 7.21 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-memory | <0.0001 | <0.0001 | <0.0001 | 12.33 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | 0.0002 | 0.0009 | 0.0053 | 6.06 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-decay | <0.0001 | <0.0001 | 0.0002 | 9.22 |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-full > MEIRA-no-xai | 0.0001 | 0.0009 | 0.0041 | 6.27 |
