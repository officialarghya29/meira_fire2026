# Alpha-Threshold Sensitivity Sweep — FIRE 2026 MEIRA

> Source: `results/k10_s10/significance_matrix_{metric}_{holm|bonferroni}.json` 
> (paired t-tests, family of all 28 pairwise tests per dataset × metric). p-values are α-independent; only the verdicts change with the threshold. Simulated evaluation-harness data — see `model_sim.py`.

Alphas swept: **0.01, 0.05, 0.10**. A pair is **lost** when it is significant under Holm-Bonferroni but not under the stricter Bonferroni at that α.

## 1. Significant pairs at α (of 28)

| α | Metric | Dataset | raw | Holm | Bonferroni | lost |
|---|---|---|---|---|---|---|
| 0.01 | F1 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.01 | F1 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.01 | nDCG@10 | FIRE-AgentIR-2026 | 27/28 | 26/28 | 26/28 | 0 |
| 0.01 | nDCG@10 | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 25/28 | 2 |
| 0.01 | MAP | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| 0.01 | MAP | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 24/28 | 3 |
| 0.01 | MRR | FIRE-AgentIR-2026 | 26/28 | 26/28 | 26/28 | 0 |
| 0.01 | MRR | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 25/28 | 2 |
| 0.05 | F1 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.05 | F1 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.05 | nDCG@10 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| 0.05 | nDCG@10 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| 0.05 | MAP | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| 0.05 | MAP | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| 0.05 | MRR | FIRE-AgentIR-2026 | 27/28 | 27/28 | 26/28 | 1 |
| 0.05 | MRR | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 26/28 | 1 |
| 0.10 | F1 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.10 | F1 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 28/28 | 0 |
| 0.10 | nDCG@10 | FIRE-AgentIR-2026 | 28/28 | 28/28 | 26/28 | 2 |
| 0.10 | nDCG@10 | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| 0.10 | MAP | FIRE-AgentIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| 0.10 | MAP | FIRE-CrossLingIR-2026 | 28/28 | 28/28 | 27/28 | 1 |
| 0.10 | MRR | FIRE-AgentIR-2026 | 27/28 | 27/28 | 26/28 | 1 |
| 0.10 | MRR | FIRE-CrossLingIR-2026 | 27/28 | 27/28 | 27/28 | 0 |

## 2. Lost pairs per α (Holm-significant, not Bonferroni)

| α | Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |
|---|---|---|---|---|---|---|---|
| 0.01 | nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0023 | 0.0266 | 4.82 |
| 0.01 | nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0008 | 0.0023 | 0.0215 | 4.97 |
| 0.01 | MAP | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0034 | 0.0068 | 0.0946 | 3.95 |
| 0.01 | MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | 3.77 |
| 0.01 | MAP | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0019 | 0.0247 | 4.87 |
| 0.01 | MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0005 | 0.0019 | 0.0148 | 5.25 |
| 0.01 | MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | 0.0005 | 0.0019 | 0.0135 | 5.32 |
| 0.01 | MRR | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0011 | 0.0032 | 0.0298 | 4.74 |
| 0.01 | MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0023 | 0.0045 | 0.0635 | 4.21 |
| 0.05 | nDCG@10 | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0105 | 0.0141 | 0.2927 | 3.22 |
| 0.05 | nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0071 | 0.0141 | 0.1979 | 3.47 |
| 0.05 | nDCG@10 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0458 | 0.0458 | 1.0000 | 2.32 |
| 0.05 | MAP | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0034 | 0.0068 | 0.0946 | 3.95 |
| 0.05 | MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | 3.77 |
| 0.05 | MAP | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0247 | 0.0247 | 0.6907 | 2.69 |
| 0.05 | MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0236 | 0.0472 | 0.6605 | 2.72 |
| 0.05 | MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0023 | 0.0045 | 0.0635 | 4.21 |
| 0.10 | nDCG@10 | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0105 | 0.0141 | 0.2927 | 3.22 |
| 0.10 | nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0071 | 0.0141 | 0.1979 | 3.47 |
| 0.10 | nDCG@10 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0458 | 0.0458 | 1.0000 | 2.32 |
| 0.10 | MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | 3.77 |
| 0.10 | MAP | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0247 | 0.0247 | 0.6907 | 2.69 |
| 0.10 | MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0236 | 0.0472 | 0.6605 | 2.72 |

## 3. α-sensitive pairs (verdict flips or lost at any α)

✓ = significant, ✗ = not significant at that α.

| Metric | Dataset | Pair | raw p | Holm p | Bonf p | Holm @0.01 | @0.05 | @0.10 | Bonf @0.01 | @0.05 | @0.10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MAP | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0034 | 0.0068 | 0.0946 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| MAP | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0044 | 0.0068 | 0.1241 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-xai > MEIRA-no-decay | 0.0005 | 0.0019 | 0.0135 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| MAP | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0005 | 0.0019 | 0.0148 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| MAP | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0019 | 0.0247 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| MAP | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0247 | 0.0247 | 0.6907 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MRR | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0236 | 0.0472 | 0.6605 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MRR | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0011 | 0.0032 | 0.0298 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| MRR | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0023 | 0.0045 | 0.0635 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| nDCG@10 | FIRE-AgentIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0071 | 0.0141 | 0.1979 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| nDCG@10 | FIRE-AgentIR-2026 | BM25 > TF-IDF | 0.0105 | 0.0141 | 0.2927 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| nDCG@10 | FIRE-CrossLingIR-2026 | MEIRA-no-decay > ColBERT-like | 0.0008 | 0.0023 | 0.0215 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| nDCG@10 | FIRE-CrossLingIR-2026 | ColBERT-like > MEIRA-no-memory | 0.0009 | 0.0023 | 0.0266 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| nDCG@10 | FIRE-CrossLingIR-2026 | BM25 > TF-IDF | 0.0458 | 0.0458 | 1.0000 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
