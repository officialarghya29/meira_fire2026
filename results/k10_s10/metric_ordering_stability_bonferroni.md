# Model-Ordering Stability Across Metrics — MEIRA (FIRE 2026)

> Loaded from the archived multi-seed run (`results/k10_s10/experiments.json`, 10 seeds). Simulated evaluation-harness data — see `model_sim.py`.

Ranks: 1 = best mean. Adjacent pairs marked with `†` are **not** significantly different — p-values are **bonferroni-corrected** for the family of all 28 pairwise tests per metric × dataset (p ≥ 0.05 after correction) — the places where the ordering could flip.

## FIRE-AgentIR-2026

### Rankings (per metric)

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| MEIRA-full | 1 | 1 | 1 | 1 |
| MEIRA-no-xai | 2 | 2 | 2 | 2 |
| MEIRA-no-decay | 3 | 3 | 3 | 3 |
| ColBERT-like | 4 | 4 | 4 | 4 |
| MEIRA-no-memory | 5 | 5 | 5 | 5 |
| Dense-IR | 6 | 6 | 6 | 6 |
| BM25 | 7 | 7 | 7 | 7 |
| TF-IDF | 8 | 8 | 8 | 8 |

### Mean performance (mean±std across seeds)

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| MEIRA-full | 0.826±0.015 | 0.969±0.008 | 0.954±0.011 | 0.644±0.020 |
| MEIRA-no-xai | 0.777±0.013 | 0.951±0.010 | 0.929±0.013 | 0.628±0.020 |
| MEIRA-no-decay | 0.748±0.012 | 0.941±0.011 | 0.915±0.015 | 0.620±0.019 |
| ColBERT-like | 0.740±0.013 | 0.937±0.012 | 0.909±0.017 | 0.617±0.021 |
| MEIRA-no-memory | 0.716±0.014 | 0.930±0.014 | 0.900±0.019 | 0.612±0.021 |
| Dense-IR | 0.650±0.017 | 0.901±0.015 | 0.858±0.020 | 0.588±0.019 |
| BM25 | 0.491±0.016 | 0.810±0.015 | 0.733±0.020 | 0.506±0.017 |
| TF-IDF | 0.480±0.017 | 0.807±0.015 | 0.728±0.021 | 0.504±0.018 |

### Adjacency significance (per metric, adjacent pairs only)

**F1:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0491 | 26.767 | <0.0001 | <0.0001 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0292 | 18.445 | <0.0001 | <0.0001 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0084 | 19.498 | <0.0001 | <0.0001 | yes |
| ColBERT-like | MEIRA-no-memory | +0.0235 | 22.747 | <0.0001 | <0.0001 | yes |
| MEIRA-no-memory | Dense-IR | +0.0662 | 27.592 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.1592 | 117.714 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0109 | 10.684 | <0.0001 | 0.0001 | yes |

**nDCG@10:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0178 | 13.409 | <0.0001 | <0.0001 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0102 | 14.831 | <0.0001 | <0.0001 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0038 | 3.468 | 0.0071 | 0.1979† | no |
| ColBERT-like | MEIRA-no-memory | +0.0065 | 8.461 | <0.0001 | 0.0004 | yes |
| MEIRA-no-memory | Dense-IR | +0.0291 | 18.339 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.0915 | 21.559 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0031 | 3.222 | 0.0105 | 0.2927† | no |

**MAP:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0250 | 14.310 | <0.0001 | <0.0001 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0146 | 14.963 | <0.0001 | <0.0001 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0056 | 3.768 | 0.0044 | 0.1241† | no |
| ColBERT-like | MEIRA-no-memory | +0.0095 | 8.596 | <0.0001 | 0.0003 | yes |
| MEIRA-no-memory | Dense-IR | +0.0416 | 19.831 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.1255 | 22.037 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0048 | 3.946 | 0.0034 | 0.0946† | no |

**MRR:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0153 | 10.612 | <0.0001 | 0.0001 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0085 | 12.057 | <0.0001 | <0.0001 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0030 | 2.720 | 0.0236 | 0.6605† | no |
| ColBERT-like | MEIRA-no-memory | +0.0051 | 6.390 | 0.0001 | 0.0035 | yes |
| MEIRA-no-memory | Dense-IR | +0.0239 | 14.111 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.0823 | 19.793 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0012 | 1.018 | 0.3351 | 1.0000† | no |

### Rank correlation between metric orderings (Spearman)

| Metric pair | ρ | p |
|---|---|---|
| F1 vs nDCG@10 | 1.0000 | 0.0000 |
| F1 vs MAP | 1.0000 | 0.0000 |
| F1 vs MRR | 1.0000 | 0.0000 |
| nDCG@10 vs MAP | 1.0000 | 0.0000 |
| nDCG@10 vs MRR | 1.0000 | 0.0000 |
| MAP vs MRR | 1.0000 | 0.0000 |

## FIRE-CrossLingIR-2026

### Rankings (per metric)

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| MEIRA-full | 1 | 1 | 1 | 1 |
| MEIRA-no-xai | 2 | 2 | 2 | 2 |
| MEIRA-no-decay | 3 | 3 | 3 | 3 |
| ColBERT-like | 4 | 4 | 4 | 4 |
| MEIRA-no-memory | 5 | 5 | 5 | 5 |
| Dense-IR | 6 | 6 | 6 | 6 |
| BM25 | 7 | 7 | 7 | 7 |
| TF-IDF | 8 | 8 | 8 | 8 |

### Mean performance (mean±std across seeds)

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| MEIRA-full | 0.780±0.030 | 0.962±0.008 | 0.947±0.011 | 0.529±0.012 |
| MEIRA-no-xai | 0.726±0.029 | 0.945±0.008 | 0.922±0.011 | 0.517±0.014 |
| MEIRA-no-decay | 0.694±0.027 | 0.937±0.011 | 0.911±0.015 | 0.511±0.015 |
| ColBERT-like | 0.680±0.029 | 0.932±0.012 | 0.905±0.016 | 0.509±0.015 |
| MEIRA-no-memory | 0.655±0.030 | 0.922±0.015 | 0.891±0.021 | 0.501±0.017 |
| Dense-IR | 0.591±0.026 | 0.896±0.017 | 0.855±0.023 | 0.482±0.018 |
| BM25 | 0.464±0.011 | 0.817±0.013 | 0.747±0.017 | 0.423±0.017 |
| TF-IDF | 0.454±0.008 | 0.815±0.014 | 0.744±0.019 | 0.422±0.018 |

### Adjacency significance (per metric, adjacent pairs only)

**F1:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0538 | 21.413 | <0.0001 | <0.0001 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0320 | 11.904 | <0.0001 | <0.0001 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0135 | 10.999 | <0.0001 | <0.0001 | yes |
| ColBERT-like | MEIRA-no-memory | +0.0248 | 11.584 | <0.0001 | <0.0001 | yes |
| MEIRA-no-memory | Dense-IR | +0.0650 | 21.250 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.1262 | 22.631 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0101 | 5.758 | 0.0003 | 0.0077 | yes |

**nDCG@10:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0171 | 7.399 | <0.0001 | 0.0012 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0082 | 5.607 | 0.0003 | 0.0093 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0044 | 4.973 | 0.0008 | 0.0215 | yes |
| ColBERT-like | MEIRA-no-memory | +0.0103 | 4.819 | 0.0009 | 0.0266 | yes |
| MEIRA-no-memory | Dense-IR | +0.0259 | 13.369 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.0789 | 21.491 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0021 | 2.316 | 0.0458 | 1.0000† | no |

**MAP:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0245 | 7.825 | <0.0001 | 0.0007 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0113 | 5.316 | 0.0005 | 0.0135 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0062 | 5.251 | 0.0005 | 0.0148 | yes |
| ColBERT-like | MEIRA-no-memory | +0.0140 | 4.873 | 0.0009 | 0.0247 | yes |
| MEIRA-no-memory | Dense-IR | +0.0357 | 13.844 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.1075 | 21.422 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0031 | 2.693 | 0.0247 | 0.6907† | no |

**MRR:**
| Higher | Lower | Δ mean | t | p raw | p (bonferroni) | sig |
|---|---|---|---|---|---|---|
| MEIRA-full | MEIRA-no-xai | +0.0117 | 6.268 | 0.0001 | 0.0041 | yes |
| MEIRA-no-xai | MEIRA-no-decay | +0.0061 | 6.055 | 0.0002 | 0.0053 | yes |
| MEIRA-no-decay | ColBERT-like | +0.0028 | 4.211 | 0.0023 | 0.0635† | no |
| ColBERT-like | MEIRA-no-memory | +0.0077 | 4.736 | 0.0011 | 0.0298 | yes |
| MEIRA-no-memory | Dense-IR | +0.0189 | 11.975 | <0.0001 | <0.0001 | yes |
| Dense-IR | BM25 | +0.0593 | 22.633 | <0.0001 | <0.0001 | yes |
| BM25 | TF-IDF | +0.0011 | 1.550 | 0.1554 | 1.0000† | no |

### Rank correlation between metric orderings (Spearman)

| Metric pair | ρ | p |
|---|---|---|
| F1 vs nDCG@10 | 1.0000 | 0.0000 |
| F1 vs MAP | 1.0000 | 0.0000 |
| F1 vs MRR | 1.0000 | 0.0000 |
| nDCG@10 vs MAP | 1.0000 | 0.0000 |
| nDCG@10 vs MRR | 1.0000 | 0.0000 |
| MAP vs MRR | 1.0000 | 0.0000 |
