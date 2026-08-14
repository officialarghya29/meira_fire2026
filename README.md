# MEIRA — FIRE 2026 / ACM SIGIR Evaluation Suite

Reproducible evaluation harness for **MEIRA** (Memory-Enhanced Interpretable
Retrieval Agent) on two synthetic-but-realistic FIRE-style IR benchmarks,
with k-fold + multi-seed experiments, an ablation study, and a SOTA
baseline comparison.

> ⚠️ **Read this before using results in a paper.**
> `model_sim.py` does **not** run a real trained model. It draws scores
> from calibrated numpy distributions so that the rest of the pipeline
> (metrics, splitting, statistics, figures) can be built and tested without
> a GPU or trained checkpoints. Every table/figure produced by these
> scripts is a **pipeline validation artifact**, not an experimental
> result. Before submitting to FIRE/SIGIR, replace `simulate_model()`
> with real forward passes from your trained MEIRA checkpoints — the rest
> of the code (metrics, ablation, SOTA, significance testing) does not
> need to change.

---

## 1. Datasets

### 1.1 FIRE-AgentIR-2026
Multi-turn agentic conversation retrieval, built in `datasets_fire.py::build_agent_ir_dataset()`.

- 10 IR/NLP topic clusters (episodic memory, dense retrieval, RAG, reranking,
  explainability, evaluation metrics, conversational search, query
  understanding, multimodal IR, agentic systems).
- 6 turns/conversation by default; later turns get progressively harder
  (fewer core terms retained → simulates topic drift).
- Hard negatives are drawn from a **sibling topic** that shares IR
  vocabulary, so the negative is topically confusable rather than random.
- Label noise (default 5%) simulates imperfect relevance judgements.

**Build it:**
```python
from datasets_fire import build_agent_ir_dataset, dataset_stats

samples = build_agent_ir_dataset(
    n_convs=350,       # number of conversations
    turns=6,           # turns per conversation
    neg_per_pos=3,      # hard/easy negatives per positive
    hard_ratio=0.65,    # fraction of negatives that are "hard"
    label_noise=0.05,
    seed=42,
)
print(dataset_stats(samples, "FIRE-AgentIR-2026"))
```

### 1.2 FIRE-CrossLingIR-2026
Cross-lingual / Indian-language IR, built in `datasets_fire.py::build_crossling_ir_dataset()`.

- 5 bilingual topic clusters (Hindi-English health & agriculture,
  Bengali-English news & education, Tamil-English technology).
- Each query/document is generated in English, transliterated-vernacular,
  or **code-switched (mixed)** form — aligned with FIRE's historical
  Indian-language IR tracks.
- Same hard-negative-sibling and label-noise design as AgentIR.

**Build it:**
```python
from datasets_fire import build_crossling_ir_dataset, dataset_stats

samples = build_crossling_ir_dataset(
    n_convs=200, turns=4, neg_per_pos=3,
    hard_ratio=0.60, label_noise=0.06, seed=42,
)
print(dataset_stats(samples, "FIRE-CrossLingIR-2026"))
```

### 1.3 Splitting utilities
Both datasets expose the same `IRSample` objects and split API:

```python
from datasets_fire import stratified_split, kfold_split

train, val, test = stratified_split(samples, ratios=(0.70, 0.15, 0.15), seed=42)
folds = kfold_split(samples, k=5, seed=42)   # list of (train, val) pairs
```

---

## 2. Models & metrics (what gets tested)

`model_sim.py::MODEL_REGISTRY` contains the baselines and the novel model:

| Model | Memory | XAI | Decay | Role |
|---|---|---|---|---|
| BM25, TF-IDF | – | – | – | classical IR baselines |
| Dense-IR, ColBERT-like | – | – | – | neural IR baselines |
| MEIRA-no-memory / -no-xai / -no-decay | partial | partial | partial | ablation variants |
| **MEIRA-full** | ✓ | ✓ | ✓ | **the paper's proposed model** |

`ir_metrics.py` implements the standard suite (F1, Precision, Recall,
Accuracy, AUC, AP, nDCG@K, MAP, MAP@K, MRR, R-Precision, P@K) **plus two
novel metrics proposed in this paper**, both evaluated on **both**
FIRE datasets in every script below:

- **XAIR@K** — eXplainability-Adjusted IR score: penalises correct
  retrievals that the model can't explain (low XAI attribution
  confidence).
- **MDS** — Memory Diversity Score: fraction of the episodic memory
  bank actually used, a mode-collapse indicator.

---

## 3. Step-by-step: running the full suite

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# (requirements.txt: numpy, scipy, scikit-learn, matplotlib, bibtexparser)

# 1. Core k-fold + multi-seed robustness experiments (both datasets)
python run_experiments.py --k 5 --seeds 5

# 2. Ablation study — memory / XAI / decay component contribution
python run_ablation.py --seeds 5

# 3. SOTA leaderboard — MEIRA-full vs BM25/TF-IDF/Dense-IR/ColBERT-like
python run_SOTA.py --seeds 5
```

Each script:
1. Builds **both** `FIRE-AgentIR-2026` and `FIRE-CrossLingIR-2026`.
2. Runs the requested experiment (k-fold, multi-seed ablation, or
   multi-seed SOTA comparison) on **each dataset separately**.
3. Computes the full metric suite — including XAIR@10/@5 and MDS — per
   run.
4. Aggregates mean/std (and min/max for `run_experiments.py`) across
   folds/seeds.
5. Saves JSON results to `results/<tag>/` and figures to `figures/<tag>/`,
   where `<tag>` encodes the run configuration (e.g. `k10_s10`) so different
   configurations are archived side-by-side instead of overwriting each other.

### Outputs

Each run writes into **config-named subfolders** so configurations are
archived side-by-side. The tag encodes the run: `run_experiments.py
--k 10 --seeds 10` → `k10_s10`; `run_ablation.py` / `run_SOTA.py`
(which take only `--seeds`) → `s10`. Re-running the same configuration
writes to the same subfolder (runs are deterministic given the fixed
seed range, so this is idempotent); different configurations never
clobber each other.

```
results/
  k5_s5/                       # run_experiments.py --k 5 --seeds 5
    experiments.json
  k10_s10/                     # run_experiments.py --k 10 --seeds 10
    experiments.json
    significance_matrix_nDCG@10_holm.json/.md  # run_significance.py --metric …
    significance_matrix_F1_holm.json/.md      #   (one file set per metric × correction)
    significance_matrix_MRR_holm.json/.md
    significance_matrix_*_bonferroni.json/.md #   Bonferroni variants (--correction)
    metric_ordering_stability_holm.json/.md   # compare_metric_orderings.py
    metric_ordering_stability_bonferroni.json/.md
    correction_comparison.json/.md            # compare_corrections.py (Holm vs Bonferroni)
    alpha_sweep.json/.md                      # sweep_alpha.py (α ∈ {0.01, 0.05, 0.10})
  s10/                         # run_ablation.py / run_SOTA.py --seeds 10
    ablation.json              #   per-variant, per-seed metrics + deltas
    ablation_table.md          #   paper-ready ablation table
    sota.json                  #   per-model, per-seed + significance test
    sota_table.md              #   paper-ready leaderboard table

figures/
  k5_s5/   exp1_kfold_boxplot.png ... exp6_xair_delta.png
  k10_s10/ exp1_kfold_boxplot.png ... exp6_xair_delta.png
           sig1_pairwise_heatmap_*_{holm,bonferroni}.png   (one per metric × correction)
           ord1_ordering_stability_{holm,bonferroni}.png
           corr1_correction_comparison.png    (status grids, Holm vs Bonferroni)
           sweep1_alpha_counts.png            (significant counts vs α)
  s10/     abl1_component_bars.png  abl2_delta_heatmap.png
           sota1_leaderboard_bars.png  sota2_novel_metrics.png
```

### Recommended run order for a paper's experiments section
1. `run_experiments.py` → dataset stats + robustness (k-fold, multi-seed).
2. `run_SOTA.py` → main results table (Table 1: MEIRA-full vs baselines).
3. `run_ablation.py` → ablation table (Table 2: component contribution).

Use `--seeds` (and `--k` for `run_experiments.py`) to scale rigor; 5 seeds
/ 5 folds is a reasonable default, 10+ is more defensible for a camera-ready
submission.

---

## 4. Current results (10 folds / 10 seeds)

> ⚠️ These are **pipeline-validation numbers** from the simulated harness
> (see the warning at the top of this README). Swap `simulate_model()` for
> real trained-model inference before citing them in a submission.

Configuration used for the numbers below:

| Script | Flags | Output folder |
|---|---|---|
| `run_experiments.py` | `--k 10 --seeds 10` | `results/k10_s10/` |
| `run_ablation.py` | `--seeds 10` | `results/s10/` |
| `run_SOTA.py` | `--seeds 10` | `results/s10/` |

The tables below mirror the generated paper-ready files
(`results/s10/sota_table.md`, `results/s10/ablation_table.md`), which are
regenerated on every run and are the single source of truth.

### 4.1 SOTA leaderboard — MEIRA-full vs baselines

**FIRE-AgentIR-2026**

| Model | F1 | AUC | AP | nDCG@5 | nDCG@10 | MAP | MAP@10 | MRR | R-Prec | P@5 | P@10 | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.491±0.016 | 0.672±0.020 | 0.450±0.028 | 0.800±0.015 | 0.810±0.015 | 0.733±0.020 | 0.733±0.020 | 0.506±0.017 | 0.559±0.029 | 0.195±0.002 | 0.101±0.001 | — | — |
| TF-IDF | 0.480±0.017 | 0.659±0.021 | 0.460±0.028 | 0.796±0.015 | 0.807±0.015 | 0.728±0.021 | 0.728±0.021 | 0.504±0.018 | 0.555±0.033 | 0.194±0.002 | 0.101±0.001 | — | — |
| Dense-IR | 0.650±0.017 | 0.841±0.014 | 0.688±0.024 | 0.898±0.015 | 0.901±0.015 | 0.858±0.020 | 0.858±0.020 | 0.588±0.019 | 0.746±0.037 | 0.199±0.002 | 0.101±0.001 | — | — |
| ColBERT-like | 0.740±0.013 | 0.908±0.010 | 0.802±0.019 | 0.936±0.013 | 0.937±0.012 | 0.909±0.017 | 0.909±0.017 | 0.617±0.021 | 0.829±0.031 | 0.200±0.002 | 0.101±0.001 | — | — |
| **MEIRA-full (ours)** | 0.826±0.015 | 0.957±0.006 | 0.898±0.014 | 0.968±0.008 | 0.969±0.008 | 0.954±0.011 | 0.954±0.011 | 0.644±0.020 | 0.909±0.022 | 0.201±0.002 | 0.101±0.001 | 0.894±0.007 | 1.000±0.000 |

**FIRE-CrossLingIR-2026**

| Model | F1 | AUC | AP | nDCG@5 | nDCG@10 | MAP | MAP@10 | MRR | R-Prec | P@5 | P@10 | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.464±0.011 | 0.580±0.034 | 0.369±0.035 | 0.814±0.014 | 0.817±0.013 | 0.747±0.017 | 0.747±0.017 | 0.423±0.017 | 0.569±0.028 | 0.146±0.004 | 0.074±0.002 | — | — |
| TF-IDF | 0.454±0.008 | 0.566±0.034 | 0.376±0.038 | 0.811±0.015 | 0.815±0.014 | 0.744±0.019 | 0.744±0.019 | 0.422±0.018 | 0.566±0.029 | 0.146±0.004 | 0.074±0.002 | — | — |
| Dense-IR | 0.591±0.026 | 0.769±0.028 | 0.585±0.041 | 0.895±0.017 | 0.896±0.017 | 0.855±0.023 | 0.855±0.023 | 0.482±0.018 | 0.739±0.043 | 0.148±0.004 | 0.074±0.002 | — | — |
| ColBERT-like | 0.680±0.029 | 0.855±0.022 | 0.707±0.037 | 0.932±0.012 | 0.932±0.012 | 0.905±0.016 | 0.905±0.016 | 0.509±0.015 | 0.821±0.031 | 0.148±0.004 | 0.074±0.002 | — | — |
| **MEIRA-full (ours)** | 0.780±0.030 | 0.923±0.015 | 0.828±0.028 | 0.962±0.008 | 0.962±0.008 | 0.947±0.011 | 0.947±0.011 | 0.529±0.012 | 0.898±0.019 | 0.148±0.003 | 0.074±0.002 | 0.886±0.010 | 1.000±0.000 |

**Significance (paired t-test on nDCG@10 across 10 seeds):**
- FIRE-AgentIR-2026: MEIRA-full (0.969) vs best baseline ColBERT-like
  (0.937) → t=20.120, p<0.0001 (significant at α=0.05).
- FIRE-CrossLingIR-2026: MEIRA-full (0.962) vs ColBERT-like (0.932) →
  t=13.058, p<0.0001 (significant at α=0.05).

### 4.2 Ablation — component contribution

**FIRE-AgentIR-2026**

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (memory + XAI + decay) | 0.826±0.015 | 0.957±0.006 | 0.969±0.008 | 0.954±0.011 | 0.644±0.020 | 0.894±0.007 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.716±0.014 | 0.893±0.011 | 0.930±0.014 | 0.900±0.019 | 0.612±0.021 | 0.859±0.012 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.777±0.013 | 0.932±0.008 | 0.951±0.010 | 0.929±0.013 | 0.628±0.020 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.748±0.012 | 0.914±0.009 | 0.941±0.011 | 0.915±0.015 | 0.620±0.019 | 0.869±0.011 | 1.000±0.000 |

**FIRE-CrossLingIR-2026**

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (memory + XAI + decay) | 0.780±0.030 | 0.923±0.015 | 0.962±0.008 | 0.947±0.011 | 0.529±0.012 | 0.886±0.010 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.655±0.030 | 0.834±0.024 | 0.922±0.015 | 0.891±0.021 | 0.501±0.017 | 0.850±0.017 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.726±0.029 | 0.888±0.019 | 0.945±0.008 | 0.922±0.011 | 0.517±0.014 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.694±0.027 | 0.866±0.021 | 0.937±0.011 | 0.911±0.015 | 0.511±0.015 | 0.865±0.011 | 1.000±0.000 |

Δ vs full (component contribution):

| Removed component | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| − Episodic memory (AgentIR / CrossLingIR) | +0.110 / +0.124 | +0.064 / +0.089 | +0.038 / +0.040 | +0.055 / +0.056 | +0.032 / +0.028 | +0.035 / +0.036 | +1.000 / +1.000 |
| − XAI attribution head (AgentIR / CrossLingIR) | +0.049 / +0.054 | +0.025 / +0.036 | +0.018 / +0.017 | +0.025 / +0.025 | +0.015 / +0.012 | +0.894 / +0.886 | +0.000 / +0.000 |
| − Temporal decay (AgentIR / CrossLingIR) | +0.078 / +0.086 | +0.043 / +0.058 | +0.028 / +0.025 | +0.040 / +0.036 | +0.024 / +0.018 | +0.025 / +0.021 | +0.000 / +0.000 |

Headline: episodic memory is the largest contributor (ΔF1 +0.110 / +0.124
on AgentIR / CrossLingIR), the XAI head drives XAIR@10 (+0.894 / +0.886),
and temporal decay adds ΔF1 +0.078 / +0.086.

### 4.3 Multi-seed robustness — MEIRA-full (mean ± std, 10 seeds)

| Dataset | F1 | AUC | nDCG@5 | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|---|
| FIRE-AgentIR-2026 | 0.826±0.015 | 0.957±0.006 | 0.968±0.008 | 0.969±0.008 | 0.954±0.011 | 0.644±0.020 | 0.894±0.007 | 1.000±0.000 |
| FIRE-CrossLingIR-2026 | 0.780±0.030 | 0.923±0.015 | 0.962±0.008 | 0.962±0.008 | 0.947±0.011 | 0.529±0.012 | 0.886±0.010 | 1.000±0.000 |

### 4.4 Model-ordering stability across metrics

Produced by `compare_metric_orderings.py` (mirrors
`results/k10_s10/metric_ordering_stability_holm.md`). All significance flags
use **Holm-Bonferroni-corrected p-values** (family of all 28 pairwise
tests per metric × dataset; see `multi_correction.py`). The 8-model
ranking is **identical across all four metrics and both datasets** —
Spearman ρ = 1.0000 between every metric pair:

| Model | F1 | nDCG@10 | MAP | MRR |
|---|---|---|---|---|
| **MEIRA-full** | 1 | 1 | 1 | 1 |
| MEIRA-no-xai | 2 | 2 | 2 | 2 |
| MEIRA-no-decay | 3 | 3 | 3 | 3 |
| ColBERT-like | 4 | 4 | 4 | 4 |
| MEIRA-no-memory | 5 | 5 | 5 | 5 |
| Dense-IR | 6 | 6 | 6 | 6 |
| BM25 | 7 | 7 | 7 | 7 |
| TF-IDF | 8 | 8 | 8 | 8 |

(Ranks shown once — they are the same on both datasets.) All 7 adjacent
pairs are significantly separated under F1, nDCG@10 and MAP; under MRR
only one adjacency fails to reach α=0.05. Every pair that is significant
at raw α=0.05 **remains significant after Holm-Bonferroni correction** —
the marginal pairs are the largest p-values in their 28-test family, so
Holm's multiplier for them is only 1–2. (A stricter Bonferroni would
instead drop the marginal pairs: 26/28 and 27/28 significant for nDCG@10,
26/28 for MRR.)

**Fragile adjacency — BM25 vs TF-IDF** (the only non-significant pair,
and the place where the ordering could flip; p-values below are
Holm-corrected):

| Metric | AgentIR p | CrossLingIR p |
|---|---|---|
| F1 | <0.0001 ✓ | 0.0003 ✓ |
| MAP | 0.0068 ✓ | 0.0247 ✓ |
| nDCG@10 | 0.0141 ✓ | 0.0458 ✓ |
| MRR | 0.3351 ✗ | 0.1554 ✗ |

BM25 stays ahead of TF-IDF on every metric on both datasets, but the gap
is only ~0.003–0.011 nDCG@10 — statistically fragile (significant under
F1, marginal under nDCG@10/MAP, non-significant under MRR). A second,
milder fragility sits at the ablation-variant/baseline boundary:
**MEIRA-no-decay vs ColBERT-like** (nDCG@10 p=0.0071 raw → 0.0141
corrected on AgentIR / 0.0008 → 0.0023 on CrossLingIR), so claims that
MEIRA without temporal decay beats the best neural baseline should be
phrased carefully.

Per-metric pairwise significance matrices are archived alongside this as
`results/k10_s10/significance_matrix_{metric}_{correction}.json/.md` for
F1, nDCG@10, MAP and MRR under both `holm` and `bonferroni` corrections
(each records raw + corrected p-values; `--correction none` omits the
suffix).

### 4.5 Correction robustness — Holm vs Bonferroni (appendix)

Produced by `compare_corrections.py` (mirrors `results/k10_s10/correction_comparison.md`;
raw data in `correction_comparison.json`). A pair is **lost** when it is significant at
α=0.05 under Holm-Bonferroni but not under the stricter Bonferroni (×28).

**Significant pairs at α=0.05 (of 28):**

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

**The 8 lost pair-instances (Holm-significant, dropped by Bonferroni):**

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

Takeaways for the paper's robustness appendix:

- **No MEIRA-full pair is ever lost** — every other model vs MEIRA-full
  stays significant (p<0.0001) even after Bonferroni on every metric.
- Lost pairs only ever involve the two already-fragile spots from §4.4:
  **BM25 vs TF-IDF** (the classical pair) and **MEIRA-no-decay vs
  ColBERT-like** (ablation-variant/baseline boundary).
- Under MRR, BM25 vs TF-IDF is never significant (raw p=0.3351), so it is
  *not* “lost” — it never passed in the first place; the MRR lost pair is
  always MEIRA-no-decay vs ColBERT-like.
- `figures/k10_s10/corr1_correction_comparison.png` shows per-pair status
  (green = significant under both, orange = lost under Bonferroni,
  grey = never significant) for every metric × dataset.

### 4.6 Alpha-threshold sensitivity (α ∈ {0.01, 0.05, 0.10})

Produced by `sweep_alpha.py` (mirrors `results/k10_s10/alpha_sweep.md`; raw
data in `alpha_sweep.json`). p-values are α-independent — only the verdicts
change with the threshold. Cells below are **A/C** = AgentIR / CrossLingIR.

| α | Metric | raw (A/C) | Holm (A/C) | Bonferroni (A/C) | lost (A/C) |
|---|---|---|---|---|---|
| 0.01 | F1 | 28/28 | 28/28 | 28/28 | 0/0 |
| 0.01 | nDCG@10 | 27/27 | 26/27 | 26/25 | 0/2 |
| 0.01 | MAP | 28/27 | 28/27 | 26/24 | 2/3 |
| 0.01 | MRR | 26/27 | 26/27 | 26/25 | 0/2 |
| 0.05 | F1 | 28/28 | 28/28 | 28/28 | 0/0 |
| 0.05 | nDCG@10 | 28/28 | 28/28 | 26/27 | 2/1 |
| 0.05 | MAP | 28/28 | 28/28 | 26/27 | 2/1 |
| 0.05 | MRR | 27/27 | 27/27 | 26/26 | 1/1 |
| 0.10 | F1 | 28/28 | 28/28 | 28/28 | 0/0 |
| 0.10 | nDCG@10 | 28/28 | 28/28 | 26/27 | 2/1 |
| 0.10 | MAP | 28/28 | 28/28 | 27/27 | 1/1 |
| 0.10 | MRR | 27/27 | 27/27 | 26/27 | 1/0 |

Reading: **F1 verdicts are invariant** — all 28 pairs significant under every
correction at every α. Holm is nearly α-stable: the only Holm verdicts that
move are the two nDCG@10-AgentIR marginals (BM25>TF-IDF, no-decay>ColBERT;
Holm p=0.0141), which drop at α=0.01. Bonferroni is where the threshold
bites hardest — lost pairs peak at 3 (MAP CrossLing @0.01) and shrink to 0
at α=0.10 (MRR CrossLing, where no-decay>ColBERT's Bonf p=0.0635 crosses;
MAP AgentIR's BM25>TF-IDF at Bonf p=0.0946 likewise gains significance). In
total 14 pair-instances are α-sensitive (verdict flips or lost at some α),
listed in full in `alpha_sweep.md` §3 and plotted in
`figures/k10_s10/sweep1_alpha_counts.png`. None of the α-sensitivity ever
involves MEIRA-full.

---

## 5. Swapping in a real model

Replace the body of `model_sim.py::simulate_model()` with a real forward
pass, keeping the same return signature:

```python
def simulate_model(samples, model_name="MEIRA-full", seed=42,
                    dataset_name="agent", memory_slots=64) -> dict:
    # real tokenized batch -> model.forward() -> return the same dict keys:
    # labels, probs, preds, threshold, xai_conf, ret_indices, conv_ids, turns
    ...
```

None of `ir_metrics.py`, `run_experiments.py`, `run_ablation.py`, or
`run_SOTA.py` need to change — they only depend on that dict's shape.

---

## 6. Paper pipeline: build, verify, test

The camera-ready paper is generated **programmatically** from
`paper_full_draft.md` so the verified numbers survive verbatim.

```bash
# 1. Regenerate the LaTeX skeleton (abstract, 6 sections, 9 tables,
#    3 figures, bibliography) from the assembled markdown
.venv/bin/python md2tex.py          # prints trims report + citation report

# 2. Compile (4 passes for correct refs/citations)
cd latex && ~/.local/bin/tectonic main.tex && cd ..
#   or: latexmk -pdf main.tex   (pdflatex + bibtex)

# 3. Structural validation (labels, refs, table/figure counts, number
#    fidelity vs the markdown - 58 checks)
.venv/bin/python latex/_validate_tex.py

# 4. Page-map report (label -> PDF page, per-page fill)
pdftotext -layout latex/main.pdf /tmp/paper.txt \
  && .venv/bin/python latex/_pagemap.py /tmp/paper.txt

# 5. Zero-collision figure audit (exit-code gated)
.venv/bin/python audit_figures.py
```

**Unit tests** (88 tests, ~0.2 s, no GPU):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

**Pre-commit hook** — runs the unit tests, the figure audit, and the paper
build on every commit (build outputs go to `/tmp`; tracked sources are
only regenerated to identical deterministic content, so the hook never
dirties the worktree):

```bash
git config core.hooksPath .githooks
# skip the paper build inside the hook, if tectonic is unavailable:
SKIP_PAPER=1 git commit …
```

**Reproduction notebook** — `notebook/Shared_Notebook_for_FIRE_2026_conf_08Aug26 (2).ipynb`
re-runs the full suite top-to-bottom (using `!{sys.executable}` so the
kernel's interpreter is always used) and inspects the archived JSONs.

> The 9-page content limit is enforced by the pipeline: references are
> forced onto a fresh page (`\clearpage`) and the trims in `_trims.py`
> keep the prose inside the budget. After any edit, re-run steps 1–3 and
> confirm `latex/main.pdf` still fits (0 overfull, 0 undefined).

---

## File manifest

| File | Purpose |
|---|---|
| `datasets_fire.py` | Builds FIRE-AgentIR-2026 & FIRE-CrossLingIR-2026, split utilities |
| `model_sim.py` | Model registry + score simulator (swap for real inference) |
| `ir_metrics.py` | Full metric suite incl. novel XAIR@K and MDS |
| `run_experiments.py` | K-fold + multi-seed robustness, both datasets, all models |
| `run_ablation.py` | Component ablation (memory/XAI/decay), both datasets |
| `run_SOTA.py` | Baseline leaderboard + significance testing, both datasets |
| `run_significance.py` | Full pairwise significance matrix across all 8 models (reads archived results) |
| `compare_metric_orderings.py` | Cross-metric ordering stability: rankings, close adjacencies, Spearman correlations |
| `multi_correction.py` | Bonferroni & Holm-Bonferroni multiple-comparison corrections (shared) |
| `compare_corrections.py` | Holm vs Bonferroni robustness appendix: summary counts, lost-pair lists, status figure |
| `sweep_alpha.py` | Alpha-threshold sensitivity sweep: verdicts & lost pairs at α ∈ {0.01, 0.05, 0.10} |
| `paper_robustness_draft.md` | Draft prose for the paper's robustness section (Holm vs Bonferroni + α sensitivity) |
| `paper_sota_ablation_draft.md` | Draft prose for the paper's results section (SOTA leaderboard + component ablation) |
| `paper_datasets_protocol_draft.md` | Draft prose for the paper's Methods section (datasets, evaluation protocol, metrics, models, statistics) |
| `paper_abstract_intro_draft.md` | Draft prose for the paper's Abstract + Introduction (contributions, findings at a glance) |
| `paper_related_work_draft.md` | Draft prose for the paper's Related Work section (positioning + citation verification checklist) |
| `paper_conclusion_draft.md` | Draft prose for the paper's Conclusion (findings, limitations, future work) |
| `paper_full_draft.md` | **Assembled complete paper** — Abstract → Conclusion stitched from the six verified drafts with consistent numbering & cross-references |
| `_assemble_paper.py` | Build tool for `paper_full_draft.md` — strips per-file front matter, renumbers sections/tables, rewrites cross-references (re-run after editing any draft) |
| `paper_references.bib` | BibTeX export of the 20 verified Related-Work references (DOIs/arXiv IDs confirmed vs publisher pages) |
| `figures/` | Publication-quality figures (300 dpi, ACM print-width) generated by the pipeline scripts — `s10/`: leaderboard bars, novel-metric bars, ablation component bars, ablation-delta heatmap; `k10_s10/`: ordering-stability rank plots (holm/bonferroni), pairwise significance heatmaps, alpha-sweep counts |
| `latex/` | **FIRE 2026 camera-ready LaTeX skeleton** (ACM `acmart` `sigconf`, 2-column) — `main.tex` + `sections/sec01..06.tex` + `.bib` copy; compiles clean with tectonic: content fills the full 9 pp limit (refs start p10, refs excluded from the count), 9 tables + **3 embedded figures**, zero undefined refs/citations/overfull; `main.pdf` built |
| `md2tex.py` | Build tool for `latex/` — converts `paper_full_draft.md` programmatically (headings, tables, math, `\Cref`/`\citep` wiring, markdown images `![caption](path){#label}` → `figure*` envs, drafting metadata → comments, camera-ready trims via `_trims.py`); re-run after editing the markdown |
| `_trims.py` | Camera-ready prose-trim table used by `md2tex.py` — 86 exact-match sentence tightenings (redundant prose only; every headline number, citation, and cross-reference preserved; chained trims rewrite text that an earlier trim in the chain produced); `md2tex.py` reports applied/never-applied (re-run `md2tex.py` after editing any draft and check the trims report — re-assembling `paper_full_draft.md` can invalidate exact-match trims) |
| `latex/_validate_tex.py` | Validator for the skeleton — label/reference integrity, structure order, 9 tables (caption+label), 3 figures (caption, label, image exists), number fidelity vs markdown (58 checks) |
| `latex/_pagemap.py` | Page-break reporter — label → PDF page from `main.aux` + per-page fill from `pdftotext` (pass the extracted text file as arg) |
| `audit_figures.py` | Zero-collision figure audit — regenerates the three paper figures through their real plotting code and measures pairwise text-bbox intersections, canvas clipping, title overflow, text-beyond-panel, and text-on-bar overlap; exit-code gated (fails the build/CI if any figure has a collision). Margins are empirically tuned — re-run after any plotting change |
| `tests/` | `unittest` suite (88 tests) — hand-computed metric checks, Bonferroni/Holm properties, dataset/split invariants, model-simulation determinism, significance helpers, md2tex abstract/trim regressions. Run: `.venv/bin/python -m unittest discover -s tests -v` |
| `.githooks/pre-commit` | Pre-commit hook — runs `audit_figures.py` + the paper build (md2tex + tectonic, fails on overfull/undefined refs); enable with `git config core.hooksPath .githooks`; skip the paper build with `SKIP_PAPER=1` |
| `notebook/` | `Shared_Notebook_for_FIRE_2026_conf_08Aug26 (2).ipynb` — end-to-end reproduction notebook: re-runs the whole suite with `!{sys.executable}` and inspects the archived camera-ready JSONs (leaderboard, ablation, significance, ordering stability, α-sweep) without extra dependencies |
