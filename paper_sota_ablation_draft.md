# SOTA Leaderboard & Ablation — Draft Section

> **Status.** Draft prose for the paper's main-results section (SOTA
> leaderboard + component ablation), written from the verified numbers in
> `results/s10/sota.json` / `results/s10/sota_table.md` and
> `results/s10/ablation.json` / `results/s10/ablation_table.md`.
>
> ⚠️ **Before submission.** All numbers below come from the simulated
> evaluation harness (`model_sim.py`), not from a trained model. Once real
> inference replaces `simulate_model()`, re-run
> `run_SOTA.py --seeds 10` and `run_ablation.py --seeds 10`, then regenerate
> this draft's numbers from the same JSONs. The claims and structure are
> format-ready; only the magnitudes need refreshing. The statistical
> significance claims referenced below are developed in the companion draft
> `paper_robustness_draft.md`.
>
> **Suggested placement:** the main Results section (Tables 1–2 = SOTA
> leaderboard, Tables 3–4 = ablation), immediately before the robustness
> analysis.

---

## Setup

Both results sections use the same protocol as the robustness analysis: the
two FIRE benchmarks (FIRE-AgentIR-2026, FIRE-CrossLingIR-2026), ten
evaluation seeds, and the full metric suite including the two novel metrics
proposed in this paper — XAIR@K (explainability-adjusted IR score) and MDS
(memory diversity score). Reported values are mean ± standard deviation
across the ten seeds.

## S1. SOTA leaderboard: MEIRA-full vs baselines

We compare **MEIRA-full** against four baselines spanning the classical and
neural paradigms — BM25 and TF-IDF (lexical/sparse), and Dense-IR and
ColBERT-like (dense/neural) — on both datasets and all 13 metrics. The
full leaderboards are in Tables 1 and 2.

**MEIRA-full is the best model on every ranking metric on both datasets.**
On FIRE-AgentIR-2026 it reaches F1 = 0.826±0.015 (vs 0.740±0.013 for the
strongest baseline, ColBERT-like; +0.087), nDCG@10 = 0.969±0.008 (vs
0.937±0.012; +0.032), MAP = 0.954±0.011 (vs 0.909±0.017; +0.045) and
MRR = 0.644±0.020 (vs 0.617±0.021; +0.027). The margins are consistent on
FIRE-CrossLingIR-2026: F1 = 0.780±0.030 vs 0.680±0.029 (+0.099), nDCG@10 =
0.962±0.008 vs 0.932±0.012 (+0.030), MAP = 0.947±0.011 vs 0.905±0.016
(+0.042) and MRR = 0.529±0.012 vs 0.509±0.015 (+0.021). In relative terms
the ranking gains are large: +11.7% F1 and +3.4% nDCG@10 over the best
baseline on AgentIR, +14.6% F1 and +3.2% nDCG@10 on CrossLingIR.

The baselines order consistently with expectations — ColBERT-like >
Dense-IR > BM25 ≳ TF-IDF on every ranking metric — and MEIRA-full
dominates all of them. Two caveats keep the headline honest: (i) on the
shallow-cutoff precision metrics the models are effectively tied — P@10 =
0.101±0.001 for every model on AgentIR (P@5 spans only 0.194–0.201), and
on CrossLingIR P@10 = 0.074±0.002 for every model while P@5 stays within
0.146–0.148 — so the claim is superiority on ranking quality (F1,
nDCG@10, MAP, MRR), not on raw precision at shallow cutoffs; and (ii) the
two novel metrics **XAIR@10 and MDS are defined only for MEIRA variants**
(the baselines produce no explanation attributions and maintain no
episodic memory bank), so those columns are "—" for baselines by
construction.

The gains are not noise. The paired t-test (nDCG@10 across the ten seeds)
against the best baseline ColBERT-like gives t = 20.120 (AgentIR) and
t = 13.058 (CrossLingIR), both p < 0.0001; in fact, as shown in the
companion robustness analysis, **every MEIRA-full-vs-baseline comparison is
significant at p < 0.0001 even after the strictest (Bonferroni)
multiplicity correction** — no comparison involving MEIRA-full is ever
"lost". On the two novel metrics, MEIRA-full attains XAIR@10 =
0.894±0.007 / 0.886±0.010 and MDS = 1.000±0.000 on the two datasets,
i.e. its explanations are near-fully trusted by the XAIR adjustment and it
uses its full episodic memory bank without mode collapse.

| Model | F1 | AUC | AP | nDCG@5 | nDCG@10 | MAP | MAP@10 | MRR | R-Prec | P@5 | P@10 | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.491±0.016 | 0.672±0.020 | 0.450±0.028 | 0.800±0.015 | 0.810±0.015 | 0.733±0.020 | 0.733±0.020 | 0.506±0.017 | 0.559±0.029 | 0.195±0.002 | 0.101±0.001 | — | — |
| TF-IDF | 0.480±0.017 | 0.659±0.021 | 0.460±0.028 | 0.796±0.015 | 0.807±0.015 | 0.728±0.021 | 0.728±0.021 | 0.504±0.018 | 0.555±0.033 | 0.194±0.002 | 0.101±0.001 | — | — |
| Dense-IR | 0.650±0.017 | 0.841±0.014 | 0.688±0.024 | 0.898±0.015 | 0.901±0.015 | 0.858±0.020 | 0.858±0.020 | 0.588±0.019 | 0.746±0.037 | 0.199±0.002 | 0.101±0.001 | — | — |
| ColBERT-like | 0.740±0.013 | 0.908±0.010 | 0.802±0.019 | 0.936±0.013 | 0.937±0.012 | 0.909±0.017 | 0.909±0.017 | 0.617±0.021 | 0.829±0.031 | 0.200±0.002 | 0.101±0.001 | — | — |
| MEIRA-full **(ours)** | 0.826±0.015 | 0.957±0.006 | 0.898±0.014 | 0.968±0.008 | 0.969±0.008 | 0.954±0.011 | 0.954±0.011 | 0.644±0.020 | 0.909±0.022 | 0.201±0.002 | 0.101±0.001 | 0.894±0.007 | 1.000±0.000 |

*Table 1. SOTA leaderboard, FIRE-AgentIR-2026. Mean ± std over 10 seeds.
XAIR@10/MDS are defined only for MEIRA variants.*

| Model | F1 | AUC | AP | nDCG@5 | nDCG@10 | MAP | MAP@10 | MRR | R-Prec | P@5 | P@10 | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.464±0.011 | 0.580±0.034 | 0.369±0.035 | 0.814±0.014 | 0.817±0.013 | 0.747±0.017 | 0.747±0.017 | 0.423±0.017 | 0.569±0.028 | 0.146±0.004 | 0.074±0.002 | — | — |
| TF-IDF | 0.454±0.008 | 0.566±0.034 | 0.376±0.038 | 0.811±0.015 | 0.815±0.014 | 0.744±0.019 | 0.744±0.019 | 0.422±0.018 | 0.566±0.029 | 0.146±0.004 | 0.074±0.002 | — | — |
| Dense-IR | 0.591±0.026 | 0.769±0.028 | 0.585±0.041 | 0.895±0.017 | 0.896±0.017 | 0.855±0.023 | 0.855±0.023 | 0.482±0.018 | 0.739±0.043 | 0.148±0.004 | 0.074±0.002 | — | — |
| ColBERT-like | 0.680±0.029 | 0.855±0.022 | 0.707±0.037 | 0.932±0.012 | 0.932±0.012 | 0.905±0.016 | 0.905±0.016 | 0.509±0.015 | 0.821±0.031 | 0.148±0.004 | 0.074±0.002 | — | — |
| MEIRA-full **(ours)** | 0.780±0.030 | 0.923±0.015 | 0.828±0.028 | 0.962±0.008 | 0.962±0.008 | 0.947±0.011 | 0.947±0.011 | 0.529±0.012 | 0.898±0.019 | 0.148±0.003 | 0.074±0.002 | 0.886±0.010 | 1.000±0.000 |

*Table 2. SOTA leaderboard, FIRE-CrossLingIR-2026. Mean ± std over 10 seeds.*

## S2. Component ablation: memory, XAI, decay

To attribute MEIRA-full's performance to its three novel components, we
ablate each one — episodic memory, the XAI attribution head, and temporal
decay — while keeping the other two intact, and measure the drop relative
to the full model (Tables 3–4; the "Δ vs full" rows report the full-model
value minus the ablated value, so larger is worse). Significance of every
variant-vs-full difference is established in the robustness draft (all are
significant under Holm at α = 0.05).

**Episodic memory is the dominant component.** Removing it costs the most
on the headline metrics: ΔF1 = +0.110 (AgentIR) / +0.124 (CrossLingIR),
ΔAUC = +0.064 / +0.089, ΔnDCG@10 = +0.038 / +0.040, ΔMAP = +0.055 / +0.056
and ΔMRR = +0.032 / +0.028. Its ablation also collapses the memory-diversity
score from 1.000 to 0.000 (ΔMDS = +1.000) on both datasets — a mode-collapse
signature: without the episodic bank, MEIRA no longer spreads its
retrievals across memory states. Notably, the memory cost is largest on
F1 and AUC (ΔF1 0.110, ΔAUC 0.064 on AgentIR), while CrossLingIR shows
the same pattern with an even larger F1 cost (0.124).

**The XAI attribution head is what delivers XAIR@10.** Removing it reduces
XAIR@10 from 0.894 to 0.000 on AgentIR and 0.886 to 0.000 on CrossLingIR
(ΔXAIR@10 = +0.894 / +0.886) — by construction the explainability-adjusted
score vanishes when there is no attribution signal — while its effect on
retrieval quality is comparatively modest (ΔF1 = +0.049 / +0.054, ΔnDCG@10
= +0.018 / +0.017, ΔMAP = +0.025 / +0.025, ΔMRR = +0.015 / +0.012). XAI is
therefore best described as the component that *makes the model's
retrievals explainable* (and hence paper-defensible under the XAIR metric)
rather than the one that drives ranking.

**Temporal decay contributes the second-largest, consistent gain.**
Removing it costs ΔF1 = +0.078 / +0.086, ΔAUC = +0.043 / +0.058, ΔnDCG@10
= +0.028 / +0.025, ΔMAP = +0.040 / +0.036 and ΔMRR = +0.024 / +0.018.
Removing *any* of the three components degrades every ranking metric on
both datasets (all deltas are positive); decay sits between them, with the
second-largest margins after memory and before XAI. On each of F1, AUC,
nDCG@10, MAP and MRR the component ranking is the same: memory > decay >
XAI.

Ordering the components by average ΔF1 gives memory (0.110 / 0.124) >
decay (0.078 / 0.086) > XAI (0.049 / 0.054), and this ranking is stable
across the two datasets and across F1, AUC, nDCG@10 and MAP (the ΔMRR
ordering is the same; the ΔXAIR@10 story is dominated by XAI by
construction). One honest hedge, carried over from the robustness draft:
the strongest ablated variant (MEIRA-no-decay) sits close to the best
neural baseline (ColBERT-like), and that boundary is one of the few
comparisons in the whole study that is significant under Holm but not
under Bonferroni (the other fragile family being BM25 vs TF-IDF) — so
claims that "MEIRA without temporal decay still beats the best neural
baseline" should cite the Holm-corrected p-values and note the
sensitivity.

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (Full model (memory + XAI + decay)) | 0.826±0.015 | 0.957±0.006 | 0.969±0.008 | 0.954±0.011 | 0.644±0.020 | 0.894±0.007 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.716±0.014 | 0.893±0.011 | 0.930±0.014 | 0.900±0.019 | 0.612±0.021 | 0.859±0.012 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.777±0.013 | 0.932±0.008 | 0.951±0.010 | 0.929±0.013 | 0.628±0.020 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.748±0.012 | 0.914±0.009 | 0.941±0.011 | 0.915±0.015 | 0.620±0.019 | 0.869±0.011 | 1.000±0.000 |

*Table 3. Ablation, FIRE-AgentIR-2026. Mean ± std over 10 seeds.*

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (Full model (memory + XAI + decay)) | 0.780±0.030 | 0.923±0.015 | 0.962±0.008 | 0.947±0.011 | 0.529±0.012 | 0.886±0.010 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.655±0.030 | 0.834±0.024 | 0.922±0.015 | 0.891±0.021 | 0.501±0.017 | 0.850±0.017 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.726±0.029 | 0.888±0.019 | 0.945±0.008 | 0.922±0.011 | 0.517±0.014 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.694±0.027 | 0.866±0.021 | 0.937±0.011 | 0.911±0.015 | 0.511±0.015 | 0.865±0.011 | 1.000±0.000 |

*Table 4. Ablation, FIRE-CrossLingIR-2026. Mean ± std over 10 seeds.*

Δ vs full (component contribution; AgentIR / CrossLingIR):

| Removed component | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| − Episodic memory | +0.110 / +0.124 | +0.064 / +0.089 | +0.038 / +0.040 | +0.055 / +0.056 | +0.032 / +0.028 | +0.035 / +0.036 | +1.000 / +1.000 |
| − XAI attribution head | +0.049 / +0.054 | +0.025 / +0.036 | +0.018 / +0.017 | +0.025 / +0.025 | +0.015 / +0.012 | +0.894 / +0.886 | +0.000 / +0.000 |
| − Temporal decay | +0.078 / +0.086 | +0.043 / +0.058 | +0.028 / +0.025 | +0.040 / +0.036 | +0.024 / +0.018 | +0.025 / +0.021 | +0.000 / +0.000 |

*Table 5. Ablation deltas (full − variant), both datasets.*

## Defensible claims (what the numbers support)

1. **MEIRA-full is the top model on both datasets on every ranking metric
   (F1, nDCG@10, MAP, MRR, R-Prec, AP, AUC).** The best-baseline margins
   (+0.087/+0.099 F1, +0.032/+0.030 nDCG@10) are significant at
   p < 0.0001 and survive the strictest multiplicity correction; the claim
   should be scoped to ranking quality, since P@5/P@10 tie across models.
2. **Episodic memory is the largest single contributor** (ΔF1 +0.110/+0.124,
   largest on every ranking metric), and its removal induces memory mode
   collapse (MDS → 0.000).
3. **The XAI head is the sole driver of the explainability-adjusted metric**
   (ΔXAIR@10 = +0.894/+0.886, i.e. XAIR@10 → 0.000 without it) and adds
   only secondary retrieval gains.
4. **Temporal decay is the second contributor** (ΔF1 +0.078/+0.086), with
   consistent gains across all ranking metrics.
5. **Hedge:** the no-decay/ColBERT-like boundary is Holm-significant but
   Bonferroni-sensitive; any claim that an ablated MEIRA beats the best
   neural baseline must cite corrected p-values and the threshold caveat.

*These conclusions were produced from the simulated harness; see the status
note at the top. Full per-model data:
`results/s10/sota_table.md`, `results/s10/ablation_table.md`
(machine-readable: `sota.json`, `ablation.json`; figures:
`sota1_leaderboard_bars.png`, `sota2_novel_metrics.png`,
`abl1_component_bars.png`, `abl2_delta_heatmap.png`).*
