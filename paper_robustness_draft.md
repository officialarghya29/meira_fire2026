# Robustness of the Statistical Comparisons — Draft Section

> **Status.** Draft prose for the paper's robustness/statistical-significance
> section, written from the verified numbers in
> `results/k10_s10/correction_comparison.json/.md` and
> `results/k10_s10/alpha_sweep.json/.md`.
>
> ⚠️ **Before submission.** All numbers below come from the simulated
> evaluation harness (`model_sim.py`), not from a trained model. Once real
> inference replaces `simulate_model()`, re-run
> `run_experiments.py --k 10 --seeds 10`, then `run_significance.py`,
> `compare_corrections.py`, and `sweep_alpha.py`, and regenerate this draft's
> numbers from the same JSONs. The claims and structure are format-ready;
> only the magnitudes need refreshing.
>
> **Suggested placement:** end of the Experiments section (after the ablation
> and ordering-stability analyses), or as a dedicated appendix section.
> Two prose subsections (multiplicity correction, threshold sensitivity) plus
> a short "defensible claims" paragraph.

---

## Setup

We assess whether the performance differences reported above are
statistically reliable, and whether those conclusions survive choices about
how significance is decided. For each dataset and each of the four headline
metrics (F1, nDCG@10, MAP, MRR), we run the two-sided paired t-test across
the ten evaluation seeds (df = 9) for **every pair of the eight models**
(the direction of each difference is given by the sign of the t-statistic)
— 28 comparisons per dataset × metric, 224 in total. Because the
per-seed scores of all models come from the same test split within each
seed, the comparisons are naturally paired, and the t-statistic preserves
the direction of the difference (the model with the higher mean always
appears on the left of the reported inequality).

With 28 simultaneous tests per family, uncorrected p-values would
overstate evidence, so we apply multiplicity correction as the primary
analysis: **Holm-Bonferroni step-down adjustment** (Holm, 1979), which
controls the family-wise error rate while remaining strictly more powerful
than plain Bonferroni. As a stress test we additionally report **Bonferroni
correction** (threshold α/28 ≈ 0.0018 at α = 0.05). Throughout,
"significant" means corrected p < α, and a pair is said to be **lost**
when it is significant under Holm but not under Bonferroni.

## R1. Multiplicity correction: Holm vs Bonferroni

At α = 0.05 the raw analysis finds all 28 comparisons significant on both
datasets for F1, nDCG@10 and MAP, and 27 of 28 for MRR; the single
non-significant raw comparison is **BM25 vs TF-IDF** under MRR
(p = 0.3351 / 0.1554 on FIRE-AgentIR-2026 / FIRE-CrossLingIR-2026), a
near-tie between the two classical baselines that is never significant at
any threshold we consider.

**Holm-Bonferroni leaves every raw-significant comparison intact:** the
corrected counts are identical to the raw counts on every metric and
dataset (F1, nDCG@10, MAP: 28/28; MRR: 27/28). The reason is that Holm's
step-down procedure applies its smallest multipliers (1–2) to the largest
raw p-values, and the marginal comparisons — those whose raw p approaches
α = 0.05 (raw p ≈ 0.002–0.046) — still land below the threshold after that
adjustment (e.g. MRR on FIRE-AgentIR-2026: p = 0.0236 → 0.0472). All
remaining comparisons have raw p ≤ 0.001 and stay far below α under any
correction. Under the harsher Bonferroni correction, however, those same
marginal pairs drop out. Table 1 summarizes the counts; the eight lost
pair-instances are enumerated in Table 2.

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

*Table 1. Significant pairwise comparisons (of 28) at α = 0.05 under each
correction, per dataset × metric. "lost" = significant under Holm but not
under Bonferroni.*

The eight lost pair-instances belong to exactly two comparison families —
**BM25 vs TF-IDF** (the two classical baselines) and **MEIRA-no-decay vs
ColBERT-like** (the strongest ablation variant against the strongest
neural baseline) — and never involve MEIRA-full:

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

*Table 2. The eight pair-instances that are significant under Holm but not
under Bonferroni at α = 0.05.*

The important observation is what is **not** in Table 2. Every comparison
involving **MEIRA-full — including MEIRA-full vs the best baseline
ColBERT-like — remains significant at p < 0.0001 even after Bonferroni
correction** on every metric and dataset (e.g. nDCG@10: t = 20.12 and
t = 13.06 on FIRE-AgentIR-2026 and FIRE-CrossLingIR-2026, respectively,
vs uncorrected differences of 0.969–0.937 and 0.962–0.932). The paper's
headline superiority claim therefore does not depend on which correction
is chosen.

## R2. Threshold sensitivity: α ∈ {0.01, 0.05, 0.10}

To verify that the verdicts above are not an artifact of the α = 0.05
convention, we re-evaluate every comparison at α = 0.01 and α = 0.10
(p-values are threshold-independent; only the verdicts move). Two
conclusions hold across the full sweep:

**F1 is invariant.** All 28 comparisons are significant under *every*
correction at *every* α on both datasets. The F1 leaderboard — the metric
most central to our claims — is completely insensitive to both the
multiplicity correction and the threshold.

**Holm is nearly threshold-stable.** Across all 24 dataset × metric
conditions (2 datasets × 4 metrics × 3 thresholds), Holm's adjustment
itself flips exactly **one** raw-significant verdict: at the strictest
threshold α = 0.01, MEIRA-no-decay > ColBERT-like on FIRE-AgentIR-2026
(raw p = 0.0071 < 0.01, but Holm p = 0.0141 > 0.01) drops out. The other
count reductions at α = 0.01 — one pair each in nDCG@10 CrossLing, MAP
CrossLing and MRR AgentIR, plus BM25 > TF-IDF on nDCG@10 AgentIR — are
raw-threshold effects: those raw p-values (0.0105–0.0458) already exceed
0.01. The MRR raw non-significance of BM25 vs TF-IDF is inherited by Holm
at every threshold.

Bonferroni is where the threshold binds. Under the strictest threshold
α = 0.01, Bonferroni drops more comparisons (e.g. MAP on
FIRE-CrossLingIR-2026: 24/28, with three lost pair-instances, and the
CrossLing micro-gaps ColBERT-like > MEIRA-no-memory and
MEIRA-no-xai > MEIRA-no-decay joining the two fragile families), while at
the loosest threshold α = 0.10 it *recovers* exactly two of the eight
α = 0.05 losses — MRR no-decay > ColBERT-like on FIRE-CrossLingIR-2026
(Bonferroni p = 0.0635) and MAP BM25 > TF-IDF on FIRE-AgentIR-2026
(Bonferroni p = 0.0946). In total, 14 pair-instances across the sweep are
**α-sensitive** — their verdict changes with the threshold or they are
lost at some α — and, again, **none of them involves MEIRA-full**.
Consistently with R1, the unstable comparisons are exactly the BM25/TF-IDF
tie, the no-decay/ColBERT boundary, and a handful of tight CrossLing
ablation-tier gaps; in no case does a *direction* of an ordering flip, only
the significance verdict.

## Defensible claims (what the numbers support)

1. **MEIRA-full is robustly superior.** Its advantage over every other
   model is significant at p < 0.0001 on the headline metrics under both
   Holm and Bonferroni correction and at all α ∈ {0.01, 0.05, 0.10}; no
   comparison involving MEIRA-full is ever lost. This claim is safe to make
   unconditionally.
2. **The ablation ladder is reliable under Holm.** All variant-vs-variant
   differences survive Holm correction at α = 0.05; under the stricter
   Bonferroni only the no-decay/ColBERT boundary (R1) and a few CrossLing
   micro-gaps at α = 0.01 (R2) are affected.
3. **BM25 > TF-IDF is fragile and should be phrased as a tendency, not a
   claim.** It is significant under F1 (p < 0.0001 on AgentIR, p = 0.0003
   on CrossLingIR) and marginal under nDCG@10/MAP (Holm p =
   0.0068–0.0458), non-significant under MRR, and lost under Bonferroni
   wherever it passes at α = 0.05. Its direction never
   flips, but the evidence for separating the two classical baselines is the
   weakest in the leaderboard.
4. **MEIRA without temporal decay vs the best neural baseline is the one
   place to hedge.** MEIRA-no-decay > ColBERT-like is significant under
   Holm on both datasets but is lost under Bonferroni on four of its eight
   significant occurrences; claims that no-decay MEIRA "beats the best
   neural baseline" should cite the Holm-corrected (not raw) p-values and
   note the Bonferroni sensitivity.

*These conclusions were produced from the simulated harness; see the status
note at the top. Full per-pair data: `results/k10_s10/correction_comparison.md`
and `results/k10_s10/alpha_sweep.md` (machine-readable: `.json` twins;
figures: `corr1_correction_comparison.png`, `sweep1_alpha_counts.png`).*
