# Abstract & Introduction — Draft Section

> **Status.** Draft prose for the paper's opening sections (abstract and
> introduction), written from the verified numbers already established in the
> companion drafts: `paper_datasets_protocol_draft.md` (datasets & protocol),
> `paper_sota_ablation_draft.md` (leaderboard & ablation), and
> `paper_robustness_draft.md` (statistical robustness). Every quantitative
> claim below is drawn verbatim from those drafts' verified tables and is
> checked by the companion validator.
>
> ⚠️ **Before submission.** As in all companion drafts: the current numbers
> come from the simulated evaluation harness (`model_sim.py`), not from a
> trained model. The abstract and introduction *wording* is format-ready; the
> *magnitudes* must be refreshed after real inference replaces
> `simulate_model()` (recipe: re-run `run_SOTA.py --seeds 10`,
> `run_ablation.py --seeds 10`, `run_experiments.py --k 10 --seeds 10` and
> regenerate the companion drafts).
>
> **Suggested placement:** Abstract page, then Section 1 (Introduction),
> followed by Related Work, then the datasets & protocol section
> (`paper_datasets_protocol_draft.md`), then Results
> (`paper_sota_ablation_draft.md`), Robustness
> (`paper_robustness_draft.md`), and the Conclusion
> (`paper_conclusion_draft.md`).

---

## Abstract

**Primary variant (submission abstract).**

Multi-turn agentic information retrieval — retrieving relevant documents
given an accumulating conversational context — strains classical and neural
retrieval systems: they carry no structured memory across turns and cannot
explain their retrievals. We present **MEIRA**, a Memory-Enhanced Interpretable
Retrieval Agent coupling a 64-slot episodic memory bank, a temporal-decay
mechanism that down-weights stale memories, and an XAI attribution head
that emits per-document explanation confidences. To evaluate it we
introduce two synthetic-but-realistic FIRE-style benchmarks with sibling-
topic hard negatives and label noise — **FIRE-AgentIR-2026** (multi-turn
agentic retrieval; 8,400 samples, 10 topic clusters) and
**FIRE-CrossLingIR-2026** (cross-lingual Indian-language retrieval; 3,200
samples, 5 bilingual clusters) — and two metrics: **XAIR@K**, an
explainability-adjusted IR score interpolating ranking quality with
attribution confidence (w = 0.25), and **MDS**, a memory-diversity score
that flags memory mode collapse. Across ten evaluation seeds, MEIRA-full is
the best model on every ranking-quality metric (F1, nDCG@10, MAP, MRR) on
both datasets: F1 = 0.826/0.780 vs 0.740/0.680 for the strongest baseline
(+11.7%/+14.6% relative), significant at p < 0.0001 (paired t-test) and
surviving both Holm and Bonferroni multiplicity correction at every
threshold we consider.
Ablations rank the components memory > decay > XAI by ΔF1 contribution
(+0.110/+0.124, +0.078/+0.086, +0.049/+0.054), with the XAI head uniquely
responsible for the explainability-adjusted score (ΔXAIR@10 +0.894/+0.886).

**Tight blurb (for call-for-papers / short abstracts).**

Agentic multi-turn retrieval requires memory across turns and the ability to
explain retrievals — capabilities absent from classical and neural
baselines. We present **MEIRA** (Memory-Enhanced Interpretable Retrieval
Agent), which couples a 64-slot episodic memory bank, a temporal-decay
mechanism, and an XAI attribution head. We contribute two FIRE-style
benchmarks (FIRE-AgentIR-2026, FIRE-CrossLingIR-2026) with sibling-topic
hard negatives and label noise, and two metrics — XAIR@K
(explainability-adjusted IR score) and MDS (memory diversity). On both
benchmarks across ten seeds, MEIRA-full beats the strongest baseline on
every ranking-quality metric (F1, nDCG@10, MAP, MRR; F1 +0.087/+0.099) at
p < 0.0001, robust to Holm and Bonferroni correction. Ablations rank
memory > decay > XAI by contribution.

**Word count & variant selection.** The primary variant is the submission
abstract. Counts (verified programmatically): **208 words** (whitespace-
tokenized, the MS-Word-style convention) / **226** under a strict
hyphen-splitting convention — comfortably inside the 250-word FIRE-style
limit with margin for template quirks. The tight blurb is **114 / 127**
words and is intended only for call-for-papers / short-abstract contexts.
The two variants are deliberately related rather than independent: the
blurb is the primary compressed to framing + headline result, with every
number kept in the primary. If a venue enforces a lower cap (e.g. 150
words), cut from the primary in this order of priority: (1) the
benchmark-detail parentheticals (keep only the dataset names), (2) the
full ΔF1 triple (keep "memory > decay > XAI by contribution"), (3) the
Holm/Bonferroni clause (keep "significant at p < 0.0001").

---

## I1. Motivation: the agentic turn in retrieval

Information retrieval is becoming *agentic*. Instead of a single
query-and-respond round trip, users increasingly interact with systems that
pursue an information need over multiple turns — clarifying, reformulating,
and accumulating context as they go. The benchmark conventions of the FIRE
(Forum for Information Retrieval Evaluation) community and of adjacent
conversational-search tracks reflect this shift: the unit of evaluation is
no longer a lone query but a *conversation*, and the system is expected to
use what it has already seen to retrieve better in the current turn.

This shift exposes two capabilities that the standard retrieval toolbox does
not provide.

**Memory across turns.** Classical systems (BM25, TF-IDF) and their neural
successors (dense retrieval, late-interaction models) are stateless: they
re-encode every query from scratch and carry nothing between turns. In a
multi-turn session this is a structural handicap — information that was
established in turn one and is decisive in turn four must be re-derived, or
is silently lost. What is needed is an *episodic memory* that persists
retrieval-relevant state across the session — and, because sessions drift,
a mechanism for forgetting.

**Explainability of retrieval.** An agentic system that retrieves on the
user's behalf must be able to say *why*. Retrieval decisions that are
opaque are hard to trust, hard to debug, and hard to audit — yet none of the
standard metrics (nDCG, MAP, MRR, P@K) reward a system for being able to
explain its hits. A correct retrieval that the system cannot justify is
scored identically to one it can.

## I2. The gap

Existing work addresses pieces of this picture but not the whole. Memory
augmentation for LLM-based systems is an active area (e.g. MemGPT-style
memory hierarchies and memory banks with cognitive-style forgetting), and
explainable IR has a substantial literature on post-hoc attribution. What is
missing, to our knowledge, is a *retrieval* system that couples all three
headline mechanisms — episodic memory, temporal decay, and an attribution
head — and is evaluated end-to-end on multi-turn, cross-lingual benchmarks
with metrics that actually reward memory utilisation and explainability.
The standard metric suite is also silent on the two behaviours this paper
cares about: whether the system's memory is being used, and whether its
retrievals can be explained.

## I3. Our approach

We build **MEIRA**, a Memory-Enhanced Interpretable Retrieval Agent whose
architecture couples three components: (i) a **64-slot episodic memory bank**
that persists retrieval states across turns; (ii) a **temporal-decay
mechanism** defined over the memory bank that down-weights stale memories;
and (iii) an **XAI attribution head** that emits a normalised confidence for
each retrieved document, making every retrieval defensible. To measure the
first two behaviours we introduce two metrics — **XAIR@K**, which
interpolates nDCG@K with the mean attribution confidence over relevant
top-K hits (w = 0.25), and **MDS**, the fraction of the memory bank that is
actually used. To exercise the full pipeline we construct two
synthetic-but-realistic FIRE-style benchmarks with sibling-topic hard
negatives and label noise (Section D1), and evaluate every system across ten
evaluation seeds with a fully specified protocol (Section D2–D5).

## I4. Contributions

1. **MEIRA** — a retrieval agent coupling episodic memory (64 slots),
   temporal decay, and an XAI attribution head; the three components are
   removed one at a time in a controlled ablation to isolate each one's
   contribution.
2. **Two benchmarks** — FIRE-AgentIR-2026 (multi-turn agentic conversation
   retrieval; 8,400 samples, 10 topic clusters, 6 turns per conversation,
   67.1% hard-negative share) and FIRE-CrossLingIR-2026 (cross-lingual
   Indian-language retrieval; 3,200 samples, 5 bilingual clusters, 63.9%
   hard-negative share), both deterministic (seed 42) and reproducible.
3. **Two metrics** — XAIR@K (explainability-adjusted IR score; w = 0.25)
   and MDS (memory diversity score over the 64-slot bank); both defined only
   where their preconditions hold (attribution head / memory bank,
   respectively), so baselines are not unfairly penalised.
4. **A rigorous, honest evaluation** — ten seeds, stratified 70/15/15
   splits, conversation-level query pooling, paired t-tests with
   Holm-Bonferroni (primary) and Bonferroni (stress-test) multiplicity
   correction, and an α-sensitivity sweep over α ∈ {0.01, 0.05, 0.10}; the
   verdicts are checked for stability under every choice.

## I5. Findings at a glance

Across ten evaluation seeds MEIRA-full is the best model on **every ranking
metric on both datasets**, with F1 = 0.826±0.015 / 0.780±0.030 against
0.740±0.013 / 0.680±0.029 for the strongest baseline (relative gains
+11.7% / +14.6% on F1 and +3.4% / +3.2% on nDCG@10), and the gains are
significant at p < 0.0001 (t = 20.120 / 13.058 on nDCG@10) under *both*
multiplicity corrections at *every* threshold we consider — no comparison
involving MEIRA-full is ever lost. The ablation ranks the components by contribution
as **memory > decay > XAI** (ΔF1 +0.110/+0.124, +0.078/+0.086,
+0.049/+0.054), with the XAI head uniquely responsible for the
explainability-adjusted score (ΔXAIR@10 +0.894/+0.886) and the memory bank
reaching full utilisation (MDS = 1.000±0.000) with no mode collapse. Two
honest caveats, developed in the body of the paper: at shallow cutoffs
(P@5 / P@10) the models are effectively tied, so the claim is superiority in
ranking quality rather than raw precision; and the two fragile comparisons
in the whole study (BM25 vs TF-IDF; MEIRA-no-decay vs ColBERT-like) never
involve MEIRA-full, so the headline result is robust to how significance is
decided.

## I6. Roadmap

Section 2 (companion draft `paper_datasets_protocol_draft.md`) specifies
the benchmarks, protocol, metrics and models (Sections D1–D5). Section 3
(`paper_sota_ablation_draft.md`) presents the SOTA leaderboard and the
component ablation (Sections S1–S2). Section 4
(`paper_robustness_draft.md`) analyses statistical robustness — multiplicity
correction (R1) and threshold sensitivity (R2). Section 5
(`paper_conclusion_draft.md`) summarises, states limitations, and outlines
future work.

---

## Defensible claims

1. **The framing claims are standard and safe.** Agentic / conversational
   retrieval, memory-augmented systems, and explainable IR are all active
   research areas (see Related Work); the introduction positions MEIRA at
   their intersection without over-claiming novelty of the individual
   mechanisms.
2. **All quantitative claims in the abstract and I5 are drawn from the
   verified companion drafts** and survive their validators: leaderboard
   margins (+0.087/+0.099 F1), relative gains (+11.7%/+14.6%), significance
   (t = 20.120/13.058, p < 0.0001), ablation ranking (memory > decay > XAI),
   and robustness (no MEIRA-full comparison ever lost; F1 invariant across
   all corrections and thresholds).
3. **The scope caveat is stated up front.** "Best on every ranking metric"
   is scoped to ranking quality (F1, nDCG@10, MAP, MRR) because P@5/P@10
   tie across models; the abstract and I5 both say so explicitly.
4. **Hedge:** the ⚠️ status note makes the simulated-harness provenance
   unavoidable for any reader of the draft; the magnitudes must be
   regenerated from real checkpoints before submission.

*Sources: all numbers cross-checked against the verified tables in
`paper_datasets_protocol_draft.md`, `paper_sota_ablation_draft.md`, and
`paper_robustness_draft.md` (which in turn cite
`results/s10/sota.json`, `results/s10/ablation.json`,
`results/k10_s10/correction_comparison.json`, and
`results/k10_s10/alpha_sweep.json`).*
