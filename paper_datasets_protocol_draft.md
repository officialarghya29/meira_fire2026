# Datasets, Evaluation Protocol & Metrics — Draft Section

> **Status.** Draft prose for the paper's experimental-setup section (datasets,
> evaluation protocol, metrics, models, and statistical analysis), written from
> the verified code in `datasets_fire.py`, `ir_metrics.py`, `model_sim.py`,
> `run_experiments.py`, `run_SOTA.py`, and `run_ablation.py`. Dataset statistics
> were recomputed directly from the builders at the fixed construction seed
> (42) and match the numbers below exactly.
>
> ⚠️ **Before submission.** The models in this harness are *simulated*:
> `model_sim.py` draws relevance scores from calibrated distributions instead
> of running trained checkpoints, so the numbers in this section describe the
> evaluation *machinery* (datasets, splits, metrics, statistics), not model
> behaviour. Once real inference replaces `simulate_model()`, the protocol
> below is unchanged — only the score distributions change. Regenerate the
> companion result drafts from `results/s10/sota.json` and
> `results/s10/ablation.json` after re-running `run_SOTA.py --seeds 10` and
> `run_ablation.py --seeds 10`.
>
> **Suggested placement:** the Setup / Experimental Design section of the
> paper (immediately before the results), which is developed in the companion
> drafts `paper_sota_ablation_draft.md` (main results) and
> `paper_robustness_draft.md` (significance & robustness).

---

## D1. Benchmark datasets

We evaluate on two synthetic-but-realistic IR benchmarks built to mirror the
task conventions of FIRE / ACM SIGIR tracks: **FIRE-AgentIR-2026** (multi-turn
agentic conversation retrieval) and **FIRE-CrossLingIR-2026** (cross-lingual /
Indian-language retrieval). Both are generated programmatically with a fixed
seed (42) so the corpora are deterministic and reproducible, and both expose a
unified `IRSample` API (token ids, attention mask, label, conversation id,
turn, hard-negative flag) so that every downstream script — k-fold
cross-validation, multi-seed evaluation, ablation, and the SOTA comparison —
runs identically on either dataset.

### D1.1 FIRE-AgentIR-2026

FIRE-AgentIR-2026 simulates multi-turn agentic retrieval: each conversation
pursues a topic over six successive turns, and the system must retrieve the
relevant document given the accumulated conversational context. The corpus is
organised around **10 IR/NLP topic clusters** (episodic memory, dense
retrieval, RAG, neural reranking, explainability, evaluation metrics,
conversational search, query understanding, multimodal IR, and agentic
systems). Difficulty increases across turns: the generative process retains a
fraction `1 − min(0.07·turn, 0.35)` of a topic's core vocabulary (with a
floor of two terms), so later turns contain fewer distinctive terms and are
progressively harder to disambiguate (simulating topic drift).

Each positive query–document pair is accompanied by **three negatives**; with
probability 0.65 a negative is *hard* — drawn from a **sibling topic** that
shares the common IR vocabulary, so it is topically confusable with the
positive rather than an easy random distractor. Relevance labels are flipped
with probability 0.05 to simulate imperfect human judgements.

### D1.2 FIRE-CrossLingIR-2026

FIRE-CrossLingIR-2026 simulates bilingual retrieval over Indian-language
tracks, aligned with FIRE's historical emphasis on Hindi, Bengali and Tamil
IR. The corpus is organised around **5 bilingual topic clusters**
(Hindi-English health and agriculture; Bengali-English news and education;
Tamil-English technology). Queries and documents are generated in English, in
transliterated vernacular, or in **code-switched (mixed)** form, injecting
vocabulary shift and transliteration noise. As in AgentIR, negatives are drawn
from a sibling topic with shared vocabulary (hard-negative probability 0.60),
and labels are flipped with probability 0.06.

### D1.3 Corpus statistics

Table D1 reports the exact statistics of the two generated corpora
(construction seed 42), as produced by the dataset builders.

**Table D1. Dataset statistics.**

| Statistic | FIRE-AgentIR-2026 | FIRE-CrossLingIR-2026 |
|---|---|---|
| Topic clusters | 10 | 5 |
| Conversations | 350 | 200 |
| Turns per conversation | 6 | 4 |
| Total samples | 8,400 | 3,200 |
| Positives (ratio) | 2,302 (27.4%) | 920 (28.7%) |
| Negatives | 6,098 | 2,280 |
| Hard negatives (share of negatives) | 4,094 (67.1%) | 1,457 (63.9%) |
| Hard negatives per positive | 1.78 | 1.58 |
| Pos : neg ratio | 1 : 2.6 | 1 : 2.5 |
| Hard-negative probability | 0.65 | 0.60 |
| Label noise | 0.05 | 0.06 |

Both corpora are class-balanced enough for stable stratified splits while
retaining a realistic positive-to-negative skew, and the hard-negative
densities (≈1.6–1.8 hard negatives per positive) make the ranking task
non-trivial for lexical baselines.

### D1.4 Tokenisation & input representation

Text is tokenised with a fixed, deterministically-assigned vocabulary of
30,522 tokens (BERT-style CLS/SEP framing, `[CLS]` = 1, `[SEP]` = 2, padding
to a maximum length of 128), so input representations are stable across
datasets and seeds. Each sample concatenates the query and candidate document
(`query [SEP] document`).

---

## D2. Evaluation protocol

**Data splits.** Both datasets expose the same split utilities. For the
multi-seed experiments the corpus is stratified-split by class into
train / validation / test at **70 / 15 / 15**, with the split re-seeded per
evaluation seed. For the k-fold experiment, a stratified **k-fold
cross-validation** (k = 10) provides per-fold train/validation pairs.

**Evaluation seeds.** All multi-seed experiments use the seed range
42…51 (ten seeds). For each seed, the models are scored on the held-out
**test** partition of that seed's stratified split (≈15% of each class:
≈1,260 samples per seed for FIRE-AgentIR-2026 and ≈480 for
FIRE-CrossLingIR-2026); the k-fold experiment scores each of the 10 folds
once (seed 42 + fold index). Reported values are **mean ± standard
deviation across the ten evaluation seeds** (or across the ten folds).

**Query pooling.** Ranked metrics are computed at the conversation level: each
conversation is treated as one query whose candidate pool (one positive and
three negatives per turn) is ranked by the model's relevance scores. All
flat predictions are regrouped into per-query pools before nDCG, MAP, MRR,
R-Precision and P@K are computed, so the ranked metrics reflect real retrieval
lists rather than instance-level classification.

**Outputs.** Every script archives its results to a configuration-named
subfolder (`results/k10_s10/` for the k-fold + multi-seed robustness suite;
`results/s10/` for the SOTA and ablation runs) so different configurations
are stored side-by-side and never overwrite each other.

---

## D3. Metrics

We report the full standard suite **plus two novel metrics proposed in this
paper**; all are computed on both datasets in every experiment.

**Standard metrics.** From the classification suite we report F1, Precision,
Recall, Accuracy, ROC-AUC and Average Precision. From the ranked-list suite
we report nDCG@5, nDCG@10, MAP, MAP@10, MRR, R-Precision, P@5 and P@10.
The main leaderboard (Table 1–2 of `paper_sota_ablation_draft.md`) reports
the 13 headline metrics: F1, AUC, AP, nDCG@5, nDCG@10, MAP, MAP@10, MRR,
R-Prec, P@5, P@10, XAIR@10, MDS.

**XAIR@K — eXplainability-Adjusted IR score (proposed).** XAIR@K penalises a
system that retrieves the right documents but cannot explain *why*:

```
XAIR@K  =  (1 − w) · nDCG@K  +  w · mean( xai_conf(d)  for d in top-K, d relevant )
```

where `w = 0.25` and `xai_conf(d) ∈ [0,1]` is the normalised XAI attribution
confidence of document *d*. The metric interpolates ranking quality with
explainability: a correct-but-unexplainable retrieval scores below a
correct-and-explainable one, rewarding interpretable retrieval. XAIR@K is
defined only for models with an XAI component (the MEIRA variants); baseline
systems without attribution heads are marked “—” (they do not receive an
unfair penalty).

**MDS — Memory Diversity Score (proposed).** MDS measures utilisation of the
episodic memory bank:

```
MDS  =  | unique memory slots accessed across all queries | / S ,   S = 64
```

MDS ranges over [0, 1]; low values indicate memory under-utilisation
(mode collapse) and a healthy bank is expected to exceed 0.3. MDS is defined
only for models with an episodic memory component.

**Decision threshold.** For models whose output is a relevance probability,
the operating threshold is chosen per run as the point on the precision-recall
curve that maximises F1 on that run's sample, and classification metrics (F1,
Precision, Recall, Accuracy) are computed at that threshold.

---

## D4. Models

Eight systems are evaluated: four baselines spanning the classical and neural
paradigms, and four MEIRA configurations (the proposed model plus three
ablations).

**Table D2. Model registry.**

| Model | Memory | XAI | Decay | Role |
|---|---|---|---|---|
| BM25 | – | – | – | classical lexical baseline |
| TF-IDF | – | – | – | classical lexical baseline |
| Dense-IR | – | – | – | dense neural baseline |
| ColBERT-like | – | – | – | late-interaction neural baseline |
| MEIRA-no-memory | ✗ | ✓ | ✗ | ablation: memory removed |
| MEIRA-no-decay | ✓ | ✓ | ✗ | ablation: temporal decay removed |
| MEIRA-no-xai | ✓ | ✗ | ✓ | ablation: XAI attribution head removed |
| **MEIRA-full (ours)** | ✓ | ✓ | ✓ | proposed model (memory + XAI + decay) |

MEIRA-full couples an episodic memory bank (64 slots) with a temporal-decay
mechanism over memory and an XAI attribution head; the three ablation variants
remove exactly one of the three headline components so their contribution can
be isolated (Section S2 of `paper_sota_ablation_draft.md`). Note that
temporal decay is defined *over* the memory bank, so the no-memory variant
disables decay as well (both flags off in the registry); consequently the
memory-ablation delta in the companion results absorbs decay's contribution
as well — the component attributions are therefore conservative, not
orthogonal. Memory-equipped models also
report which memory slots they access per retrieval, which feeds MDS.

> ⚠️ **Simulation note (honesty requirement).** In the current harness every
> model's relevance scores are drawn from per-model calibrated score
> distributions (`model_sim.py::MODEL_REGISTRY`), so no model is trained and
> no weights are fit. The purpose of the harness is to validate the full
> evaluation pipeline — metrics, splitting, statistics, figures — without a
> GPU. For the submission, replace `simulate_model()` with real forward
> passes from the trained checkpoints; the protocol above and every downstream
> script are unchanged.

---

## D5. Statistical analysis

All significance claims are computed from the ten-seed evaluation data with
**paired two-sided t-tests** (across the ten seeds; df = 9) over every pair of
the eight models — 28 pairwise comparisons per dataset × metric. Because the
t-tests are not independent, p-values are reported **Holm-Bonferroni
corrected** (primary), with **Bonferroni** (×28) as the conservative
stress-test and an **α-sensitivity sweep** over α ∈ {0.01, 0.05, 0.10} to show
that conclusions do not hinge on the threshold. Full details, tables, and the
defensible-claims list are in the companion draft `paper_robustness_draft.md`;
the pairwise matrices and α-sweep data are archived in
`results/k10_s10/significance_matrix_*_{holm,bonferroni}.json/.md` and
`results/k10_s10/alpha_sweep.json/.md`.

---

## Defensible claims

1. **The two benchmarks exercise the intended difficulty regimes.** Both
   corpora have ≈1.6–1.8 hard negatives per positive drawn from sibling
   topics, making the ranking task non-trivial for lexical baselines (BM25,
   TF-IDF lag the neural systems in the leaderboard).
2. **The evaluation protocol is fully specified and reproducible.**
   Deterministic corpora (seed 42), ten evaluation seeds (42–51), stratified
   70/15/15 splits, conversation-level query pooling, and per-run
   F1-optimal thresholds are all defined above; any re-implementation should
   reproduce the reported numbers exactly.
3. **Two novel metrics are well-defined and interpretable.** XAIR@K (w =
   0.25 interpolation of nDCG@K with mean XAI confidence over relevant
   top-K hits) and MDS (fraction of the 64-slot memory bank used) are the
   paper's measurement contributions, both defined only where their
   preconditions hold (XAI component / memory component respectively).
4. **Hedge:** the models are currently simulated; every claim in this section
   is a claim about the *harness*, and results must be regenerated from real
   checkpoints before submission (see status note above).

*Sources: `datasets_fire.py` (builders, splits, statistics), `ir_metrics.py`
(formulas, threshold rule), `model_sim.py` (registry, simulation),
`run_experiments.py` / `run_SOTA.py` / `run_ablation.py` (seed ranges,
splits, pooling, aggregation). Dataset statistics recomputed at seed 42 by the
builders; seed and split constants taken directly from the run scripts.*
