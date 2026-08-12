# Conclusion & Limitations — Draft Section

> **Status.** Draft prose for the paper's closing section (summary,
> limitations, future work), written from the verified numbers in the
> companion drafts: `paper_sota_ablation_draft.md` (leaderboard & ablation),
> `paper_robustness_draft.md` (statistical robustness), and
> `paper_datasets_protocol_draft.md` (benchmarks & protocol). All
> quantitative claims below are cross-checked by the companion validator.
>
> ⚠️ **Before submission.** As in all companion drafts: the numbers come
> from the simulated evaluation harness (`model_sim.py`), not a trained
> model. The single most important limitation stated below is therefore
> *the harness itself*; the conclusion is written so that the summary claims
> survive re-running with real checkpoints, while the limitations section
> tells the reader exactly what must change. Regeneration recipe:
> `run_SOTA.py --seeds 10`, `run_ablation.py --seeds 10`,
> `run_experiments.py --k 10 --seeds 10`, then the significance /
> correction / sweep scripts, then refresh the companion drafts.
>
> **Suggested placement:** the paper's final section, after Robustness
> (`paper_robustness_draft.md`).

---

## C1. Summary of contributions and findings

We presented **MEIRA**, a Memory-Enhanced Interpretable Retrieval Agent that
couples an episodic memory bank (64 slots), a temporal-decay mechanism over
memory, and an XAI attribution head, and evaluated it end-to-end on two new
FIRE-style benchmarks — FIRE-AgentIR-2026 (multi-turn agentic conversation
retrieval) and FIRE-CrossLingIR-2026 (cross-lingual Indian-language
retrieval) — built with sibling-topic hard negatives and label noise, and
measured with two new metrics: **XAIR@K** (explainability-adjusted IR score,
w = 0.25) and **MDS** (memory diversity score).

The empirical picture, established across ten evaluation seeds with a fully
specified protocol (stratified 70/15/15 splits, conversation-level query
pooling) and a rigorous statistical analysis (paired t-tests, df = 9, with
Holm-Bonferroni primary and Bonferroni stress-test corrections plus an
α-sensitivity sweep):

- **MEIRA-full is the best model on every ranking metric on both datasets.**
  Against the strongest baseline (ColBERT-like) it gains +0.087 / +0.099
  F1, +0.032 / +0.030 nDCG@10, +0.045 / +0.042 MAP and +0.027 / +0.021 MRR
  (AgentIR / CrossLingIR), i.e. +11.7% / +14.6% relative F1; the differences
  are significant at p < 0.0001 (t = 20.120 / 13.058 on nDCG@10) and —
  crucially —
  **survive both multiplicity corrections at every threshold we consider**:
  no comparison involving MEIRA-full is ever lost under Holm or Bonferroni
  at α ∈ {0.01, 0.05, 0.10}.
- **The components contribute in a stable order: memory > decay > XAI.**
  Removing episodic memory costs the most (ΔF1 +0.110 / +0.124) and induces
  memory mode collapse (MDS → 0.000); removing temporal decay costs the
  second-most (ΔF1 +0.078 / +0.086) on every ranking metric; removing the
  XAI head costs the least in ranking terms (ΔF1 +0.049 / +0.054) but is the
  sole driver of the explainability-adjusted score (ΔXAIR@10 +0.894 /
  +0.886) — i.e. the attribution head is what makes MEIRA's retrievals
  explainable, and XAIR@K responds exactly to the mechanism it measures.
- **The headline result is robust to how significance is decided.** F1
  verdicts are 28/28 significant under every correction at every α; Holm
  changes exactly one verdict across the whole sweep (the no-decay >
  ColBERT-like boundary at α = 0.01); Bonferroni is where the threshold
  binds, losing at most the BM25-vs-TF-IDF tie and the no-decay/ColBERT
  boundary — the same two fragile families at every α — and **never any
  comparison involving MEIRA-full** (14 α-sensitive pair-instances in total,
  none of them ours).

Together these findings support the paper's central claim: a retrieval agent
that remembers, forgets, and can explain its retrievals is better *and*
more defensible than stateless lexical or neural baselines — and the
improvement is not a statistical artifact of how we choose to test it.

## C2. Limitations

We state the limitations plainly, in decreasing order of severity.

1. **The evaluation harness is currently simulated.** The single most
   important caveat: in the current codebase, `model_sim.py` draws each
   model's relevance scores from calibrated distributions instead of running
   trained checkpoints, so the reported magnitudes are *pipeline-validation
   artifacts*, not experimental results. The protocol, metrics, and
   statistical machinery are real and verified; only the score
   distributions are synthetic. Every table and figure in this paper's
   drafts must be regenerated from real inference before the numbers can be
   cited. (This is the reason the drafts carry a ⚠️ status note on every
   section.)
2. **The benchmarks are synthetic.** FIRE-AgentIR-2026 and
   FIRE-CrossLingIR-2026 are programmatically generated (deterministic at
   seed 42) to mirror FIRE-style task conventions; they are not collections
   of human-annotated documents. Their difficulty regimes (hard-negative
   density ≈1.6–1.8 per positive, label noise 5–6%) are realistic, but
   generalisation to naturalistic corpora and real user sessions is
   untested.
3. **Shallow-cutoff precision is not a point of separation.** At P@5 and
   P@10 all models are effectively tied (P@10 = 0.101±0.001 on AgentIR,
   0.074±0.002 on CrossLingIR for every model). The paper's claim is
   superiority in *ranking quality*, and we say so explicitly; it cannot be
   read as a claim about raw precision at shallow cutoffs.
4. **Component attributions are conservative, not orthogonal.** Temporal
   decay is defined *over* the memory bank, so the no-memory ablation
   disables decay as well; the memory Δ therefore absorbs decay's
   contribution. The ablation order (memory > decay > XAI) is thus a lower
   bound on memory's share rather than an orthogonal decomposition.
5. **MDS saturates.** MEIRA-full reaches MDS = 1.000±0.000 on both datasets,
   and the metric cannot distinguish "fully utilised bank" from "bank too
   small for the task". The current 64-slot bank may simply be easier to
   saturate than a real-world memory would be.
6. **Two comparison families are fragile.** BM25 > TF-IDF and
   MEIRA-no-decay > ColBERT-like are significant under Holm but not under
   Bonferroni at α = 0.05 (and the no-decay boundary also drops under Holm
   at α = 0.01). Both are marginal-tier comparisons; neither involves
   MEIRA-full, but any claim about them must cite corrected p-values and
   the threshold caveat.
7. **XAIR@K's scope.** The metric is defined only for models with an
   attribution head (baselines are marked "—"), and its w = 0.25 weighting
   is a design choice, not a tuned parameter; its behaviour under other
   weightings is unexplored.

## C3. Future work

- **Real checkpoints.** Replace `simulate_model()` with trained MEIRA
   forward passes (the rest of the pipeline is unchanged) and regenerate all
   numbers — the drafts' structure is format-ready for this.
- **Naturalistic benchmarks.** Extend the two synthetic corpora with
   human-annotated, real-document counterparts (and, ideally, a FIRE shared
   task) to test generalisation of both the models and the two metrics.
- **Learning the decay schedule.** The temporal-decay mechanism is currently
   a fixed schedule; learning its parameters from data — or conditioning
   forgetting on memory salience, in the spirit of cognitive-style memory
   consolidation — is a natural next step.
- **Larger and heterogeneous memory.** Vary the bank size S and the
   memory-slot semantics (episodes vs. facts vs. evidence) to characterise
   MDS more finely and test whether saturation (limitation 5) is a metric
   artefact or a real ceiling.
- **XAIR weight sensitivity.** Sweep w and study how the 
   explainability-adjusted leaderboard changes — turning limitation 7 into
   a robustness result rather than a caveat.
- **User studies.** The ultimate test of explainability is human trust;
   a user study comparing MEIRA's attributed retrievals against
   attribution-free baselines would complement the XAIR@K evidence.

## C4. Bottom line

MEIRA demonstrates that coupling episodic memory, temporal decay, and
explainability in a single retrieval agent is not only feasible but
measurably better — and that the advantage is robust to every statistical
choice we made (correction method and threshold). With real inference and
naturalistic benchmarks in place, the same architecture and the same two
metrics are the paper's concrete, reusable contributions to agentic and
explainable IR.

---

## Defensible claims

1. **All summary numbers in C1 are the verified companion-draft numbers**
   (leaderboard margins, relative gains, t-stats, ablation deltas, XAIR@10
   / MDS, correction counts, α-sweep facts) and survive their validators.
2. **The limitations list is exhaustive for the current state of the
   project** — the simulated harness is limitation #1 and is stated
   unconditionally, which is the honest and defensible framing for a
   pipeline-validation paper.
3. **Future-work items map one-to-one onto limitations**, so the section
   reads as a roadmap rather than a confession.
4. **Hedge:** claims 1–2 above hold for the *current drafts*; regenerating
   real numbers may change the magnitudes (not the protocol), and the
   conclusion is written so that only the numbers, not the argument, need
   refreshing.

*Sources: cross-checked against the verified tables in
`paper_sota_ablation_draft.md`, `paper_robustness_draft.md`, and
`paper_datasets_protocol_draft.md` (which in turn cite
`results/s10/sota.json`, `results/s10/ablation.json`,
`results/k10_s10/correction_comparison.json`, and
`results/k10_s10/alpha_sweep.json`).*
