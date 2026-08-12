"""Camera-ready prose trims for the LaTeX pipeline.

Exact (old, new) string pairs, applied by md2tex.py to single-space-joined
markdown paragraphs before conversion. The trims remove redundant prose that
duplicates what the tables/other sections already state, so the paper fits
the FIRE 2026 9-page content limit with a comfortable margin. Every headline
number, citation, and cross-reference is preserved verbatim.

Entries are (name, old, new). "old" must match the joined paragraph text
exactly; unmatched entries are reported as MISSED by md2tex.py.
"""

TRIMS = [
    # ---- 1.1 Motivation ---------------------------------------------------
    ("1.1 explainability",
     "**Explainability of retrieval.** An agentic system that retrieves on the "
     "user's behalf must be able to say *why*. Retrieval decisions that are "
     "opaque are hard to trust, hard to debug, and hard to audit — yet none "
     "of the standard metrics (nDCG, MAP, MRR, P@K) reward a system for being "
     "able to explain its hits. A correct retrieval that the system cannot "
     "justify is scored identically to one it can.",
     "**Explainability of retrieval.** An agentic system that retrieves on the "
     "user's behalf must be able to say *why*: retrieval decisions that are "
     "opaque are hard to trust, debug, and audit, yet none of the standard "
     "metrics (nDCG, MAP, MRR, P@K) reward a system for being able to explain "
     "its hits."),

    ("1.1 memory",
     "**Memory across turns.** Classical systems (BM25, TF-IDF) and their "
     "neural successors (dense retrieval, late-interaction models) are "
     "stateless: they re-encode every query from scratch and carry nothing "
     "between turns. In a multi-turn session this is a structural handicap — "
     "information that was established in turn one and is decisive in turn "
     "four must be re-derived, or is silently lost. What is needed is an "
     "*episodic memory* that persists retrieval-relevant state across the "
     "session — and, because sessions drift, a mechanism for forgetting.",
     "**Memory across turns.** Classical systems (BM25, TF-IDF) and their "
     "neural successors are stateless: they re-encode every query from "
     "scratch and carry nothing between turns — a structural handicap in "
     "multi-turn sessions, where information established in turn one and "
     "decisive in turn four must be re-derived or is silently lost. What is "
     "needed is an *episodic memory* that persists retrieval-relevant state "
     "across the session, plus a mechanism for forgetting as sessions drift."),

    # ---- 1.2 The gap ------------------------------------------------------
    ("1.2 gap",
     "Existing work addresses pieces of this picture but not the whole. "
     "Memory augmentation for LLM-based systems is an active area (e.g. "
     "MemGPT-style memory hierarchies and memory banks with cognitive-style "
     "forgetting), and explainable IR has a substantial literature on "
     "post-hoc attribution. What is missing, to our knowledge, is a "
     "*retrieval* system that couples all three headline mechanisms — "
     "episodic memory, temporal decay, and an attribution head — and is "
     "evaluated end-to-end on multi-turn, cross-lingual benchmarks with "
     "metrics that actually reward memory utilisation and explainability. "
     "The standard metric suite is also silent on the two behaviours this "
     "paper cares about: whether the system's memory is being used, and "
     "whether its retrievals can be explained.",
     "Existing work addresses pieces of this picture but not the whole. "
     "Memory augmentation for LLM-based systems is an active area (e.g. "
     "MemGPT-style memory hierarchies and memory banks with cognitive-style "
     "forgetting), and explainable IR has a substantial literature on "
     "post-hoc attribution. What is missing, to our knowledge, is a "
     "*retrieval* system that couples all three headline mechanisms — "
     "episodic memory, temporal decay, and an attribution head — evaluated "
     "end-to-end on multi-turn, cross-lingual benchmarks with metrics that "
     "reward memory utilisation and explainability."),

    # ---- 1.5 Findings at a glance -----------------------------------------
    ("1.5 caveats",
     "Two honest caveats, developed in the body of the paper: at shallow "
     "cutoffs (P@5 / P@10) the models are effectively tied, so the claim is "
     "superiority in ranking quality rather than raw precision; and the two "
     "fragile comparisons in the whole study (BM25 vs TF-IDF; MEIRA-no-decay "
     "vs ColBERT-like) never involve MEIRA-full, so the headline result is "
     "robust to how significance is decided.",
     "Two caveats, developed below: at shallow cutoffs (P@5 / P@10) the "
     "models are effectively tied, so the claim is superiority in ranking "
     "quality rather than raw precision; and the only fragile comparisons "
     "(BM25 vs TF-IDF; MEIRA-no-decay vs ColBERT-like) never involve "
     "MEIRA-full."),

    # ---- 2.2 Agentic and conversational IR --------------------------------
    ("2.2 eval side",
     "On the evaluation side, the conversational-assistance track of TREC "
     "established multi-turn, context-dependent retrieval as a standard task "
     "(Dalton, Xiong & Callan, 2020) ✓, and the FIRE (Forum for Information "
     "Retrieval Evaluation) community has run a sustained programme of "
     "shared tasks with a strong Indian-language and code-mixed emphasis, "
     "including cross-lingual and conversational tracks whose conventions "
     "our benchmarks mirror (FIRE proceedings, CEUR-WS) ✓.",
     "On the evaluation side, TREC's conversational-assistance track "
     "established multi-turn, context-dependent retrieval as a standard task "
     "(Dalton, Xiong & Callan, 2020) ✓, and FIRE has run sustained shared "
     "tasks with a strong Indian-language and code-mixed emphasis whose "
     "conventions our benchmarks mirror (FIRE proceedings, CEUR-WS) ✓."),

    ("2.2 systems side",
     "MEIRA belongs to this agentic paradigm: each conversation is a session "
     "over which the system accumulates memory and refines its retrievals. "
     "Where we differ from the agentic-IR position papers is in the concrete "
     "mechanisms — an explicit episodic memory bank, a decay schedule, and "
     "an attribution head — and in the measurement of those mechanisms "
     "(XAIR@K, MDS).",
     "MEIRA belongs to this paradigm: each conversation is a session over "
     "which the system accumulates memory and refines its retrievals; we "
     "differ from the position papers in the concrete mechanisms — an "
     "episodic memory bank, a decay schedule, and an attribution head — and "
     "in measuring them (XAIR@K, MDS)."),

    # ---- 2.5 Hard negatives -----------------------------------------------
    ("2.5 strands",
     "Two further strands of related work inform the evaluation design. "
     "First, hard-negative construction: the sibling-topic negatives in our "
     "benchmarks follow the established finding that hard negatives — "
     "topically confusable distractors — are what make ranking tasks "
     "non-trivial and drive meaningful separation between lexical and neural "
     "systems (contrastive-training literature, incl. Karpukhin et al., 2020 "
     "✓, and surveys of negative-sampling techniques (Wischounig et al., "
     "2026) ✓). Our corpus statistics confirm the design works as intended: "
     "67.1% / 63.9% of negatives are hard, and the lexical baselines "
     "separate cleanly from the neural ones in the leaderboard. Second, "
     "evaluation methodology: LLM-as-judge approaches propose LLMs as "
     "relevance judges and meta-evaluate their agreement with human "
     "judgements (Li et al., 2024) ✓, and answerability-aware metrics argue "
     "that graded relevance alone under-describes retrieval quality (Farzi "
     "& Dietz, 2024) ✓.",
     "Two further strands inform the evaluation design. First, "
     "hard-negative construction: sibling-topic negatives follow the "
     "established finding that hard negatives — topically confusable "
     "distractors — make ranking tasks non-trivial and drive separation "
     "between lexical and neural systems (Karpukhin et al., 2020 ✓; "
     "Wischounig et al., 2026 ✓); our corpus statistics confirm this (67.1% "
     "/ 63.9% of negatives are hard). Second, evaluation methodology: "
     "LLM-as-judge approaches meta-evaluate judges' agreement with humans "
     "(Li et al., 2024) ✓, and answerability-aware metrics argue that graded "
     "relevance alone under-describes retrieval quality (Farzi & Dietz, "
     "2024) ✓."),

    # ---- 3.1 Benchmark datasets -------------------------------------------
    ("3.1 IRSample",
     "Both are generated programmatically with a fixed seed (42) so the "
     "corpora are deterministic and reproducible, and both expose a unified "
     "`IRSample` API (token ids, attention mask, label, conversation id, "
     "turn, hard-negative flag) so that every downstream script — k-fold "
     "cross-validation, multi-seed evaluation, ablation, and the SOTA "
     "comparison — runs identically on either dataset.",
     "Both are generated programmatically with a fixed seed (42) and expose "
     "a unified `IRSample` API (token ids, attention mask, label, "
     "conversation id, turn, hard-negative flag), so every downstream script "
     "— k-fold cross-validation, multi-seed evaluation, ablation, and the "
     "SOTA comparison — runs identically on either dataset."),

    # ---- 4.1 SOTA leaderboard ---------------------------------------------
    ("4.1 cross-ling margins",
     "The margins are consistent on FIRE-CrossLingIR-2026: F1 = 0.780±0.030 "
     "vs 0.680±0.029 (+0.099), nDCG@10 = 0.962±0.008 vs 0.932±0.012 "
     "(+0.030), MAP = 0.947±0.011 vs 0.905±0.016 (+0.042) and MRR = "
     "0.529±0.012 vs 0.509±0.015 (+0.021). In relative terms the ranking "
     "gains are large: +11.7% F1 and +3.4% nDCG@10 over the best baseline on "
     "AgentIR, +14.6% F1 and +3.2% nDCG@10 on CrossLingIR.",
     "The margins are consistent on FIRE-CrossLingIR-2026 (F1 = 0.780±0.030 "
     "vs 0.680±0.029, +0.099; nDCG@10 +0.030; MAP +0.042; MRR +0.021). In "
     "relative terms the gains are large: +11.7% / +14.6% F1 and +3.4% / "
     "+3.2% nDCG@10 (AgentIR / CrossLingIR)."),

    ("4.1 caveats",
     "Two caveats keep the headline honest: (i) on the shallow-cutoff "
     "precision metrics the models are effectively tied — P@10 = 0.101±0.001 "
     "for every model on AgentIR (P@5 spans only 0.194–0.201), and on "
     "CrossLingIR P@10 = 0.074±0.002 for every model while P@5 stays within "
     "0.146–0.148 — so the claim is superiority on ranking quality (F1, "
     "nDCG@10, MAP, MRR), not on raw precision at shallow cutoffs; and (ii) "
     "the two novel metrics **XAIR@10 and MDS are defined only for MEIRA "
     "variants** (the baselines produce no explanation attributions and "
     "maintain no episodic memory bank), so those columns are \"—\" for "
     "baselines by construction.",
     "Two caveats keep the headline honest: (i) at shallow cutoffs the "
     "models are effectively tied (P@10 = 0.101±0.001 on AgentIR, "
     "0.074±0.002 on CrossLingIR for every model), so the claim is "
     "superiority on ranking quality, not raw precision; and (ii) **XAIR@10 "
     "and MDS are defined only for MEIRA variants** — baselines produce no "
     "attributions and maintain no memory bank — so those columns are "
     "\"—\" by construction."),

    # ---- 4.2 Component ablation -------------------------------------------
    ("4.2 memory",
     "**Episodic memory is the dominant component.** Removing it costs the "
     "most on the headline metrics: ΔF1 = +0.110 (AgentIR) / +0.124 "
     "(CrossLingIR), ΔAUC = +0.064 / +0.089, ΔnDCG@10 = +0.038 / +0.040, "
     "ΔMAP = +0.055 / +0.056 and ΔMRR = +0.032 / +0.028. Its ablation also "
     "collapses the memory-diversity score from 1.000 to 0.000 (ΔMDS = "
     "+1.000) on both datasets — a mode-collapse signature: without the "
     "episodic bank, MEIRA no longer spreads its retrievals across memory "
     "states. Notably, the memory cost is largest on F1 and AUC (ΔF1 0.110, "
     "ΔAUC 0.064 on AgentIR), while CrossLingIR shows the same pattern with "
     "an even larger F1 cost (0.124).",
     "**Episodic memory is the dominant component.** Removing it costs the "
     "most on every ranking metric (ΔF1 = +0.110 / +0.124 on AgentIR / "
     "CrossLingIR; per-metric deltas in Table 7) and collapses the "
     "memory-diversity score from 1.000 to 0.000 (ΔMDS = +1.000) on both "
     "datasets — a mode-collapse signature: without the episodic bank, "
     "MEIRA no longer spreads its retrievals across memory states."),

    ("4.2 xai",
     "**The XAI attribution head is what delivers XAIR@10.** Removing it "
     "reduces XAIR@10 from 0.894 to 0.000 on AgentIR and 0.886 to 0.000 on "
     "CrossLingIR (ΔXAIR@10 = +0.894 / +0.886) — by construction the "
     "explainability-adjusted score vanishes when there is no attribution "
     "signal — while its effect on retrieval quality is comparatively modest "
     "(ΔF1 = +0.049 / +0.054, ΔnDCG@10 = +0.018 / +0.017, ΔMAP = +0.025 / "
     "+0.025, ΔMRR = +0.015 / +0.012). XAI is therefore best described as "
     "the component that *makes the model's retrievals explainable* (and "
     "hence paper-defensible under the XAIR metric) rather than the one that "
     "drives ranking.",
     "**The XAI attribution head is what delivers XAIR@10.** Removing it "
     "reduces XAIR@10 from 0.894 / 0.886 to 0.000 (ΔXAIR@10 = +0.894 / "
     "+0.886), while its effect on retrieval quality is comparatively modest "
     "(ΔF1 = +0.049 / +0.054). XAI is therefore best described as the "
     "component that *makes the model's retrievals explainable* rather than "
     "the one that drives ranking."),

    ("4.2 decay",
     "**Temporal decay contributes the second-largest, consistent gain.** "
     "Removing it costs ΔF1 = +0.078 / +0.086, ΔAUC = +0.043 / +0.058, "
     "ΔnDCG@10 = +0.028 / +0.025, ΔMAP = +0.040 / +0.036 and ΔMRR = +0.024 / "
     "+0.018. Removing *any* of the three components degrades every ranking "
     "metric on both datasets (all deltas are positive); decay sits between "
     "them, with the second-largest margins after memory and before XAI. On "
     "each of F1, AUC, nDCG@10, MAP and MRR the component ranking is the "
     "same: memory > decay > XAI.",
     "**Temporal decay contributes the second-largest, consistent gain** "
     "(ΔF1 = +0.078 / +0.086). Removing *any* of the three components "
     "degrades every ranking metric on both datasets (all deltas positive); "
     "on F1, AUC, nDCG@10, MAP and MRR the component ranking is the same: "
     "memory > decay > XAI."),

    ("4.2 ordering",
     "Ordering the components by average ΔF1 gives memory (0.110 / 0.124) > "
     "decay (0.078 / 0.086) > XAI (0.049 / 0.054), and this ranking is "
     "stable across the two datasets and across F1, AUC, nDCG@10 and MAP "
     "(the ΔMRR ordering is the same; the ΔXAIR@10 story is dominated by XAI "
     "by construction). One honest hedge, carried over from Section 5: the "
     "strongest ablated variant (MEIRA-no-decay) sits close to the best "
     "neural baseline (ColBERT-like), and that boundary is one of the few "
     "comparisons in the whole study that is significant under Holm but not "
     "under Bonferroni (the other fragile family being BM25 vs TF-IDF) — so "
     "claims that \"MEIRA without temporal decay still beats the best neural "
     "baseline\" should cite the Holm-corrected p-values and note the "
     "sensitivity.",
     "This ranking is stable across the two datasets and across F1, AUC, "
     "nDCG@10 and MAP. One honest hedge, carried over from Section 5: the "
     "strongest ablated variant (MEIRA-no-decay) sits close to the best "
     "neural baseline (ColBERT-like), and that boundary — like BM25 vs "
     "TF-IDF — is significant under Holm but not under Bonferroni, so claims "
     "that \"MEIRA without temporal decay still beats the best neural "
     "baseline\" should cite the Holm-corrected p-values and note the "
     "sensitivity."),

    # ---- 5.1 Holm vs Bonferroni -------------------------------------------
    ("5.1 holm reason",
     "The reason is that Holm's step-down procedure applies its smallest "
     "multipliers (1–2) to the largest raw p-values, and the marginal "
     "comparisons — those whose raw p approaches α = 0.05 (raw p ≈ "
     "0.002–0.046) — still land below the threshold after that adjustment "
     "(e.g. MRR on FIRE-AgentIR-2026: p = 0.0236 → 0.0472). All remaining "
     "comparisons have raw p ≤ 0.001 and stay far below α under any "
     "correction. Under the harsher Bonferroni correction, however, those "
     "same marginal pairs drop out.",
     "Holm applies its smallest multipliers (1–2) to the largest raw "
     "p-values, so the marginal comparisons (raw p ≈ 0.002–0.046) still land "
     "below α (e.g. MRR on AgentIR: p = 0.0236 → 0.0472), while all "
     "remaining comparisons have raw p ≤ 0.001. Under the harsher Bonferroni "
     "correction, however, those same marginal pairs drop out."),

    # ---- 5.2 Threshold sensitivity ----------------------------------------
    ("5.2 bonferroni",
     "Bonferroni is where the threshold binds. Under the strictest threshold "
     "α = 0.01, Bonferroni drops more comparisons (e.g. MAP on "
     "FIRE-CrossLingIR-2026: 24/28, with three lost pair-instances, and the "
     "CrossLing micro-gaps ColBERT-like > MEIRA-no-memory and "
     "MEIRA-no-xai > MEIRA-no-decay joining the two fragile families), while "
     "at the loosest threshold α = 0.10 it *recovers* exactly two of the "
     "eight α = 0.05 losses — MRR no-decay > ColBERT-like on "
     "FIRE-CrossLingIR-2026 (Bonferroni p = 0.0635) and MAP BM25 > TF-IDF on "
     "FIRE-AgentIR-2026 (Bonferroni p = 0.0946). In total, 14 pair-instances "
     "across the sweep are **α-sensitive** — their verdict changes with the "
     "threshold or they are lost at some α — and, again, **none of them "
     "involves MEIRA-full**.",
     "Bonferroni is where the threshold binds: at α = 0.01 it drops more "
     "comparisons (e.g. MAP on FIRE-CrossLingIR-2026: 24/28, plus the "
     "CrossLing micro-gaps ColBERT-like > MEIRA-no-memory and "
     "MEIRA-no-xai > MEIRA-no-decay joining the fragile families), while at "
     "α = 0.10 it *recovers* exactly two of the eight α = 0.05 losses (MRR "
     "no-decay > ColBERT-like on CrossLingIR, p = 0.0635; MAP BM25 > TF-IDF "
     "on AgentIR, p = 0.0946). In total, 14 pair-instances across the sweep "
     "are **α-sensitive**, and **none of them involves MEIRA-full**."),

    # ---- 6.1 Summary -------------------------------------------------------
    ("6.1 bullet 1",
     "- **MEIRA-full is the best model on every ranking metric on both "
     "datasets.** Against the strongest baseline (ColBERT-like) it gains "
     "+0.087 / +0.099 F1, +0.032 / +0.030 nDCG@10, +0.045 / +0.042 MAP and "
     "+0.027 / +0.021 MRR (AgentIR / CrossLingIR), i.e. +11.7% / +14.6% "
     "relative F1; the differences are significant at p < 0.0001 (t = "
     "20.120 / 13.058 on nDCG@10) and — crucially — **survive both "
     "multiplicity corrections at every threshold we consider**: no "
     "comparison involving MEIRA-full is ever lost under Holm or Bonferroni "
     "at α ∈ {0.01, 0.05, 0.10}.",
     "- **MEIRA-full is the best model on every ranking metric on both "
     "datasets.** Against the strongest baseline (ColBERT-like) it gains "
     "+0.087 / +0.099 F1 (+11.7% / +14.6% relative); the differences are "
     "significant at p < 0.0001 (t = 20.120 / 13.058 on nDCG@10) and "
     "**survive both multiplicity corrections at every threshold** (α ∈ "
     "{0.01, 0.05, 0.10}): no comparison involving MEIRA-full is ever lost."),

    ("6.1 bullet 2",
     "- **The components contribute in a stable order: memory > decay > "
     "XAI.** Removing episodic memory costs the most (ΔF1 +0.110 / +0.124) "
     "and induces memory mode collapse (MDS → 0.000); removing temporal "
     "decay costs the second-most (ΔF1 +0.078 / +0.086) on every ranking "
     "metric; removing the XAI head costs the least in ranking terms (ΔF1 "
     "+0.049 / +0.054) but is the sole driver of the explainability-adjusted "
     "score (ΔXAIR@10 +0.894 / +0.886) — i.e. the attribution head is what "
     "makes MEIRA's retrievals explainable, and XAIR@K responds exactly to "
     "the mechanism it measures.",
     "- **The components contribute in a stable order: memory > decay > "
     "XAI.** Removing episodic memory costs the most (ΔF1 +0.110 / +0.124) "
     "and induces memory mode collapse (MDS → 0.000); temporal decay is "
     "second (ΔF1 +0.078 / +0.086); the XAI head costs the least in ranking "
     "terms (ΔF1 +0.049 / +0.054) but is the sole driver of the "
     "explainability-adjusted score (ΔXAIR@10 +0.894 / +0.886)."),

    ("6.1 bullet 3",
     "- **The headline result is robust to how significance is decided.** F1 "
     "verdicts are 28/28 significant under every correction at every α; Holm "
     "changes exactly one verdict across the whole sweep (the no-decay > "
     "ColBERT-like boundary at α = 0.01); Bonferroni is where the threshold "
     "binds, losing at most the BM25-vs-TF-IDF tie and the no-decay/ColBERT "
     "boundary — the same two fragile families at every α — and **never any "
     "comparison involving MEIRA-full** (14 α-sensitive pair-instances in "
     "total, none of them ours).",
     "- **The headline result is robust to how significance is decided.** F1 "
     "verdicts are 28/28 significant under every correction at every α; Holm "
     "changes exactly one verdict (the no-decay > ColBERT-like boundary at "
     "α = 0.01); Bonferroni loses at most the BM25-vs-TF-IDF tie and the "
     "no-decay/ColBERT boundary — the same two fragile families at every α — "
     "and **never any comparison involving MEIRA-full** (14 α-sensitive "
     "pair-instances, none ours)."),

    # ---- 6.3 Future work ---------------------------------------------------
    ("6.3 decay schedule",
     "- **Learning the decay schedule.** The temporal-decay mechanism is "
     "currently a fixed schedule; learning its parameters from data — or "
     "conditioning forgetting on memory salience, in the spirit of "
     "cognitive-style memory consolidation — is a natural next step.",
     "- **Learning the decay schedule.** The temporal-decay mechanism is "
     "currently fixed; learning its parameters from data — or conditioning "
     "forgetting on memory salience — is a natural next step."),

    ("6.3 larger memory",
     "- **Larger and heterogeneous memory.** Vary the bank size S and the "
     "memory-slot semantics (episodes vs. facts vs. evidence) to "
     "characterise MDS more finely and test whether saturation (limitation "
     "5) is a metric artefact or a real ceiling.",
     "- **Larger and heterogeneous memory.** Vary the bank size S and slot "
     "semantics (episodes vs. facts vs. evidence) to characterise MDS more "
     "finely and test whether saturation (limitation 5) is a metric artefact "
     "or a real ceiling."),

    # ---- batch 2: extra breathing room -----------------------------------
    ("1.4 eval item",
     "**A rigorous, honest evaluation** — ten seeds, stratified 70/15/15 "
     "splits, conversation-level query pooling, paired t-tests with "
     "Holm-Bonferroni (primary) and Bonferroni (stress-test) multiplicity "
     "correction, and an α-sensitivity sweep over α ∈ {0.01, 0.05, 0.10}; "
     "the verdicts are checked for stability under every choice.",
     "**A rigorous, honest evaluation** — ten seeds, stratified 70/15/15 "
     "splits, conversation-level query pooling, paired t-tests with "
     "Holm-Bonferroni (primary) and Bonferroni (stress-test) correction, "
     "and an α-sensitivity sweep; the verdicts are checked for stability "
     "under every choice."),

    ("5 setup lead",
     "We assess whether the performance differences reported above are "
     "statistically reliable, and whether those conclusions survive choices "
     "about how significance is decided.",
     "We assess whether the reported performance differences are "
     "statistically reliable and survive choices about how significance is "
     "decided."),

    ("5 setup pairing",
     "Because the per-seed scores of all models come from the same test "
     "split within each seed, the comparisons are naturally paired, and the "
     "t-statistic preserves the direction of the difference (the model with "
     "the higher mean always appears on the left of the reported "
     "inequality).",
     "Because all models are scored on the same per-seed test splits, the "
     "comparisons are naturally paired and the t-statistic preserves "
     "direction (the higher-mean model appears on the left of each reported "
     "inequality)."),

    ("6.1 lead-in",
     "The empirical picture, established across ten evaluation seeds with a "
     "fully specified protocol (stratified 70/15/15 splits, "
     "conversation-level query pooling) and a rigorous statistical analysis "
     "(paired t-tests, df = 9, with Holm-Bonferroni primary and Bonferroni "
     "stress-test corrections plus an α-sensitivity sweep):",
     "The empirical picture, established across ten seeds with the protocol "
     "and statistical machinery of Sections 3–5 (paired t-tests, df = 9, "
     "Holm-Bonferroni primary and Bonferroni stress-test corrections, "
     "α-sensitivity sweep):"),

    ("6.2 lim1 tail",
     "Every table and figure in this paper must be regenerated from real "
     "inference before the numbers can be cited. (This is the reason this "
     "paper carries a single ⚠️ status note at the top.)",
     "Every table and figure in this paper must be regenerated from real "
     "inference before the numbers can be cited."),

    ("6.2 lim5 tail",
     "The current 64-slot bank may simply be easier to saturate than a "
     "real-world memory would be.",
     "A 64-slot bank may simply be easier to saturate than a real-world "
     "memory."),

    ("6.2 lim3 tail",
     "The paper's claim is superiority in *ranking quality*, and we say so "
     "explicitly; it cannot be read as a claim about raw precision at "
     "shallow cutoffs.",
     "The paper's claim is superiority in *ranking quality* — stated "
     "explicitly — not raw precision at shallow cutoffs."),

    ("6.2 lim6 tail",
     "Both are marginal-tier comparisons; neither involves MEIRA-full, but "
     "any claim about them must cite corrected p-values and the threshold "
     "caveat.",
     "Neither involves MEIRA-full, but any claim about them must cite "
     "corrected p-values and the threshold caveat."),

    ("5.2 opening",
     "To verify that the verdicts above are not an artifact of the α = 0.05 "
     "convention, we re-evaluate every comparison at α = 0.01 and α = 0.10 "
     "(p-values are threshold-independent; only the verdicts move). Two "
     "conclusions hold across the full sweep:",
     "To verify the verdicts are not an artifact of the α = 0.05 "
     "convention, we re-evaluate every comparison at α = 0.01 and α = 0.10 "
     "(p-values are threshold-independent; only verdicts move). Two "
     "conclusions hold:"),

    ("4.1 opening",
     "We compare **MEIRA-full** against four baselines spanning the "
     "classical and neural paradigms — BM25 and TF-IDF (lexical/sparse), "
     "and Dense-IR and ColBERT-like (dense/neural) — on both datasets and "
     "all 13 metrics. The full leaderboards are in Tables 3 and 4.",
     "We compare **MEIRA-full** against four baselines spanning the "
     "classical and neural paradigms — BM25 and TF-IDF (lexical/sparse), "
     "Dense-IR and ColBERT-like (dense/neural) — on both datasets and all "
     "13 metrics (full leaderboards: Tables 3–4)."),

    ("6.3 checkpoints tail",
     "- **Real checkpoints.** Replace `simulate_model()` with trained MEIRA "
     "forward passes (the rest of the pipeline is unchanged) and regenerate "
     "all numbers — this paper's structure is format-ready for this.",
     "- **Real checkpoints.** Replace `simulate_model()` with trained MEIRA "
     "forward passes (the rest of the pipeline is unchanged) and regenerate "
     "all numbers."),

    # ---- batch 3: recover the page-10 spill (figures added ~1.5pp) ------
    ("5.2 f1 invariant",
     "**F1 is invariant.** All 28 comparisons are significant under *every* "
     "correction at *every* α on both datasets. The F1 leaderboard — the "
     "metric most central to our claims — is completely insensitive to both "
     "the multiplicity correction and the threshold.",
     "**F1 is invariant.** All 28 comparisons are significant under every "
     "correction at every α on both datasets: the F1 leaderboard is "
     "completely insensitive to both the correction and the threshold."),

    ("5.2 holm nearly",
     "**Holm is nearly threshold-stable.** Across all 24 dataset × metric "
     "conditions (2 datasets × 4 metrics × 3 thresholds), Holm's adjustment "
     "itself flips exactly **one** raw-significant verdict: at the strictest "
     "threshold α = 0.01, MEIRA-no-decay > ColBERT-like on FIRE-AgentIR-2026 "
     "(raw p = 0.0071 < 0.01, but Holm p = 0.0141 > 0.01) drops out. The other "
     "count reductions at α = 0.01 — one pair each in nDCG@10 CrossLing, MAP "
     "CrossLing and MRR AgentIR, plus BM25 > TF-IDF on nDCG@10 AgentIR — are "
     "raw-threshold effects: those raw p-values (0.0105–0.0458) already exceed "
     "0.01. The MRR raw non-significance of BM25 vs TF-IDF is inherited by Holm "
     "at every threshold.",
     "**Holm is nearly threshold-stable.** Across the 24 dataset × metric × "
     "threshold conditions, Holm's adjustment flips exactly **one** "
     "raw-significant verdict: at α = 0.01, MEIRA-no-decay > ColBERT-like on "
     "AgentIR (raw p = 0.0071 < 0.01, Holm p = 0.0141 > 0.01) drops out. The "
     "other α = 0.01 reductions are raw-threshold effects — those raw "
     "p-values (0.0105–0.0458) already exceed 0.01 — and the MRR "
     "non-significance of BM25 vs TF-IDF is inherited by Holm at every "
     "threshold."),

    ("5.2 closing",
     "Consistently with Section 5.1, the unstable comparisons are exactly the "
     "BM25/TF-IDF tie, the no-decay/ColBERT boundary, and a handful of tight "
     "CrossLing ablation-tier gaps; in no case does a *direction* of an "
     "ordering flip, only the significance verdict.",
     "Consistently with Section 5.1, the unstable comparisons are exactly the "
     "BM25/TF-IDF tie, the no-decay/ColBERT boundary, and a few tight "
     "CrossLing ablation-tier gaps; no ordering *direction* ever flips, only "
     "the verdict."),

    ("6.2 lim1 compress",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified; only the score "
     "distributions are synthetic. Every table and figure in this paper must "
     "be regenerated from real inference before the numbers can be cited.",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified — only the "
     "score distributions are synthetic — and every table and figure must be "
     "regenerated from real inference before the numbers can be cited."),

    ("6.2 lim2 compress",
     "**The benchmarks are synthetic.** FIRE-AgentIR-2026 and "
     "FIRE-CrossLingIR-2026 are programmatically generated (deterministic at "
     "seed 42) to mirror FIRE-style task conventions; they are not collections "
     "of human-annotated documents. Their difficulty regimes (hard-negative "
     "density ≈1.6–1.8 per positive, label noise 5–6%) are realistic, but "
     "generalisation to naturalistic corpora and real user sessions is "
     "untested.",
     "**The benchmarks are synthetic.** FIRE-AgentIR-2026 and "
     "FIRE-CrossLingIR-2026 are programmatically generated (deterministic at "
     "seed 42) to mirror FIRE-style conventions, not collections of "
     "human-annotated documents. Their difficulty regimes (hard-negative "
     "density ≈1.6–1.8 per positive, label noise 5–6%) are realistic, but "
     "generalisation to naturalistic corpora and real sessions is untested."),

    ("6.2 lim4 compress",
     "**Component attributions are conservative, not orthogonal.** Temporal "
     "decay is defined *over* the memory bank, so the no-memory ablation "
     "disables decay as well; the memory Δ therefore absorbs decay's "
     "contribution. The ablation order (memory > decay > XAI) is thus a lower "
     "bound on memory's share rather than an orthogonal decomposition.",
     "**Component attributions are conservative, not orthogonal.** Because "
     "temporal decay is defined *over* the memory bank, the no-memory "
     "ablation disables decay as well; the memory Δ absorbs decay's "
     "contribution, so the ablation order is a lower bound on memory's share."),

    ("6.2 lim5 compress",
     "**MDS saturates.** MEIRA-full reaches MDS = 1.000±0.000 on both "
     "datasets, and the metric cannot distinguish \"fully utilised bank\" "
     "from \"bank too small for the task\". A 64-slot bank may simply be "
     "easier to saturate than a real-world memory.",
     "**MDS saturates.** MEIRA-full reaches MDS = 1.000±0.000 on both "
     "datasets, so the metric cannot distinguish \"fully utilised bank\" "
     "from \"bank too small for the task\": a 64-slot bank may simply be "
     "easier to saturate than a real-world memory."),

    ("6.2 lim6 head",
     "**Two comparison families are fragile.** BM25 > TF-IDF and "
     "MEIRA-no-decay > ColBERT-like are significant under Holm but not under "
     "Bonferroni at α = 0.05 (and the no-decay boundary also drops under Holm "
     "at α = 0.01).",
     "**Two comparison families are fragile.** BM25 > TF-IDF and "
     "MEIRA-no-decay > ColBERT-like are significant under Holm but not under "
     "Bonferroni at α = 0.05 (the no-decay boundary also drops under Holm at "
     "α = 0.01)."),

    ("6.2 lim7 compress",
     "**XAIR@K's scope.** The metric is defined only for models with an "
     "attribution head (baselines are marked \"—\"), and its w = 0.25 "
     "weighting is a design choice, not a tuned parameter; its behaviour "
     "under other weightings is unexplored.",
     "**XAIR@K's scope.** The metric is defined only for models with an "
     "attribution head (baselines are \"—\"), and its w = 0.25 weighting is "
     "a design choice whose behaviour under other values is unexplored."),

    ("6.3 naturalistic",
     "- **Naturalistic benchmarks.** Extend the two synthetic corpora with "
     "human-annotated, real-document counterparts (and, ideally, a FIRE "
     "shared task) to test generalisation of both the models and the two "
     "metrics.",
     "- **Naturalistic benchmarks.** Extend the two synthetic corpora with "
     "human-annotated, real-document counterparts (ideally via a FIRE shared "
     "task) to test generalisation of the models and metrics."),

    ("6.3 xair weight",
     "- **XAIR weight sensitivity.** Sweep w and study how the "
     "explainability-adjusted leaderboard changes — turning limitation 7 into "
     "a robustness result rather than a caveat.",
     "- **XAIR weight sensitivity.** Sweep w and turn limitation 7 into a "
     "robustness result rather than a caveat."),

    ("6.3 user studies",
     "- **User studies.** The ultimate test of explainability is human trust; "
     "a user study comparing MEIRA's attributed retrievals against "
     "attribution-free baselines would complement the XAIR@K evidence.",
     "- **User studies.** The ultimate test of explainability is human trust; "
     "a user study comparing MEIRA's attributed retrievals with "
     "attribution-free baselines would complement the XAIR@K evidence."),

    ("6.4 bottom line",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and that the advantage is robust to every "
     "statistical choice we made (correction method and threshold). With real "
     "inference and naturalistic benchmarks in place, the same architecture "
     "and the same two metrics are the paper's concrete, reusable "
     "contributions to agentic and explainable IR.",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made "
     "(correction method and threshold). With real inference and naturalistic "
     "benchmarks in place, the architecture and the two metrics are the "
     "paper's concrete, reusable contributions to agentic and explainable IR."),

    ("6.4 bottom line 2",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made "
     "(correction method and threshold). With real inference and naturalistic "
     "benchmarks in place, the architecture and the two metrics are the "
     "paper's concrete, reusable contributions to agentic and explainable IR.",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made. "
     "With real inference and naturalistic benchmarks in place, the "
     "architecture and the two metrics are the paper's concrete, reusable "
     "contributions to agentic and explainable IR."),

    ("6.4 bottom line 3",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made. "
     "With real inference and naturalistic benchmarks in place, the "
     "architecture and the two metrics are the paper's concrete, reusable "
     "contributions to agentic and explainable IR.",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made. "
     "With real inference and naturalistic benchmarks in place, the "
     "architecture and its two metrics are concrete, reusable contributions "
     "to agentic and explainable IR."),

    ("6.3 checkpoints 2",
     "- **Real checkpoints.** Replace `simulate_model()` with trained MEIRA "
     "forward passes (the rest of the pipeline is unchanged) and regenerate "
     "all numbers.",
     "- **Real checkpoints.** Replace `simulate_model()` with trained MEIRA "
     "forward passes and regenerate all numbers."),

    ("5.1 opening 2",
     "At α = 0.05 the raw analysis finds all 28 comparisons significant on "
     "both datasets for F1, nDCG@10 and MAP, and 27 of 28 for MRR; the single "
     "non-significant raw comparison is **BM25 vs TF-IDF** under MRR (p = "
     "0.3351 / 0.1554 on FIRE-AgentIR-2026 / FIRE-CrossLingIR-2026), a "
     "near-tie between the two classical baselines that is never significant "
     "at any threshold we consider.",
     "At α = 0.05 the raw analysis finds 28/28 comparisons significant on "
     "both datasets for F1, nDCG@10 and MAP, and 27/28 for MRR; the single "
     "non-significant raw comparison is **BM25 vs TF-IDF** under MRR (p = "
     "0.3351 / 0.1554 on FIRE-AgentIR-2026 / FIRE-CrossLingIR-2026), a "
     "near-tie that is never significant at any threshold we consider."),

    ("6.2 lim1 squeeze",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified — only the "
     "score distributions are synthetic — and every table and figure must be "
     "regenerated from real inference before the numbers can be cited.",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified; only the "
     "score distributions are synthetic, so every table and figure must be "
     "regenerated from real inference before the numbers can be cited."),

    ("6.3 user studies 2",
     "- **User studies.** The ultimate test of explainability is human trust; "
     "a user study comparing MEIRA's attributed retrievals with "
     "attribution-free baselines would complement the XAIR@K evidence.",
     "- **User studies.** A user study comparing MEIRA's attributed "
     "retrievals with attribution-free baselines would complement the XAIR@K "
     "evidence on human trust."),

    ("6.4 bottom line 4",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is not only feasible but "
     "measurably better — and robust to every statistical choice we made. "
     "With real inference and naturalistic benchmarks in place, the "
     "architecture and its two metrics are concrete, reusable contributions "
     "to agentic and explainable IR.",
     "MEIRA demonstrates that coupling episodic memory, temporal decay, and "
     "explainability in a single retrieval agent is feasible, measurably "
     "better, and robust to every statistical choice we made; with real "
     "inference and naturalistic benchmarks, the architecture and its two "
     "metrics are concrete, reusable contributions to agentic and "
     "explainable IR."),

    ("6.2 lim1 squeeze 2",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified; only the "
     "score distributions are synthetic, so every table and figure must be "
     "regenerated from real inference before the numbers can be cited.",
     "**The evaluation harness is currently simulated.** The single most "
     "important caveat: in the current codebase, `model_sim.py` draws each "
     "model's relevance scores from calibrated distributions instead of "
     "running trained checkpoints, so the reported magnitudes are "
     "*pipeline-validation artifacts*, not experimental results. The protocol, "
     "metrics, and statistical machinery are real and verified; only the "
     "score distributions are synthetic, so every table and figure must be "
     "regenerated before the numbers can be cited."),


    # ---- batch 4: final squeeze for the 9-page content limit -------------
    ("3.4 models prose",
     "MEIRA-full couples an episodic memory bank (64 slots) with a "
     "temporal-decay mechanism over memory and an XAI attribution head; the "
     "three ablation variants remove exactly one of the three headline "
     "components so their contribution can be isolated (Section 4.2). Note "
     "that temporal decay is defined *over* the memory bank, so the no-memory "
     "variant disables decay as well (both flags off in the registry); "
     "consequently the memory-ablation delta in the results of Section 4 "
     "absorbs decay's contribution as well — the component attributions are "
     "therefore conservative, not orthogonal. Memory-equipped models also "
     "report which memory slots they access per retrieval, which feeds MDS.",
     "MEIRA-full couples an episodic memory bank (64 slots) with a "
     "temporal-decay mechanism and an XAI attribution head; the three "
     "ablation variants remove exactly one of the three components so their "
     "contribution can be isolated (Section 4.2). Because temporal decay is "
     "defined *over* the memory bank, the no-memory variant disables decay as "
     "well (both flags off in the registry), so the memory-ablation delta "
     "absorbs decay's contribution — the attributions are conservative, not "
     "orthogonal. Memory-equipped models also report which slots they access "
     "per retrieval, which feeds MDS."),

    ("2.6 positioning",
     "In summary: MEIRA is an agentic retrieval system (Section 2.2) whose two "
     "defining mechanisms are memory with forgetting (Section 2.3) and "
     "explainability (Section 2.4), evaluated on benchmarks that stress hard "
     "negatives and cross-lingual, code-mixed multi-turn retrieval (Sections "
     "2.1, 2.2, 2.5). Its novelty is not any single mechanism — memory banks, "
     "decay schedules and attribution heads all exist in the literature — but "
     "their *coupling inside one retrieval agent* and, we argue, the "
     "*measurement* of them: XAIR@K and MDS make explainability and memory "
     "utilisation first-class citizens of the leaderboard rather than "
     "properties asserted in prose. The evaluation (Sections 4–5) then holds "
     "MEIRA to the same standard of statistical rigour as any leaderboard "
     "claim: paired tests, multiplicity correction, and a threshold sweep.",
     "In summary: MEIRA is an agentic retrieval system (Section 2.2) whose "
     "defining mechanisms are memory with forgetting (Section 2.3) and "
     "explainability (Section 2.4), evaluated on benchmarks that stress hard "
     "negatives and cross-lingual multi-turn retrieval (Sections 2.1, 2.2, "
     "2.5). Its novelty is not any single mechanism — all exist in the "
     "literature — but their *coupling inside one retrieval agent* and, we "
     "argue, the *measurement* of them: XAIR@K and MDS make explainability "
     "and memory utilisation first-class citizens of the leaderboard. The "
     "evaluation (Sections 4–5) then holds MEIRA to the same statistical "
     "rigour as any leaderboard claim: paired tests, multiplicity correction, "
     "and a threshold sweep."),

    ("3.2 outputs",
     "**Outputs.** Every script archives its results to a configuration-named "
     "subfolder (`results/k10_s10/` for the k-fold + multi-seed robustness "
     "suite; `results/s10/` for the SOTA and ablation runs) so different "
     "configurations are stored side-by-side and never overwrite each other.",
     "**Outputs.** Every script archives results to a configuration-named "
     "subfolder (`results/k10_s10/` for the robustness suite; `results/s10/` "
     "for the SOTA and ablation runs) so configurations never overwrite each "
     "other."),

    ("3.2 seeds",
     "**Evaluation seeds.** All multi-seed experiments use the seed range "
     "42…51 (ten seeds). For each seed, the models are scored on the held-out "
     "**test** partition of that seed's stratified split (≈15% of each class: "
     "≈1,260 samples per seed for FIRE-AgentIR-2026 and ≈480 for "
     "FIRE-CrossLingIR-2026); the k-fold experiment scores each of the 10 "
     "folds once (seed 42 + fold index). Reported values are **mean ± "
     "standard deviation across the ten evaluation seeds** (or across the ten "
     "folds).",
     "**Evaluation seeds.** All multi-seed experiments use seeds 42…51 "
     "(ten). For each seed, models are scored on the held-out **test** "
     "partition of that seed's stratified split (≈15% of each class: ≈1,260 "
     "samples per seed for FIRE-AgentIR-2026, ≈480 for FIRE-CrossLingIR-2026); "
     "the k-fold experiment scores each of the 10 folds once (seed 42 + fold "
     "index). Reported values are **mean ± std across the ten seeds** (or "
     "folds)."),

    ("3.2 query pooling",
     "**Query pooling.** Ranked metrics are computed at the conversation "
     "level: each conversation is treated as one query whose candidate pool "
     "(one positive and three negatives per turn) is ranked by the model's "
     "relevance scores. All flat predictions are regrouped into per-query "
     "pools before nDCG, MAP, MRR, R-Precision and P@K are computed, so the "
     "ranked metrics reflect real retrieval lists rather than "
     "instance-level classification.",
     "**Query pooling.** Ranked metrics are computed at the conversation "
     "level: each conversation is one query whose candidate pool (one "
     "positive and three negatives per turn) is ranked by the model's "
     "relevance scores, so the ranked metrics reflect real retrieval lists "
     "rather than instance-level classification."),

    ("3.3 standard metrics",
     "**Standard metrics.** From the classification suite we report F1, "
     "Precision, Recall, Accuracy, ROC-AUC and Average Precision. From the "
     "ranked-list suite we report nDCG@5, nDCG@10, MAP, MAP@10, MRR, "
     "R-Precision, P@5 and P@10. The main leaderboard (Tables 3–4 of Section "
     "4) reports the 13 headline metrics: F1, AUC, AP, nDCG@5, nDCG@10, MAP, "
     "MAP@10, MRR, R-Prec, P@5, P@10, XAIR@10, MDS.",
     "**Standard metrics.** From the classification suite we report F1, "
     "Precision, Recall, Accuracy, ROC-AUC and Average Precision; from the "
     "ranked-list suite, nDCG@5, nDCG@10, MAP, MAP@10, MRR, R-Precision, P@5 "
     "and P@10. The leaderboard (Tables 3–4) reports the 13 headline metrics: "
     "F1, AUC, AP, nDCG@5, nDCG@10, MAP, MAP@10, MRR, R-Prec, P@5, P@10, "
     "XAIR@10, MDS."),

    ("3.3 xair where",
     "where `w = 0.25` and `xai_conf(d) ∈ [0,1]` is the normalised XAI "
     "attribution confidence of document *d*. The metric interpolates ranking "
     "quality with explainability: a correct-but-unexplainable retrieval "
     "scores below a correct-and-explainable one, rewarding interpretable "
     "retrieval. XAIR@K is defined only for models with an XAI component (the "
     "MEIRA variants); baseline systems without attribution heads are marked "
     "“—” (they do not receive an unfair penalty).",
     "where `w = 0.25` and `xai_conf(d) ∈ [0,1]` is the normalised XAI "
     "confidence of document *d*: a correct-but-unexplainable retrieval "
     "scores below a correct-and-explainable one. XAIR@K is defined only for "
     "models with an XAI component (the MEIRA variants); baselines without "
     "attribution heads are marked “—” and receive no unfair penalty."),

    ("3.3 mds tail",
     "MDS ranges over [0, 1]; low values indicate memory under-utilisation "
     "(mode collapse) and a healthy bank is expected to exceed 0.3. MDS is "
     "defined only for models with an episodic memory component.",
     "MDS ranges over [0, 1]; low values indicate memory under-utilisation "
     "(mode collapse), a healthy bank exceeding 0.3. MDS is defined only for "
     "models with an episodic memory component."),

    ("6.1 tail",
     "Together these findings support the paper's central claim: a retrieval "
     "agent that remembers, forgets, and can explain its retrievals is better "
     "*and* more defensible than stateless lexical or neural baselines — and "
     "the improvement is not a statistical artifact of how we choose to test "
     "it.",
     "Together these findings support the paper's central claim: a retrieval "
     "agent that remembers, forgets, and can explain its retrievals is better "
     "*and* more defensible than stateless lexical or neural baselines — and "
     "not a statistical artifact of how we choose to test it."),

    ("6.3 larger memory 2",
     "- **Larger and heterogeneous memory.** Vary the bank size S and slot "
     "semantics (episodes vs. facts vs. evidence) to characterise MDS more "
     "finely and test whether saturation (limitation 5) is a metric artefact "
     "or a real ceiling.",
     "- **Larger and heterogeneous memory.** Vary the bank size S and slot "
     "semantics to characterise MDS more finely and test whether saturation "
     "(limitation 5) is a metric artefact or a real ceiling."),
]
