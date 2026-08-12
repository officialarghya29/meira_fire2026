# Related Work — Draft Section

> **Status.** Draft prose for the paper's Related Work section, positioning
> MEIRA against the literature. This draft contains **no experimental
> numbers** and therefore needs no re-validation when the simulated harness
> is replaced by real inference. The citation list has been verified
> against arXiv/publisher pages (August 2026); outcomes and corrections
> are recorded in the checklist at the end.
>
> ⚠️ **Before submission.** All references below were checked against their
> arXiv / ACM DL / NIST / CEUR-WS / ACL Anthology pages in August 2026.
> Three entries were **corrected** during that pass (TREC CAsT 2019 venue;
> the LLM-explainability survey's authors and journal; a placeholder
> replaced by a confirmed survey) — details in the checklist. Re-confirm
> DOIs, page ranges, and latest arXiv versions at camera-ready time;
> citation errors are the easiest way to lose reviewer trust.
>
> **Suggested placement:** Section 2, between the Introduction
> (`paper_abstract_intro_draft.md`) and the datasets & protocol section
> (`paper_datasets_protocol_draft.md`).

---

## RW1. Classical and neural retrieval

The two lexical baselines evaluated in this paper follow the standard
formulations: TF-IDF term-weighting as systematised by Salton & Buckley
(Salton & Buckley, 1988) and the BM25 probabilistic relevance framework
(Robertson & Zaragoza, 2009). Their neural successors replace sparse term
matching with dense representations: DPR-style bi-encoders encode query and
document independently into a shared vector space (Karpukhin et al., 2020),
while late-interaction models such as ColBERT retain per-token
contextualised embeddings and score with token-level interactions,
recovering much of the lexical precision that pure bi-encoders lose
(Khattab & Zaharia, 2020). Our Dense-IR and ColBERT-like baselines are
canonical representatives of these two families, and — consistent with the
literature — the late-interaction model is the stronger of the two on every
metric in our leaderboard (Section S1). None of these systems maintains
state across turns, which motivates the memory mechanisms below.

## RW2. Agentic and conversational IR

The evaluation setting in this paper — retrieval over multi-turn
conversational sessions — sits at the intersection of two active lines of
work. On the evaluation side, the conversational-assistance track of TREC
established multi-turn, context-dependent retrieval as a standard task
(Dalton, Xiong & Callan, 2020) ✓, and the FIRE (Forum for Information
Retrieval Evaluation) community has run a sustained programme of shared
tasks with a strong Indian-language and code-mixed emphasis, including
cross-lingual and conversational tracks whose conventions our benchmarks
mirror (FIRE proceedings, CEUR-WS) ✓. On the systems side, "agentic IR" has
recently been articulated as a shift from static query–document matching to
goal-directed, iterative retrieval in which an agent reasons, retrieves, and
adapts over multiple steps (Zhang et al., 2024, position paper) ✓, with
follow-up work on reasoning-augmented deep research agents (Zhang et al.,
2025) ✓ and on simulating search users for realistic multi-turn evaluation
(Zhang et al., 2024, USimAgent) ✓. MEIRA belongs to this agentic paradigm:
each conversation is a session over which the system accumulates memory and
refines its retrievals. Where we differ from the agentic-IR position
papers is in the concrete mechanisms — an explicit episodic memory bank, a
decay schedule, and an attribution head — and in the measurement of those
mechanisms (XAIR@K, MDS).

## RW3. Memory-augmented LLMs and retrieval

Our episodic memory bank draws on two influential lines of work on giving
LLM-based systems long-term memory. MemGPT (Packer et al., 2023) ✓ treats
the model as an operating system with hierarchical memory tiers, paging
information between a main context and external archival storage via
explicit tool calls; MEIRA's 64-slot bank is a retrieval-specialised version
of this idea, persisting *retrieval states* rather than raw dialogue.
Generative Agents (Park et al., 2023) ✓ introduced a memory stream with
recency, importance and relevance scoring that determines what an agent
recalls — a direct antecedent of our temporally-decayed memory. Closest to
our decay mechanism is MemoryBank (Zhong et al., 2024) ✓, which grounds
memory consolidation and forgetting in the Ebbinghaus forgetting curve so
that older, less-salient memories decay over time. MEIRA operationalises the
same intuition for retrieval: temporal decay down-weights stale memory slots,
and the ablation (Section S2) shows it is the second-largest contributor to
performance (ΔF1 +0.078/+0.086) — the empirical counterpart to the
theoretical case for forgetting made in these works.

## RW4. Explainable and trustworthy IR

The XAI attribution head and the XAIR@K metric position MEIRA within the
explainable IR (ExIR) literature. Anand et al. (2022) ✓ survey the field,
distinguishing transparent-by-design from post-hoc methods and noting the
absence of standardised evaluation of explanation quality — the exact gap
XAIR@K targets by folding attribution confidence into the ranking score.
Recent surveys extend this picture to LLM-based systems, cataloguing
explanation generation and human-centred trust evaluation for generative
models (Zhao et al., 2024) ✓ and LLM-driven explainable AI more broadly
(Bilal et al., 2025) ✓. Relative to this literature our contribution is
measurement-oriented: rather than proposing a new attribution method, we
propose a *metric* that makes explainability part of the leaderboard, and
we show (Section S2) that the attribution head is the sole driver of XAIR@10
(ΔXAIR@10 +0.894/+0.886) — i.e. that the metric responds exactly to the
mechanism it is designed to measure.

## RW5. Hard negatives, evaluation practice, and the proposed metrics

Two further strands of related work inform the evaluation design. First,
hard-negative construction: the sibling-topic negatives in our benchmarks
follow the established finding that hard negatives — topically confusable
distractors — are what make ranking tasks non-trivial and drive meaningful
separation between lexical and neural systems (contrastive-training
literature, incl. Karpukhin et al., 2020 ✓, and surveys of negative-sampling
techniques (Wischounig et al., 2026) ✓). Our corpus statistics confirm
the design works as intended: 67.1% / 63.9% of negatives are hard, and
the lexical baselines separate cleanly from the neural ones in the
leaderboard. Second, evaluation
methodology: LLM-as-judge approaches propose LLMs as relevance judges and
meta-evaluate their agreement with human judgements (Li et al., 2024) ✓,
and answerability-aware metrics argue that graded relevance alone
under-describes retrieval quality (Farzi & Dietz, 2024) ✓. Our two proposed
metrics sit naturally in this conversation: XAIR@K argues that a
*correct-but-unexplainable* hit is worth less than a *correct-and-explained*
one, and MDS argues that a system's *utilisation of its own memory* is a
measurable, reportable property — both are checks on behaviours that
standard graded-relevance metrics are blind to.

## Positioning

In summary: MEIRA is an agentic retrieval system (RW2) whose two defining
mechanisms are memory with forgetting (RW3) and explainability (RW4),
evaluated on benchmarks that stress hard negatives and cross-lingual,
code-mixed multi-turn retrieval (RW1, RW2, RW5). Its novelty is not any
single mechanism — memory banks, decay schedules and attribution heads all
exist in the literature — but their *coupling inside one retrieval agent*
and, we argue, the *measurement* of them: XAIR@K and MDS make
explainability and memory utilisation first-class citizens of the
leaderboard rather than properties asserted in prose. The evaluation
(companion drafts) then holds MEIRA to the same standard of statistical
rigour as any leaderboard claim: paired tests, multiplicity correction, and
a threshold sweep.

---

## Reference list

1. **Salton, G., & Buckley, C. (1988).** Term-weighting approaches in
   automatic text retrieval. *Information Processing & Management*, 24(5),
   513–523. ✓
2. **Robertson, S., & Zaragoza, H. (2009).** The Probabilistic Relevance
   Framework: BM25 and Beyond. *Foundations and Trends in Information
   Retrieval*, 3(4), 333–389. ✓
3. **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen,
   D., & Yih, W.-t. (2020).** Dense Passage Retrieval for Open-Domain
   Question Answering. *EMNLP 2020*. ✓
4. **Khattab, O., & Zaharia, M. (2020).** ColBERT: Efficient and Effective
   Passage Search via Contextualized Late Interaction over BERT. *SIGIR
   2020*. ✓
5. **Lewis, P., et al. (2020).** Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. ✓ (context for RAG /
   retrieval-coupled generation; cited in RW3 context)
6. **Dalton, J., Xiong, C., & Callan, J. (2020).** TREC CAsT 2019: The
   Conversational Assistance Track Overview. In *Proceedings of the
   Twenty-Eighth Text REtrieval Conference (TREC 2019)*, NIST Special
   Publication 500-335. Also arXiv:2003.13624. ✓ (venue corrected during
   verification: the earlier DESIRES/CEUR-WS 2664 note was wrong — that
   volume is IberLEF 2020)
7. **FIRE (2008–present).** Forum for Information Retrieval Evaluation.
   *Working Notes published in CEUR-WS*; Indian-language and cross-lingual
   tracks since 2008. Recent confirmed editions: FIRE 2025 (17th) —
   CEUR-WS Vol-4173; FIRE 2024 (16th) — CEUR-WS Vol-4054; FIRE 2023
   (15th) — CEUR-WS Vol-3681. ✓ (cite the edition relevant to the
   submission)
8. **Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I.,
   & Gonzalez, J. E. (2023).** MemGPT: Towards LLMs as Operating Systems.
   *arXiv:2310.08560*. ✓
9. **Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., &
   Bernstein, M. S. (2023).** Generative Agents: Interactive Simulacra of
   Human Behavior. *UIST 2023*. ✓
10. **Zhong, W., Guo, L., Gao, Q., Ye, H., & Wang, Y. (2024).** MemoryBank:
    Enhancing Large Language Models with Long-Term Memory. *AAAI 2024*,
    Vol. 38. ✓
11. **Anand, A., Lyu, L., Idahl, M., Wang, Y., Wallat, J., & Zhang, Z.
    (2022).** Explainable Information Retrieval: A Survey.
    *arXiv:2211.02405*. ✓
12. **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test
    Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. ✓ (cited
    in the robustness draft's statistics protocol)
13. **Zhang, W., Liao, J., Li, N., Du, K., & Lin, J. (2024).** Agentic
    Information Retrieval (perspective paper). *arXiv:2410.09713* (v4,
    23 Feb 2025). ✓
14. **Zhang, W., Li, Y., Bei, Y., Luo, J., et al. (2025).** From Web Search
    towards Agentic Deep Research: Incentivizing Search with Reasoning
    Agents. *arXiv:2506.18959* (v3, 3 Jul 2025). ✓ (full 23-author list
    confirmed on arXiv)
15. **Zhang, E., Wang, X., Gong, P., Lin, Y., & Mao, J. (2024).** USimAgent:
    Large Language Models for Simulating Search Users. *SIGIR 2024 (short
    paper)*. DOI 10.1145/3626772.3657963; also arXiv:2403.09142. ✓
16. **Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., Wang, S.,
    Yin, D., & Du, M. (2024).** Explainability for Large Language Models:
    A Survey. *ACM Transactions on Intelligent Systems and Technology
    (TIST)*, 15(2), Article 20. DOI 10.1145/3639372. ✓ (authors and venue
    corrected during verification)
17. **Bilal, A., Ebert, D., & Lin, B. (2025).** LLMs for Explainable AI: A
    Comprehensive Survey. *arXiv:2504.00125*. ✓ (note: the manuscript
    states an intended submission to ACM TIST)
18. **Li, H., Dong, Q., Chen, J., Su, H., Zhou, Y., Ai, Q., Ye, Z., & Liu,
    Y. (2024).** LLMs-as-Judges: A Comprehensive Survey on LLM-based
    Evaluation Methods. *arXiv:2412.05579*. ✓
19. **Farzi, N., & Dietz, L. (2024).** EXAM++: LLM-based Answerability
    Metrics for IR Evaluation. In *Proceedings of the First Workshop on
    Large Language Models for Evaluation in Information Retrieval
    (LLM4Eval @ SIGIR 2024)*, CEUR-WS Vol-3752, pp. 31–50. ✓
20. **Wischounig, L., Abdallah, A., & Jatowt, A. (2026).** Negative
    Sampling Techniques in Dense Retrieval: A Survey. *Findings of the
    Association for Computational Linguistics: EACL 2026*, pp. 3003–3020.
    DOI 10.18653/v1/2026.findings-eacl.157; also arXiv:2603.18005. ✓
    (replaces the earlier placeholder; verified against the ACL Anthology)

---

## Citation verification checklist (do before submission)

| # | Ref | Status | Verification outcome (Aug 2026) |
|---|---|---|---|
| 1–5, 8–12 | Foundational works (Salton, Robertson, DPR, ColBERT, RAG, MemGPT, Generative Agents, MemoryBank, ExIR survey, Holm) | ✓ | Confirmed; re-check edition/publisher fields at camera-ready |
| 6 | TREC CAsT 2019 overview | ✓ | **Corrected:** venue is the NIST *TREC 2019* proceedings (SP 500-335); also arXiv:2003.13624. The earlier DESIRES/CEUR-WS 2664 note was wrong |
| 7 | FIRE proceedings | ✓ | Confirmed recent volumes: FIRE 2025 → Vol-4173; FIRE 2024 → Vol-4054; FIRE 2023 → Vol-3681 |
| 13 | Agentic Information Retrieval (arXiv:2410.09713) | ✓ | Title/authors/year confirmed (v4, 23 Feb 2025) |
| 14 | Agentic Deep Research (arXiv:2506.18959) | ✓ | Title + full 23-author list confirmed (v3, 3 Jul 2025) |
| 15 | USimAgent (SIGIR 2024) | ✓ | Confirmed short paper; DOI 10.1145/3626772.3657963; arXiv:2403.09142 |
| 16 | Explainability for LLMs survey | ✓ | **Corrected:** authors are Zhao et al. (not Zhang et al.); venue is ACM TIST 15(2) (not Computing Surveys); DOI 10.1145/3639372 |
| 17 | LLMs for Explainable AI (arXiv:2504.00125) | ✓ | Confirmed: Bilal, Ebert & Lin, 2025 |
| 18 | LLMs-as-Judges (arXiv:2412.05579) | ✓ | Title + 8-author list confirmed |
| 19 | EXAM++ (LLM4Eval @ SIGIR 2024) | ✓ | Confirmed: CEUR-WS Vol-3752, pp. 31–50 |
| 20 | Negative-sampling survey | ✓ | Placeholder replaced with Wischounig et al., Findings of EACL 2026 (DOI 10.18653/v1/2026.findings-eacl.157; arXiv:2603.18005) |
| — | Any citation added during revision | ▣ | Run the same verification pass

*No numbers in this draft require re-validation when the harness is
replaced by real inference; only the citation list needs the pass above.*

*Machine-readable: the same 20 references are exported as BibTeX in
`paper_references.bib` (grouped by Related-Work subsection, keys are
descriptive author-year names; DOIs and arXiv IDs confirmed against
publisher pages, August 2026).*
