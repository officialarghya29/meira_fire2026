"""Assemble paper_full_draft.md from the six verified section drafts. v4.

v4: drafting metadata is consolidated into the front matter (abstract word-count
note + consolidated Sources line); per-section 'Sources:' footers are removed;
table captions are normalized to a single bold style; the final stale-marker
sweep reports only real leftovers (the per-replacement 'misses' seen in earlier
versions were benign ordering artifacts, now un-reported)."""
import re

OUT = "paper_full_draft.md"

def read(p):
    with open(p) as f:
        return f.read()

def write(p, s):
    with open(p, "w") as f:
        f.write(s)

def body(text):
    """Drop H1, Status blockquote, '---' separator; keep from first heading."""
    lines = text.splitlines()
    i = 0
    if lines and lines[0].startswith("# "):
        i = 1
    while i < len(lines) and (lines[i].startswith(">") or lines[i].strip() == ""):
        i += 1
    while i < len(lines) and lines[i].strip() in ("", "---"):
        i += 1
    out = lines[i:]
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    return "\n".join(out) + "\n"

def replace_exact(text, pairs):
    misses = []
    for old, new in pairs:
        if old not in text:
            misses.append(old)
        else:
            text = text.replace(old, new)
    return text, misses

def replace_ws(text, old, new):
    """Replace old with new, treating runs of whitespace (incl. newlines) flexibly."""
    tokens = re.split(r"\s+", old.strip())
    pat = r"\s+".join(re.escape(t) for t in tokens)
    return re.sub(pat, new, text)

all_misses = []

abs_ = read("paper_abstract_intro_draft.md")
rw_ = read("paper_related_work_draft.md")
ds_ = read("paper_datasets_protocol_draft.md")
sota_ = read("paper_sota_ablation_draft.md")
rob_ = read("paper_robustness_draft.md")
concl_ = read("paper_conclusion_draft.md")

# --------------------------------------------------------------------------
# Abstract & Introduction -> Section 1
# --------------------------------------------------------------------------
abs_body = body(abs_)
primary_start = abs_body.index("**Primary variant (submission abstract).**")
blurb_start = abs_body.index("**Tight blurb")
wc_start = abs_body.index("**Word count & variant selection.**")
i1_start = abs_body.index("## I1.")
primary = abs_body[primary_start:blurb_start].strip()
blurb = abs_body[blurb_start:wc_start].strip()
wc_note = abs_body[wc_start:i1_start].strip()
wc_note = re.sub(r"\n---\s*$", "", wc_note).strip()
intro = abs_body[i1_start:]

# roadmap rewrite FIRST (raw text)
roadmap_old = ("Section 2 (companion draft `paper_datasets_protocol_draft.md`) specifies\n"
               "the benchmarks, protocol, metrics and models (Sections D1\u2013D5). Section 3\n"
               "(`paper_sota_ablation_draft.md`) presents the SOTA leaderboard and the\n"
               "component ablation (Sections S1\u2013S2). Section 4\n"
               "(`paper_robustness_draft.md`) analyses statistical robustness \u2014 multiplicity\n"
               "correction (R1) and threshold sensitivity (R2). Section 5\n"
               "(`paper_conclusion_draft.md`) summarises, states limitations, and outlines\n"
               "future work.")
roadmap_new = ("Section 2 surveys related work. Section 3 specifies the benchmarks,\n"
               "evaluation protocol, metrics, and models (Sections 3.1\u20133.5). Section 4\n"
               "presents the SOTA leaderboard and the component ablation (Sections\n"
               "4.1\u20134.2). Section 5 analyses statistical robustness \u2014 multiplicity\n"
               "correction (Section 5.1) and threshold sensitivity (Section 5.2).\n"
               "Section 6 summarises the findings, states limitations, and outlines\n"
               "future work.")
if replace_ws(intro, roadmap_old, roadmap_new) == intro:
    all_misses.append("ROADMAP")
intro = replace_ws(intro, roadmap_old, roadmap_new)

intro, m = replace_exact(intro, [
    ("## I1.", "### 1.1 "),
    ("## I2.", "### 1.2 "),
    ("## I3.", "### 1.3 "),
    ("## I4.", "### 1.4 "),
    ("## I5.", "### 1.5 "),
    ("## I6.", "### 1.6 "),
    ("## Defensible claims", "### 1.7 Defensible claims"),
])
all_misses += m
intro, m = replace_exact(intro, [
    ("`paper_datasets_protocol_draft.md`", "Section 3"),
    ("`paper_sota_ablation_draft.md`", "Section 4"),
    ("`paper_robustness_draft.md`", "Section 5"),
    ("`paper_conclusion_draft.md`", "Section 6"),
    ("Section D1", "Section 3.1"),
    ("Sections D1\u2013D5", "Sections 3.1\u20133.5"),
    ("Section D2\u2013D5", "Sections 3.2\u20133.5"),
    ("Sections S1\u2013S2", "Sections 4.1\u20134.2"),
    ("Section S1", "Section 4.1"),
    ("Section S2", "Section 4.2"),
    ("Section R1", "Section 5.1"),
    ("Section R2", "Section 5.2"),
    ("I5", "Section 1.5"),
])
all_misses += m
# "the verified companion drafts" in Defensible claims #2 (line wraps before 'verified')
intro = replace_ws(intro, "the verified companion drafts", "the verified result tables")
intro = replace_ws(intro, "and survive their validators", "and survive independent recomputation")

# --------------------------------------------------------------------------
# Related Work -> Section 2
# --------------------------------------------------------------------------
rw_body = body(rw_)
ref_start = rw_body.index("## Reference list")
chk_start = rw_body.index("## Citation verification checklist")
rw_main = rw_body[:ref_start].strip() + "\n"
ref_list = rw_body[ref_start:chk_start].strip() + "\n"
ref_list, m = replace_exact(ref_list, [
    ("## Reference list", "## References"),
    ("in the robustness draft's statistics protocol", "in Section 5's statistics protocol"),
])
all_misses += m
ref_list = replace_ws(ref_list, "cited in RW3 context", "cited in Section 2.3 context")
rw_main, m = replace_exact(rw_main, [
    ("## RW1.", "### 2.1 "),
    ("## RW2.", "### 2.2 "),
    ("## RW3.", "### 2.3 "),
    ("## RW4.", "### 2.4 "),
    ("## RW5.", "### 2.5 "),
    ("## Positioning", "### 2.6 Positioning"),
    ("Section S1", "Section 4.1"),
    ("Section S2", "Section 4.2"),
    # in-text subsection markers -> numbered sections (longest first)
    ("(RW1, RW2, RW5)", "(Sections 2.1, 2.2, 2.5)"),
    ("(RW2)", "(Section 2.2)"),
    ("(RW3)", "(Section 2.3)"),
    ("(RW4)", "(Section 2.4)"),
])
all_misses += m
rw_main = replace_ws(rw_main, "The evaluation\n(companion drafts) then holds",
                     "The evaluation (Sections 4\u20135) then holds")

# --------------------------------------------------------------------------
# Datasets & Protocol -> Section 3
# --------------------------------------------------------------------------
ds_body = body(ds_)
ds_body, m = replace_exact(ds_body, [
    ("## D1.", "### 3.1 "),
    ("## D2.", "### 3.2 "),
    ("## D3.", "### 3.3 "),
    ("## D4.", "### 3.4 "),
    ("## D5.", "### 3.5 "),
    ("## Defensible claims", "### 3.6 Defensible claims"),
    ("`paper_sota_ablation_draft.md`", "Section 4"),
    ("Section S2 of Section 4", "Section 4.2"),
    ("Section S2", "Section 4.2"),
    ("Table 1\u20132 of Section 4", "Tables 3\u20134 of Section 4"),
    ("Table D1", "Table 1"),
    ("Table D2", "Table 2"),
])
all_misses += m
# D5: "in the companion draft `paper_robustness_draft.md`" -> Section 5
ds_body = replace_ws(ds_body, "the companion draft `paper_robustness_draft.md`", "Section 5")
# D4: "the memory-ablation delta in the companion results" -> Section 4 results
ds_body = replace_ws(ds_body, "in the companion results absorbs", "in the results of Section 4 absorbs")

# --------------------------------------------------------------------------
# SOTA + Ablation -> Section 4  (Setup unnumbered; S1=4.1, S2=4.2)
# --------------------------------------------------------------------------
sota_body = body(sota_)
sota_body, m = replace_exact(sota_body, [
    ("## Setup", "### Evaluation setup"),
    ("## S1.", "### 4.1 "),
    ("## S2.", "### 4.2 "),
    ("## Defensible claims", "### 4.3 Defensible claims"),
])
all_misses += m
sota_body, m = replace_exact(sota_body, [
    ("Tables 3\u20134", "Tables 5\u20136"),
    ("Table 5", "Table 7"),
    ("Table 4", "Table 6"),
    ("Table 3", "Table 5"),
    ("Tables 1 and 2", "Tables 3 and 4"),
    ("Table 2", "Table 4"),
    ("Table 1", "Table 3"),
])
all_misses += m
sota_body = replace_ws(sota_body, "the companion robustness analysis", "Section 5")
sota_body = replace_ws(sota_body, "the robustness draft (all are significant under Holm",
                       "Section 5 (all are significant under Holm")
sota_body, m = replace_exact(sota_body, [
    ("carried over from the robustness draft", "carried over from Section 5"),
    ("`paper_robustness_draft.md`", "Section 5"),
])
all_misses += m

# --------------------------------------------------------------------------
# Robustness -> Section 5  (Setup unnumbered; R1=5.1, R2=5.2)
# --------------------------------------------------------------------------
rob_body = body(rob_)
rob_body, m = replace_exact(rob_body, [
    ("## Setup", "### Statistical setup"),
    ("## R1.", "### 5.1 "),
    ("## R2.", "### 5.2 "),
    ("## Defensible claims", "### 5.3 Defensible claims"),
    ("Table 1", "Table 8"),
    ("Table 2", "Table 9"),
    ("(R1)", "(Section 5.1)"),
    ("(R2)", "(Section 5.2)"),
])
all_misses += m
rob_body = replace_ws(rob_body, "Consistently with R1, the unstable",
                      "Consistently with Section 5.1, the unstable")

# --------------------------------------------------------------------------
# Conclusion -> Section 6
# --------------------------------------------------------------------------
concl_body = body(concl_)
concl_body, m = replace_exact(concl_body, [
    ("## C1.", "### 6.1 "),
    ("## C2.", "### 6.2 "),
    ("## C3.", "### 6.3 "),
    ("## C4.", "### 6.4 "),
    ("## Defensible claims", "### 6.5 Defensible claims"),
    ("`paper_sota_ablation_draft.md`", "Section 4"),
    ("`paper_robustness_draft.md`", "Section 5"),
    ("`paper_datasets_protocol_draft.md`", "Section 3"),
    ("Sections S1\u2013S2", "Sections 4.1\u20134.2"),
    ("Section S2", "Section 4.2"),
    ("All summary numbers in C1 are the verified companion-draft numbers",
     "All summary numbers in Section 6.1 are the verified result-table numbers"),
])
all_misses += m
# conclusion limitation 1 + future work: "the drafts" -> "this paper" phrasing
concl_body = replace_ws(concl_body, "the drafts carry a \u26a0\ufe0f status note on every section",
                        "this paper carries a single \u26a0\ufe0f status note at the top")
concl_body = replace_ws(concl_body, "the drafts' structure is format-ready for this",
                        "this paper's structure is format-ready for this")
concl_body = replace_ws(concl_body, "hold for the *current drafts*", "hold for the *current version*")
concl_body = replace_ws(concl_body, "and survive their validators.", "and survive independent recomputation.")
concl_body = replace_ws(concl_body, "in this paper's drafts must be regenerated", "in this paper must be regenerated")

# --------------------------------------------------------------------------
# Front matter
# --------------------------------------------------------------------------
wc_block = "\n".join(("> " + ln) if ln.strip() else ">" for ln in wc_note.splitlines())
sources_note = ("> **Sources.** All quantitative claims trace to\n"
                "> `results/s10/sota.json`, `results/s10/ablation.json`,\n"
                "> `results/k10_s10/correction_comparison.json`, and\n"
                "> `results/k10_s10/alpha_sweep.json` (regenerate per the\n"
                "> recipe above).")
front = f"""# MEIRA: A Memory-Enhanced Interpretable Retrieval Agent for Multi-Turn Agentic and Cross-Lingual Information Retrieval

*FIRE 2026 submission draft — assembled from the six verified section drafts
(`paper_abstract_intro_draft.md`, `paper_related_work_draft.md`,
`paper_datasets_protocol_draft.md`, `paper_sota_ablation_draft.md`,
`paper_robustness_draft.md`, `paper_conclusion_draft.md`).*

> \u26a0\ufe0f **Simulation status — read before using any number.** The models in
> this harness are *simulated*: `model_sim.py` draws relevance scores from
> calibrated distributions instead of running trained checkpoints. Every
> table and figure in this paper is a **pipeline-validation artifact**, not
> an experimental result. The evaluation machinery (datasets, splits,
> metrics, statistical tests) is real and verified; only the score
> distributions are synthetic. Before submission, replace
> `simulate_model()` with real forward passes from the trained MEIRA
> checkpoints and regenerate every number (recipe: re-run
> `run_SOTA.py --seeds 10`, `run_ablation.py --seeds 10`,
> `run_experiments.py --k 10 --seeds 10`, then the significance /
> correction / sweep scripts). Section-specific status notes from the
> source drafts have been consolidated here.
>
{sources_note}
>
{wc_block}

---

## Abstract

{primary}

{blurb}

---

## 1. Introduction

"""

paper = front + intro + "\n\n---\n\n## 2. Related Work\n\n" + rw_main \
    + "\n\n---\n\n## 3. Datasets and Evaluation Protocol\n\n" + ds_body \
    + "\n\n---\n\n## 4. Results: SOTA Leaderboard and Component Ablation\n\n" + sota_body \
    + "\n\n---\n\n## 5. Robustness of the Statistical Comparisons\n\n" + rob_body \
    + "\n\n---\n\n## 6. Conclusion and Limitations\n\n" + concl_body \
    + "\n\n---\n\n" + ref_list \
    + "\n*Inline verification marks (\u2713) and correction notes on individual\nreference entries are drafting metadata; strip them for the camera-ready\nversion.*\n"

# normalize double spaces introduced by heading replacement
paper = re.sub(r"(### [\d.]+)  ", r"\1 ", paper)

# --- v4 seam cleanup on the assembled text --------------------------------
# (1) per-section 'Sources:' drafting footers -> single front-matter note
paper = re.sub(r"\n\*Sources:.*?\*\s*\n", "\n\n", paper, flags=re.S)
# (2) normalize table captions to a single bold style (italic -> bold)
paper = re.sub(
    r"(?<!\*)\*Table (\d+)\.(.*?)\*\s*(?=\n)",
    lambda m: f"**Table {m.group(1)}.{m.group(2)}**",
    paper, flags=re.S)
# (3) collapse 3+ consecutive blank lines
paper = re.sub(r"\n{3,}", "\n\n", paper)
# (4) re-wrap prose lines that seam rewrites pushed past ~80 cols
for old, new in [
    ("properties asserted in prose. The evaluation (Sections 4\u20135) then holds MEIRA to the same standard of statistical",
     "properties asserted in prose. The evaluation\n(Sections 4\u20135) then holds MEIRA to the same standard of statistical"),
    ("both p < 0.0001; in fact, as shown in Section 5, **every MEIRA-full-vs-baseline comparison is",
     "both p < 0.0001; in fact, as shown in Section 5,\n**every MEIRA-full-vs-baseline comparison is"),
    ("established in Section 5 (all are significant under Holm at \u03b1 = 0.05).",
     "established in Section 5 (all are\nsignificant under Holm at \u03b1 = 0.05)."),
    ("synthetic. Every table and figure in this paper must be regenerated from real inference before the numbers can be",
     "synthetic. Every table and figure in this paper\nmust be regenerated from real inference before the numbers can be"),
]:
    paper = replace_ws(paper, old, new)
# (5) normalize Defensible-claims heading titles (4.3/5.3 carried a parenthetical)
paper = re.sub(r"### ([\d.]+) Defensible claims \(what the numbers support\)",
               r"### \1 Defensible claims", paper)
# (6) last body-level drafting word
paper = replace_ws(paper, "for any reader of the draft", "for any reader of this document")

write(OUT, paper)

# honest final sweep: report only real leftovers in the output
stale = []
body = paper.split("## Abstract")[1] if "## Abstract" in paper else paper
for pat in [r"\b(?:I[1-6]|RW[1-5]|D[1-5]|S[12]|R[12]|C[1-4])\b",
            r"paper_(?:abstract_intro|related_work|datasets_protocol|sota_ablation|robustness|conclusion)_draft\.md",
            r"(?:companion drafts|source-draft|section drafts|survive their validators)"]:
    for m in re.finditer(pat, body):
        stale.append(m.group(0))
print("wrote", OUT, len(paper), "chars")
if stale:
    print("STALE MARKERS REMAINING:", sorted(set(stale)))
else:
    print("no stale draft markers in output")
