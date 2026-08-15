"""md2tex.py - convert paper_full_draft.md into a FIRE 2026 LaTeX skeleton.

Target: ACM ICPS "sigconf" (acmart) template as required by the FIRE 2026 CFP
(https://fire.irsi.org.in/fire/2026/call_for_papers). Produces:

    latex/main.tex                      - documentclass, title, CCS, keywords,
                                          abstract, \\input of the sections,
                                          bibliography
    latex/sections/sec01_..sec06.tex    - the six converted section files
    latex/paper_references.bib          - copy of the verified bibliography

Design: the markdown is converted *programmatically* so the verified numbers
survive verbatim. Drafting metadata (simulation-status notes, "Defensible
claims" subsections, "Sources:" footers, word-count note, verification marks)
is demoted to LaTeX comments so the compiled paper is camera-ready while the
notes stay available to the authors. In-text section/table references are
rewritten to \\Cref/\\Crefrange, and author-year prose citations are rewritten
to \\citep/\\citet against paper_references.bib.
"""
import os
import re
import shutil

from _trims import TRIMS   # camera-ready prose trims (redundant-text removal)

ROOT = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(ROOT, "paper_full_draft.md")
BIB_SRC = os.path.join(ROOT, "paper_references.bib")
OUTDIR = os.path.join(ROOT, "latex")
SECDIR = os.path.join(OUTDIR, "sections")

# ---------------------------------------------------------------------------
# label maps (assembled numbering -> LaTeX label)
# ---------------------------------------------------------------------------
SEC_LABELS = {
    "1": "sec:intro", "2": "sec:rw", "3": "sec:datasets", "4": "sec:results",
    "5": "sec:rob", "6": "sec:concl",
    "1.1": "sec:intro-motivation", "1.2": "sec:intro-gap", "1.3": "sec:intro-approach",
    "1.4": "sec:intro-contributions", "1.5": "sec:intro-findings", "1.6": "sec:intro-roadmap",
    "2.1": "sec:rw-classical", "2.2": "sec:rw-agentic", "2.3": "sec:rw-memory",
    "2.4": "sec:rw-xai", "2.5": "sec:rw-hardneg", "2.6": "sec:rw-positioning",
    "3.1": "sec:datasets-benchmarks", "3.2": "sec:datasets-protocol",
    "3.3": "sec:datasets-metrics", "3.4": "sec:datasets-models",
    "3.5": "sec:datasets-stats",
    "4.1": "sec:results-sota", "4.2": "sec:results-ablation",
    "5.1": "sec:rob-corr", "5.2": "sec:rob-alpha",
    "6.1": "sec:concl-summary", "6.2": "sec:concl-limitations",
    "6.3": "sec:concl-future", "6.4": "sec:concl-bottomline",
}
TAB_LABELS = {
    "1": "tab:datasets", "2": "tab:models", "3": "tab:sota-agent",
    "4": "tab:sota-cross", "5": "tab:abl-agent", "6": "tab:abl-cross",
    "7": "tab:deltas", "8": "tab:corr-counts", "9": "tab:lost-pairs",
}
FIG_LABELS = {
    "1": "fig:sota", "2": "fig:ablation", "3": "fig:ordering",
}

# ---------------------------------------------------------------------------
# citation forms from paper_references.bib
# ---------------------------------------------------------------------------
def parse_authors(author_field):
    surnames = []
    for part in re.split(r"\s+and\s+", author_field):
        part = part.strip()
        if "," in part:
            surnames.append(part.split(",")[0].strip())
        else:
            surnames.append(part.strip())
    return surnames

def citation_forms(surnames, year):
    forms = []
    if len(surnames) == 1:
        forms.append(f"{surnames[0]}, {year}")
    elif len(surnames) == 2:
        forms.append(f"{surnames[0]} & {surnames[1]}, {year}")
    elif len(surnames) == 3:
        forms.append(f"{surnames[0]}, {surnames[1]} & {surnames[2]}, {year}")
        forms.append(f"{surnames[0]} et al., {year}")
    else:
        forms.append(f"{surnames[0]} et al., {year}")
    return forms

import bibtexparser
from bibtexparser.bparser import BibTexParser

CITE_FORMS = []          # (form, key), longest first
NARRATIVE = {}           # form prefix -> key
with open(BIB_SRC) as f:
    bib = bibtexparser.load(f, parser=BibTexParser(common_strings=True))
BIB_KEYS = [e["ID"] for e in bib.entries]
for e in bib.entries:
    year = e.get("year", "")
    surnames = parse_authors(e.get("author", ""))
    for form in citation_forms(surnames, year):
        CITE_FORMS.append((form, e["ID"]))
    if len(surnames) == 1:
        NARRATIVE[f"{surnames[0]}, {year}"] = e["ID"]
    elif len(surnames) == 2:
        NARRATIVE[f"{surnames[0]} & {surnames[1]}, {year}"] = e["ID"]
    elif len(surnames) >= 3:
        NARRATIVE[f"{surnames[0]}, {surnames[1]} & {surnames[2]}, {year}"] = e["ID"]
        NARRATIVE[f"{surnames[0]} et al., {year}"] = e["ID"]
CITE_FORMS.sort(key=lambda x: -len(x[0]))
NARRATIVE_ITEMS = sorted(NARRATIVE.items(), key=lambda x: -len(x[0]))

# ---------------------------------------------------------------------------
# inline markdown -> LaTeX
# ---------------------------------------------------------------------------
CODE_SPAN = re.compile(r"`([^`]+)`")

def tex_code_span(s):
    """Escape a code span for \\texttt, then map unicode symbols to math.
    Long file-path spans get discretionary break points after / and . so they
    do not overflow the column."""
    s = (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
         .replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
         .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
         .replace("{", r"\{").replace("}", r"\}").replace("$", r"\$"))
    s = uni_replace(s)
    if len(s) > 40:                       # long spans only; {} ends the
        s = s.replace("/", r"/\allowbreak{}").replace(".", r".\allowbreak{}")
    return s


def convert_citations(s):
    """Author-year prose citations -> @@CITEP/@@CITET placeholders (resolved
    after escaping, so the inserted braces survive)."""
    s = s.replace("(Zhang et al., 2024, position paper)", "(@@CITEP:zhang2024agentic@@)")
    s = s.replace("(Zhang et al., 2024, USimAgent)", "(@@CITEP:zhang2024usimagent@@)")
    s = s.replace("(FIRE proceedings, CEUR-WS)", "(FIRE proceedings @@CITEP:fire2025proceedings@@)")
    for form, key in CITE_FORMS:                     # parenthetical
        s = re.sub(r"\(\s*" + re.escape(form) + r"\s*\)",
                   lambda m, k=key: "(@@CITEP:" + k + "@@)", s)
    for form, key in CITE_FORMS:                     # naked mid-sentence
        s = re.sub(r"(?<![\w(])" + re.escape(form) + r"(?![\w)])",
                   lambda m, k=key: "@@CITET:" + k + "@@", s)
    for form, key in NARRATIVE_ITEMS:                # "Surname et al. (YYYY)"
        base = form.rsplit(",", 1)[0]
        s = re.sub(r"(?<![\w(])" + re.escape(base) + r"\s*\((\d{4})\)",
                   lambda m, k=key: "@@CITET:" + k + "@@", s)
    return s


def convert_refs(s):
    """'Section N.M' / 'Table N' / ranges -> @@CREF placeholders (resolved
    after escaping, so inserted braces survive; runs before the en-dash
    unicode rewrite, so '3.1--3.5' ranges still match on the raw '\u2013')."""
    def tab_ref(a, b):
        if b is not None and a in TAB_LABELS and b in TAB_LABELS:
            return "@@CREFRANGE|" + TAB_LABELS[a] + "|" + TAB_LABELS[b] + "|@@"
        if a in TAB_LABELS:
            return "@@CREF:" + TAB_LABELS[a] + "@@"
        return None
    def table_range(m):
        r = tab_ref(m.group(1), m.group(2) or m.group(3))
        return r if r else m.group(0)
    s = re.sub(r"\bTables\s+(\d+)(?:\s+(?:and|to)\s+(\d+)|[\u2013-](\d+))?",
               table_range, s)
    s = re.sub(r"\bTable\s+(\d+)\b",
               lambda m: tab_ref(m.group(1), None) or m.group(0), s)
    def fig_ref(m):
        a = m.group(1)
        if a in FIG_LABELS:
            return "@@CREF:" + FIG_LABELS[a] + "@@"
        return m.group(0)
    s = re.sub(r"\bFigure\s+(\d+)\b", fig_ref, s)
    def sec_range(m):
        a, b = m.group(1), m.group(2)
        if a in SEC_LABELS and b in SEC_LABELS:
            return "@@CREFRANGE|" + SEC_LABELS[a] + "|" + SEC_LABELS[b] + "|@@"
        return m.group(0)
    s = re.sub(r"\bSections\s+([\d.]+)\s*(?:[\u2013-]|to)\s*([\d.]+)", sec_range, s)
    def sec_list(m):
        # 'Sections 2.1, 2.2, 2.5' -> \Cref{a,b,c} (cleveref multi-target)
        nums = [n for n in re.split(r"[\s,]+", m.group(0))[1:] if n]
        labs = [SEC_LABELS[n] for n in nums if n in SEC_LABELS]
        if labs and len(labs) == len(nums):
            return "@@CREFLIST|" + ",".join(labs) + "|@@"
        return m.group(0)
    s = re.sub(r"\bSections\s+[\d.]+(?:\s*,\s*[\d.]+)+", sec_list, s)
    def sec_ref(m):
        a = m.group(1)
        if a in SEC_LABELS:
            return "@@CREF:" + SEC_LABELS[a] + "@@"
        return m.group(0)
    s = re.sub(r"\bSection\s+([\d.]+)\b", sec_ref, s)
    return s


UNI = {
    "α": r"$\alpha$", "Δ": r"$\Delta$", "δ": r"$\delta$", "β": r"$\beta$",
    "∈": r"$\in$", "±": r"$\pm$", "×": r"$\times$", "≤": r"$\leq$",
    "≥": r"$\geq$", "≈": r"$\approx$", "→": r"$\rightarrow$",
    "−": r"$-$", "·": r"$\cdot$", "≳": r"$\geq$", "…": r"\ldots",
    "—": "---", "–": "--", "✓": "", "✗": "", "▣": "", "⚠️": "", "⚠": "",
    "“": "``", "”": "''", "‘": "`", "’": "'",
}


def uni_replace(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


_TRIM_APPLIED, _TRIM_MISSED = [], []


def apply_trims(s):
    """Apply the camera-ready prose trims to a joined paragraph.
    Exact-match only: every headline number/citation/ref is preserved."""
    for name, old, new in TRIMS:
        if old in s:
            s = s.replace(old, new)
            if name not in _TRIM_APPLIED:
                _TRIM_APPLIED.append(name)
        else:
            # once applied, never report as missed (later paragraphs
            # legitimately do not contain this trim's text)
            if name not in _TRIM_APPLIED and name not in _TRIM_MISSED:
                _TRIM_MISSED.append(name)
    return s


def report_trims():
    never = [n for n in _TRIM_MISSED if n not in _TRIM_APPLIED]
    print("trims: applied", len(_TRIM_APPLIED), "| never-applied", len(never))
    for name in never:
        print(f"  TRIM MISSED: {name}")


def resolve_placeholders(s):
    """Resolve @@...@@ placeholders inserted by convert_citations/convert_refs
    and the code-span pass (after escaping, so inserted braces survive)."""
    s = re.sub(r"@@CREFRANGE\|([^|]+)\|([^|]+)\|@@",
               lambda m: "\\Crefrange{" + m.group(1) + "}{" + m.group(2) + "}", s)
    s = re.sub(r"@@CREFLIST\|([^|]+)\|@@",
               lambda m: "\\Cref{" + m.group(1) + "}", s)
    s = re.sub(r"@@CREF:([^@]+)@@", lambda m: "\\Cref{" + m.group(1) + "}", s)
    s = re.sub(r"@@CITEP:([^@]+)@@", lambda m: "\\citep{" + m.group(1) + "}", s)
    s = re.sub(r"@@CITET:([^@]+)@@", lambda m: "\\citet{" + m.group(1) + "}", s)
    s = re.sub(r"@@CODE:([^@]+)@@", lambda m: "\\texttt{" + m.group(1) + "}", s)
    return s


def tex_inline(s):
    """Convert a plain markdown paragraph to LaTeX."""
    s = convert_citations(s)
    s = convert_refs(s)
    s = CODE_SPAN.sub(lambda m: "@@CODE:" + tex_code_span(m.group(1)) + "@@", s)

    # targeted math phrases
    s = re.sub(r"(\d(?:\.\d+)?)\s*±\s*(\d(?:\.\d+)?)", r"\1$\\pm$\2", s)
    s = s.replace("w = 0.25", "$w = 0.25$")
    s = re.sub(r"α\s*∈\s*\{([^}]*)\}", r"$\\alpha \\in \\{\1\\}$", s)
    s = re.sub(r"α\s*=\s*(\d(?:\.\d+)?)", r"$\\alpha = \1$", s)
    s = re.sub(r"Δ([A-Z@0-9]+)", r"$\\Delta$\1", s)
    def p_math(m):
        op = {"<": "<", ">": ">", "=": "=", "≤": r"\leq", "≥": r"\geq"}[m.group(1)]
        # emit \alpha directly: uni_replace runs later and would otherwise
        # turn a raw α inside the f-string into $\alpha$, breaking the math
        val = r"\alpha" if m.group(2) == "α" else m.group(2)
        return f"$p {op} {val}$"
    s = re.sub(r"\bp\s*([<≤>≥=])\s*((?:\d(?:\.\d+)?)|α)", p_math, s)
    s = re.sub(r"\bt\s*=\s*(\d+\.\d+)", r"$t = \1$", s)
    s = s.replace("1 − min(0.07·turn, 0.35)",
                  r"$1 - \\min(0.07\\cdot\\text{turn}, 0.35)$")
    s = re.sub(r"(\d)…(\d)", r"\1--\2", s)
    s = s.replace("S = 64", "$S = 64$")
    s = s.replace("df = 9", "$df = 9$")

    # unicode -> LaTeX
    s = uni_replace(s)

    # escape specials FIRST (so inserted \textbf/\emph braces survive)
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)#", r"\\#", s)
    s = re.sub(r"(?<!\\)_", r"\\_", s)
    s = re.sub(r"(?<!\\)\{", r"\\{", s)
    s = re.sub(r"(?<!\\)\}", r"\\}", s)

    # bold, then italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)

    # straight double quotes -> `` ''
    out, in_q = [], False
    for ch in s:
        if ch == '"':
            out.append("``" if not in_q else "''")
            in_q = not in_q
        else:
            out.append(ch)
    return resolve_placeholders("".join(out))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
TABLE_CAP_RE = re.compile(r"^\*\*Table\s+(\d+)\.\s*(.*?)\s*\*\*$", re.S)

def is_table_row(ln):
    return ln.strip().startswith("|")

def locate_tables(body_lines):
    """{start_idx: (end_idx, rows, caption_or_None, label_or_None)}

    Line-based scanner: caption runs start with '**Table N.' and end at a line
    ending in '**'; table runs are consecutive '|' lines. Each table is paired
    with its nearest *unused* caption (forward or backward), which correctly
    handles captions glued to the following table/prose (no blank line)."""
    n = len(body_lines)
    caps = []  # (line_idx, num, caption_text)
    tabs = []  # (start_idx, end_idx, rows)
    i = 0
    while i < n:
        ln = body_lines[i].strip()
        if ln.startswith("**Table ") and re.match(r"^\*\*Table \d+\.", ln):
            j = i
            parts = []
            while j < n:
                l2 = body_lines[j].strip()
                if not l2 or (j > i and l2.startswith("|")):
                    break
                parts.append(l2)
                j += 1
                if l2.endswith("**"):
                    break
            cap_text = " ".join(parts)
            m = re.match(r"^\*\*Table (\d+)\.\s*(.*?)\s*\*\*$", cap_text, re.S)
            if m:
                caps.append((i, m.group(1), m.group(2)))
            i = j
            continue
        if ln.startswith("|"):
            j = i
            rows = []
            while j < n and body_lines[j].strip().startswith("|"):
                rows.append(body_lines[j])
                j += 1
            tabs.append((i, j, rows))
            i = j
            continue
        i += 1
    used = set()
    result = {}
    for start, end, rows in tabs:
        best = None
        for ci, num, cap in caps:
            if num in used:
                continue
            d = abs(ci - start)
            if best is None or d < best[0]:
                best = (d, num, cap)
        if best:
            used.add(best[1])
            label = TAB_LABELS.get(best[1])
            if label is None:
                print(f"WARN: Table {best[1]} has no TAB_LABELS entry - "
                      "no \\label emitted (Crefs to it would break)")
            result[start] = (end, rows, best[2], label)
        else:
            result[start] = (end, rows, None, None)
    return result


def convert_table(rows, caption, label):
    parsed = []
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        parsed.append(cells)
    ncols = max(len(r) for r in parsed)
    wide = ncols >= 7 or any(len(c) > 14 for r in parsed for c in r)
    very_wide = ncols >= 12
    env = "table*" if wide else "table"
    # long free-text last columns (e.g. model "role"/description) -> paragraph
    # columns so the row wraps instead of overflowing the line width
    last_long = any(len(r[ncols - 1]) > 30 for r in parsed if len(r) == ncols)
    if last_long:
        spec = "l" + "c" * (ncols - 2) + ("p{0.5\\textwidth}" if env == "table*"
                                           else "p{0.7\\columnwidth}")
    else:
        spec = "l" + "c" * (ncols - 1)
    if very_wide:
        body = [f"\\begin{{{env}}}[tb]", "\\setlength{\\tabcolsep}{2pt}",
                "\\scriptsize"]
    elif wide:
        body = [f"\\begin{{{env}}}[tb]", "\\setlength{\\tabcolsep}{3pt}",
                "\\footnotesize"]
    else:
        body = [f"\\begin{{{env}}}[tb]", "\\small"]
    body += [f"\\caption{{{tex_inline(caption)}}}"]
    if label:
        body.append(f"\\label{{{label}}}")
    body += ["\\centering", "\\begin{tabular}{" + spec + "}", "\\toprule"]
    for idx, r in enumerate(parsed):
        cells = []
        for c in r:
            c = convert_citations(c)      # keep table cells consistent with prose
            c = convert_refs(c)
            c = c.replace("±", r"$\pm$").replace("—", "---").replace("–", "--")
            c = c.replace("✓", r"$\checkmark$").replace("✗", r"$\times$")
            c = c.replace("−", r"$- $")
            # escape LaTeX specials FIRST (so inserted \textbf braces survive)
            c = re.sub(r"(?<!\\)&", r"\\&", c)
            c = re.sub(r"(?<!\\)%", r"\\%", c)
            c = re.sub(r"(?<!\\)#", r"\\#", c)
            c = re.sub(r"(?<!\\)_", r"\\_", c)
            c = re.sub(r"(?<!\\)\{", r"\\{", c)
            c = re.sub(r"(?<!\\)\}", r"\\}", c)
            c = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", c)
            cells.append(resolve_placeholders(c))
        row = " & ".join(cells)
        if len(cells) < ncols:
            row += " & " * (ncols - len(cells))
        body.append(row.rstrip() + r" \\")
        if idx == 0:
            body.append("\\midrule")
    body += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(body)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
# Markdown image syntax: ![caption](path){#fig:label}
FIG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)(?:\{([^}]*)\})?\s*$", re.S)

def convert_figure(alt, path, attrs):
    """Convert a markdown image line into a full-width figure* environment.
    The alt text doubles as the caption (ACM TAPS encourages alt text); the
    {#label} attribute becomes the \\label. Image paths are project-relative
    (figures/...) and rewritten to ../figures/... for the latex/ build dir."""
    label = ""
    if attrs:
        m = re.search(r"#([A-Za-z0-9:_-]+)", attrs)
        if m:
            label = m.group(1)
    if not os.path.exists(os.path.join(ROOT, path)):
        print(f"WARN: figure file not found: {path}")
    if not path.startswith("figures/"):
        print(f"WARN: image path not under figures/: {path} "
              "(includegraphics resolves relative to the latex/ build dir)")
    tex_path = "../" + path if path.startswith("figures/") else path
    body = [r"\begin{figure*}[tb]", r"\centering",
            # 0.94\textwidth (not full width): the three figures were sized
            # for nominal font rendering at full width; this small reduction
            # helps the paper stay within the 9-page content limit while the
            # 300-dpi source images keep their tuned layouts.
            f"\\includegraphics[width=0.94\\textwidth]{{{tex_path}}}",
            f"\\caption{{{tex_inline(alt)}}}"]
    if label:
        body.append(f"\\label{{{label}}}")
    body.append(r"\end{figure*}")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# section conversion
# ---------------------------------------------------------------------------
HEAD_RE = re.compile(r"^(#{2,4})\s+(.*)$")
LIST_ITEM_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def block_to_comments(text):
    out = []
    for ln in text.splitlines():
        out.append("% " + ln if ln.strip() else "%")
    return "\n".join(out)


def block_trim_texts(blk):
    """Return the apply_trims-ready text strings for a non-special block.

    Prose blocks yield one single-space-joined string; numbered-list blocks
    yield one string per item (item text plus its continuation lines). This
    is the single source of truth shared by convert_section and the test
    suite (tests/test_md2tex.py), so the trims-integrity test can never
    drift from the converter's actual paragraph stream.
    """
    items, cur = [], None
    for l in blk:
        m = LIST_ITEM_RE.match(l)
        if m:
            cur = [m.group(2)]
            items.append(cur)
        elif cur is not None and l.strip():
            cur.append(l.strip())
    if len(items) >= 2:
        return [" ".join(it) for it in items]
    return [re.sub(r"\s*\n\s*", " ", "\n".join(blk).strip())]


def convert_section(body_lines, out_path, title, label):
    lines = body_lines
    tables = locate_tables(lines)
    out = [f"\\section{{{title}}}" + (f"\n\\label{{{label}}}" if label else "")]
    i, n = 0, len(lines)

    def comment_until_heading(start):
        j = start
        while j < n and not HEAD_RE.match(lines[j]):
            j += 1
        return j

    while i < n:
        if i in tables:
            j, rows, caption, label = tables[i]
            if caption:
                out.append(convert_table(rows, caption, label))
            else:
                out.append("% TABLE WITHOUT CAPTION - inspect manually")
            i = j
            continue

        ln = lines[i]
        m = HEAD_RE.match(ln)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 3:
                if re.match(r"^\d+\.\d+\s+Defensible claims", title):
                    j = comment_until_heading(i + 1)
                    out.append(f"% ==== DRAFTING METADATA (was: {title}) ====")
                    out.append(block_to_comments("\n".join(lines[i + 1:j])))
                    i = j
                    continue
                num = re.match(r"^(\d+\.\d+)\s+(.*)$", title)
                if num:
                    lab = SEC_LABELS.get(num.group(1), "")
                    out.append(f"\\subsection{{{tex_inline(num.group(2))}}}"
                               + (f"\n\\label{{{lab}}}" if lab else ""))
                elif title in ("Evaluation setup", "Statistical setup"):
                    out.append(f"\\subsection*{{{title}}}")
                else:
                    out.append(f"\\subsection{{{tex_inline(title)}}}")
            else:  # level 4 -> subsubsection
                t = re.sub(r"^[\d. ]+", "", title)
                out.append(f"\\subsubsection{{{tex_inline(t)}}}")
            i += 1
            continue

        if ln.strip().startswith("```"):
            j = i + 1
            fence = []
            while j < n and not lines[j].strip().startswith("```"):
                fence.append(lines[j])
                j += 1
            out.append(display_math(" ".join(x.strip() for x in fence).strip()))
            i = j + 1
            continue

        if ln.strip().startswith(">"):
            j = i
            blk = []
            while j < n and lines[j].strip().startswith(">"):
                blk.append(lines[j][1:].strip() if lines[j].strip() != ">" else "")
                j += 1
            out.append("% NOTE: " + " ".join(x for x in blk if x))
            i = j
            continue

        # paragraph: consecutive non-special lines
        j = i
        blk = []
        while j < n:
            l2 = lines[j]
            if (not l2.strip() or l2.strip() == "---" or HEAD_RE.match(l2)
                    or l2.strip().startswith("```") or l2.strip().startswith(">")
                    or is_table_row(l2)):
                break
            blk.append(l2.rstrip())
            j += 1
        if not blk:
            i += 1
            continue
        para = "\n".join(blk).strip()

        m_img = FIG_RE.match(para)
        if m_img and "\n" not in para:
            out.append(convert_figure(m_img.group(1), m_img.group(2), m_img.group(3)))
            i = j
            continue
        elif m_img:
            print(f"WARN: image line merged into a paragraph "
                  f"(add a blank line around it): {para[:60]}...")

        if para.startswith(("*Sources:", "*These conclusions",
                            "*Verification marks", "*FIRE 2026 submission")):
            out.append(block_to_comments(para))
            i = j
            continue

        if TABLE_CAP_RE.match(para):
            i = j
            continue  # caption consumed with its table

        texts = block_trim_texts(blk)
        if len(texts) > 1:
            out.append("\\begin{enumerate}")
            for t in texts:
                out.append("  \\item " + tex_inline(apply_trims(t)))
            out.append("\\end{enumerate}")
        else:
            out.append(tex_inline(apply_trims(texts[0])))
        i = j
        continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n\n".join(out) + "\n")
    print("wrote", out_path)


def display_math(formula):
    if formula.startswith("XAIR@K"):
        return (r"\begin{align}\mathrm{XAIR@K} &= (1-w)\cdot\mathrm{nDCG@K} \nonumber\\"
                r"&\quad + w\cdot\mathrm{mean}\bigl(\mathrm{xai\_conf}(d)"
                r" \text{ for } d \in \mathrm{top}\text{-}K,\ d\ \text{relevant}\bigr)\end{align}")
    if formula.startswith("MDS"):
        return (r"{\small \[ \mathrm{MDS} = \frac{|\text{unique memory slots accessed}"
                r"\text{ across all queries}|}{S}, \qquad S = 64 \]}")
    return "\\[ " + formula + " \\]"


# ---------------------------------------------------------------------------
# main.tex
# ---------------------------------------------------------------------------
def render_main(abstract_tex, front_comment, inputs):
    return f"""% ============================================================================
% FIRE 2026 Conference Track - camera-ready LaTeX skeleton
% ----------------------------------------------------------------------------
% Template: ACM ICPS "sigconf" (acmart), as required by the FIRE 2026 CFP
%   https://fire.irsi.org.in/fire/2026/call_for_papers
%   (template: https://authors.acm.org/proceedings/production-information/overleaf)
%
% CFP requirements to honour:
%   * maximum 9 pages of CONTENT (references excluded) - trim if over
%   * ACM CCS concepts and keywords are REQUIRED (below)
%   * do NOT use the "manuscript" option (that is single-column)
%   * restrict packages to the ACM TAPS whitelist
%   * alt text for figures/tables is strongly encouraged
%   * disclose any AI-assisted writing per ACM policy (see "Use of Generative AI")
%
% Compile (4 passes for correct refs/citations):
%   pdflatex main && bibtex main && pdflatex main && pdflatex main
% or: latexmk -pdf main.tex
% or (from this directory): tectonic main.tex
%
% Submissions:
%   * Regular / Perspective tracks -> double-blind: keep `anonymous=true`
%   * Resource / Demo track        -> single-blind: drop `anonymous`
% Camera-ready: real author block below; re-enable `anonymous=true` for
% double-blind review and comment the author block out again.
% ============================================================================

\\documentclass[sigconf,natbib=true]{{acmart}}
% Double-blind review: \\documentclass[sigconf,natbib=true,anonymous=true]{{acmart}}

\\usepackage{{booktabs}}   % publication-quality tables (ACM whitelist)
\\usepackage{{cleveref}}   % \\Cref/\\Crefrange for section/table references
% Note: amssymb deliberately NOT loaded - it clashes with acmart's fonts
\\emergencystretch=3em     % let narrow columns absorb long inline tokens
% tighten float spacing so wide table* floats leave smaller column gaps
\\setlength{{\\textfloatsep}}{{10pt plus 2pt minus 4pt}}
\\setlength{{\\floatsep}}{{8pt plus 2pt minus 2pt}}
\\setlength{{\\dbltextfloatsep}}{{10pt plus 2pt minus 4pt}}
\\setlength{{\\dblfloatsep}}{{8pt plus 2pt minus 2pt}}

% ----------------------------------------------------------------------------
\\title{{MEIRA: A Memory-Enhanced Interpretable Retrieval Agent for Multi-Turn
Agentic and Cross-Lingual Information Retrieval}}

% --- Author block --------------------------------------------------------------
\\author{{Arghya Bose}}
\\affiliation{{%
  \\institution{{KIIT (Deemed to be University)}}
  \\city{{Bhubaneswar, Odisha}}
  \\country{{India}}}}
\\email{{officialarghya29@gmail.com}}

\\author{{Arindam Tripathi}}
\\affiliation{{%
  \\institution{{KIIT (Deemed to be University)}}
  \\city{{Bhubaneswar, Odisha}}
  \\country{{India}}}}
\\email{{arindamtripathi.619@gmail.com}}

\\author{{Rajdeep Chatterjee}}
\\authornote{{Corresponding author}}
\\affiliation{{%
  \\institution{{KIIT (Deemed to be University)}}
  \\city{{Bhubaneswar, Odisha}}
  \\country{{India}}}}
\\email{{cse.rajdeep@gmail.com}}

% --- Conference / copyright metadata (camera-ready; replace DOI/ISBN with the
% publisher-assigned values before final submission) --------------------------
\\setcopyright{{acmcopyright}}
\\copyrightyear{{2026}}
\\acmYear{{2026}}
\\acmConference[FIRE '26]{{Proceedings of the 18th Forum for Information Retrieval Evaluation}}{{December 17--20, 2026}}{{Kolkata, India}}
\\acmBooktitle{{Proceedings of the 18th Forum for Information Retrieval Evaluation (FIRE '26), December 17--20, 2026, Kolkata, India}}
\\acmDOI{{10.1145/XXXXXXX.XXXXXXX}}
\\acmISBN{{979-8-4007-XXXX-X/26/12}}

% --- ACM CCS concepts (REQUIRED) -----------------------------------------------
% Verify against the ACM CCS 2012 thesaurus: https://dl.acm.org/ccs
\\begin{{CCSXML}}
<ccs2012>
<concept>
<concept_id>10002951.10003317</concept_id>
<concept_desc>Information systems~Information retrieval</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002951.10003317.10003335</concept_id>
<concept_desc>Information systems~Information retrieval~Retrieval models and ranking</concept_desc>
<concept_significance>300</concept_significance>
</concept>
<concept>
<concept_id>10002951.10003317.10003350</concept_id>
<concept_desc>Information systems~Information retrieval~Evaluation of retrieval results</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
\\end{{CCSXML}}
\\ccsdesc[500]{{Information systems~Information retrieval}}
\\ccsdesc[300]{{Information systems~Information retrieval~Retrieval models and ranking}}
\\ccsdesc[300]{{Information systems~Information retrieval~Evaluation of retrieval results}}

\\keywords{{agentic information retrieval; conversational search; episodic memory;
explainable IR; memory diversity; evaluation metrics}}

% --- Abstract -------------------------------------------------------------------
\\begin{{abstract}}
{abstract_tex}
\\end{{abstract}}

% Tight blurb variant (for call-for-papers / short abstracts) - commented out:
% Agentic multi-turn retrieval requires memory across turns and the ability to
% explain retrievals - capabilities absent from classical and neural baselines.
% We present MEIRA (Memory-Enhanced Interpretable Retrieval Agent), which
% couples a 64-slot episodic memory bank, a temporal-decay mechanism, and an
% XAI attribution head. We contribute two FIRE-style benchmarks
% (FIRE-AgentIR-2026, FIRE-CrossLingIR-2026) with sibling-topic hard negatives
% and label noise, and two metrics - XAIR@K and MDS. On both benchmarks across
% ten seeds, MEIRA-full beats the strongest baseline on every ranking-quality
% metric (F1, nDCG@10, MAP, MRR; F1 +0.087/+0.099) at p < 0.0001, robust to
% Holm and Bonferroni correction. Ablations rank memory > decay > XAI.

% ============================================================================
% DRAFTING NOTES - strip before submission (kept for the authors)
% ============================================================================
% Figures: three camera-ready figures (leaderboard bars, ablation-delta
% heatmap, ordering-stability rank plot) are generated at 300 dpi by the
% pipeline scripts (run_SOTA.py, run_ablation.py, compare_metric_orderings.py)
% into figures/, and embedded here from the markdown image syntax
% (![caption](path){{#label}}) - md2tex.py converts each into a figure* env.
% The markdown alt text doubles as the caption (ACM TAPS encourages
% descriptive alt text). Re-run the pipeline scripts after any data change.
%
% PAGE BUDGET: content currently fills the 9-page limit exactly; references
% are FORCED onto a fresh page (p10) by an explicit \\clearpage before the
% bibliography - this is intentional (references are excluded from the count)
% and keeps the last content page from overflowing. Any prose/figure
% addition must be re-checked with latex/_pagemap.py - add a _trims.py entry
% if it overflows.
%
{front_comment}

% ============================================================================
% ACM policy: AI-assisted writing must be disclosed. Adapt or delete.
% ============================================================================
% \\section{{Use of Generative AI}}
% \\paragraph{{Declaration of AI use.}} We used an LLM-assisted writing tool
% to draft and edit parts of the text; all technical claims were verified
% against the experimental pipeline before submission.

\\begin{{document}}

\\maketitle

{inputs}

% references start on a fresh page: the 9-page content limit excludes them,
% and a clean page break keeps the last content page from overflowing
\\clearpage
\\bibliographystyle{{ACM-Reference-Format}}
\\bibliography{{paper_references}}
% @@NOCITE@@   % replaced by an explicit \\nocite{{...}} of uncited keys

\\end{{document}}
"""


def split_sections(md):
    lines = md.splitlines()
    chunks, order, cur = {}, [], None
    for ln in lines:
        m = re.match(r"^##\s+(.*)$", ln)
        if m:
            cur = m.group(1).strip()
            order.append(cur)
            chunks[cur] = []
        elif cur is not None:
            chunks[cur].append(ln)
        else:
            chunks.setdefault("FRONT", []).append(ln)
    return chunks, order


def extract_abstract(chunks):
    """Parse the Abstract section into (primary_body, blurb_body, warnings).

    `primary_body` is the single-space-joined, de-hyphenated text of the
    "Primary variant (submission abstract)" paragraph - the text rendered in
    the compiled paper's \\begin{abstract} block. `blurb_body` is the
    optional short "Tight blurb" variant (kept for CfP submissions).

    Warnings are returned (not printed) so callers and tests can inspect
    them; md2tex.main() prints them.
    """
    warnings = []
    abs_paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(chunks.get("Abstract", [])))
                  if p.strip()]
    primary, blurb, mode = [], [], None
    for p in abs_paras:
        if p.startswith("**Primary variant"):
            mode = "primary"
        elif p.startswith("**Tight blurb"):
            mode = "blurb"
        elif p.startswith("**Word count"):
            mode = None
        elif mode == "primary":
            primary.append(p)
        elif mode == "blurb":
            blurb.append(p)
    if not primary:
        warnings.append("no '**Primary variant' abstract paragraph found - "
                        "the \\begin{abstract} block will be EMPTY. Check the "
                        "Abstract section of paper_full_draft.md.")
    # safety net: trims do not apply to the abstract - flag any overlap so a
    # future markdown edit cannot silently leave abstract/body text divergent
    for name, old, _ in TRIMS:
        if old in " ".join(primary):
            warnings.append(f"trim '{name}' also matches abstract text "
                            "(abstract bypasses trims - check consistency)")

    def _join(paras):
        body = " ".join(paras)
        body = re.sub(r"-\s*\n\s*", "-", body)   # de-hyphenate wrapped lines
        body = re.sub(r"\s*\n\s*", " ", body)
        return body

    return _join(primary), _join(blurb), warnings


def main():
    with open(MD_PATH) as f:
        md = f.read()
    chunks, order = split_sections(md)

    # ---- abstract ---------------------------------------------------------
    primary_body, blurb_body, abs_warnings = extract_abstract(chunks)
    for w in abs_warnings:
        print("WARN: " + w)
    abstract_tex = tex_inline(primary_body)

    # ---- front-matter drafting notes --------------------------------------
    front_comment = block_to_comments("\n".join(chunks.get("FRONT", [])))

    # ---- six sections -------------------------------------------------------
    sec_map = {
        "1. Introduction": ("sec01_introduction", "Introduction", "sec:intro"),
        "2. Related Work": ("sec02_related_work", "Related Work", "sec:rw"),
        "3. Datasets and Evaluation Protocol": ("sec03_datasets", "Datasets and Evaluation Protocol", "sec:datasets"),
        "4. Results: SOTA Leaderboard and Component Ablation": ("sec04_results", "Results", "sec:results"),
        "5. Robustness of the Statistical Comparisons": ("sec05_robustness", "Robustness of the Statistical Comparisons", "sec:rob"),
        "6. Conclusion and Limitations": ("sec06_conclusion", "Conclusion and Limitations", "sec:concl"),
    }
    names = [s for s in order if s in sec_map]
    skipped = {"Abstract", "References", "FRONT"}
    unmatched = [s for s in order if s not in sec_map and s not in skipped]
    for u in unmatched:
        print(f"WARN: section heading not converted: '{u}'")
    if len(names) != 6:
        print(f"WARN: expected 6 section files, produced {len(names)}")
    for name in names:
        fname, title, label = sec_map[name]
        convert_section(chunks[name], os.path.join(SECDIR, fname + ".tex"), title, label)

    inputs = "\n".join(r"\input{sections/" + sec_map[s][0] + "}" for s in names)
    with open(os.path.join(OUTDIR, "main.tex"), "w") as f:
        f.write(render_main(abstract_tex, front_comment, inputs))
    print("wrote", os.path.join(OUTDIR, "main.tex"))

    shutil.copy(BIB_SRC, os.path.join(OUTDIR, "paper_references.bib"))
    print("copied", os.path.join(OUTDIR, "paper_references.bib"))

    # ---- report --------------------------------------------------------------
    cited = set()
    for root, _, files in os.walk(SECDIR):
        for fn in files:
            txt = open(os.path.join(root, fn)).read()
            cited.update(re.findall(r"\\cite[pt]?\{([^}]+)\}", txt))
    missing = [k for k in cited if k not in BIB_KEYS]
    uncited = [k for k in BIB_KEYS if k not in cited]
    print("citation report: cited", len(cited), "| missing keys:", missing or "none",
          "| uncited bib keys:", uncited or "none")
    report_trims()
    # make \nocite explicit (a \nocite{*} would pull any future bib entries)
    main_path = os.path.join(OUTDIR, "main.tex")
    txt = open(main_path).read()
    if uncited:
        txt = txt.replace("% @@NOCITE@@",
                          f"\\nocite{{{','.join(sorted(uncited))}}}")
    else:
        txt = txt.replace("% @@NOCITE@@",
                          "% (all bibliography keys are cited in the text)")
    open(main_path, "w").write(txt)
    print("updated", main_path)


if __name__ == "__main__":
    main()
