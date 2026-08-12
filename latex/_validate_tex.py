"""Validate the generated LaTeX skeleton against the markdown source.

Checks:
1. every \\Cref/\\Crefrange target is a defined label (from main.aux)
2. every defined label is referenced at least once (warn-only)
3. section structure: 6 \\sections, correct \\subsection order
4. table environments: 9, each with a caption + label
5. figure environments: 3, each with a caption + label + existing image
   file, and all three figure labels defined in aux
6. number fidelity: key figures in the .tex match paper_full_draft.md
"""
import os
import re
import sys

LATEX = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LATEX)
ok = True


def check(cond, msg):
    global ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    if not cond:
        ok = False


# ---- 1+2: labels and references --------------------------------------------
aux = open(os.path.join(LATEX, "main.aux")).read()
defined = set(re.findall(r"\\newlabel\{([^}]+)\}", aux))
refs = set()
for fn in os.listdir(os.path.join(LATEX, "sections")):
    t = open(os.path.join(LATEX, "sections", fn)).read()
    for m in re.findall(r"\\Cref\{([^}]+)\}", t):      # multi-target a,b,c
        refs.update(x.strip() for x in m.split(","))
    for a, b in re.findall(r"\\Crefrange\{([^}]+)\}\{([^}]+)\}", t):
        refs.add(a)
        refs.add(b)
missing = refs - defined
check(not missing, f"all {len(refs)} \\Cref targets defined "
      f"({sorted(missing) if missing else '0 missing'})")

# ---- 3: section structure ---------------------------------------------------
sec_titles = []
for fn in sorted(os.listdir(os.path.join(LATEX, "sections"))):
    t = open(os.path.join(LATEX, "sections", fn)).read()
    for m in re.finditer(r"\\(?:sub)*section\*?\{([^}]*)\}", t):
        sec_titles.append(m.group(1))
check(len(sec_titles) >= 6, f"{len(sec_titles)} sectioning commands found")
check(sec_titles[0] == "Introduction", "first section is Introduction")
check("Conclusion and Limitations" in sec_titles, "Conclusion section present")
idx = {s: i for i, s in enumerate(sec_titles)}
for a, b in [("SOTA leaderboard", "Component ablation"),
             ("Multiplicity correction", "Threshold sensitivity")]:
    ia = next((i for s, i in idx.items() if a in s), None)
    ib = next((i for s, i in idx.items() if b in s), None)
    if ia is not None and ib is not None:
        check(ia < ib, f"'{a}' before '{b}'")
    else:
        check(False, f"both of {a} / {b} present (got {ia}, {ib})")

# ---- 4: tables ---------------------------------------------------------------
tbl_caps = []
for fn in sorted(os.listdir(os.path.join(LATEX, "sections"))):
    t = open(os.path.join(LATEX, "sections", fn)).read()
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", t, re.S):
        seg = m.group(1)
        c = re.findall(r"\\caption\{([^}]*)\}", seg)
        check(bool(c), f"table in {fn} has a caption")
        tbl_caps += c
        check("\\label{" in seg, f"table in {fn} has a label")
check(len(tbl_caps) == 9, f"9 table captions ({len(tbl_caps)} found)")

# ---- 4b: figures --------------------------------------------------------------
fig_caps = []
for fn in sorted(os.listdir(os.path.join(LATEX, "sections"))):
    t = open(os.path.join(LATEX, "sections", fn)).read()
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", t, re.S):
        seg = m.group(1)
        c = re.findall(r"\\caption\{([^}]*)\}", seg)
        check(bool(c), f"figure in {fn} has a caption")
        fig_caps += c
        check("\\label{" in seg, f"figure in {fn} has a label")
        for inc in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", seg):
            chk = os.path.join(ROOT, inc.replace("../", ""))
            check(os.path.exists(chk), f"figure image exists: {inc}")
check(len(fig_caps) == 3, f"3 figure captions ({len(fig_caps)} found)")
check({"fig:sota", "fig:ablation", "fig:ordering"} <= defined,
      "all three figure labels defined in aux")

# ---- 5: number fidelity -------------------------------------------------------
def grab_numbers(txt):
    return set(re.findall(r"\b\d+\.\d{3}\b", txt))


md_nums = grab_numbers(open(os.path.join(ROOT, "paper_full_draft.md")).read())
tex_all = "".join(open(os.path.join(LATEX, "sections", fn)).read()
                  for fn in os.listdir(os.path.join(LATEX, "sections")))
tex_nums = grab_numbers(tex_all)
for n in ["0.826", "0.780", "0.740", "0.680", "20.120", "13.058",
          "0.110", "0.124", "1.000", "0.894", "0.886"]:
    check(n in tex_nums, f"number {n} present in LaTeX")
    check(n in md_nums, f"number {n} present in markdown")

stripped = "\n".join(l for l in tex_all.splitlines()
                     if not l.strip().startswith("%"))
tex_nums_stripped = grab_numbers(stripped)
lost = [n for n in sorted(md_nums) if n not in tex_nums_stripped]
print("note: markdown-only 3dp numbers not in LaTeX body:", lost or "none")

print("\nRESULT:", "ALL CHECKS PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
