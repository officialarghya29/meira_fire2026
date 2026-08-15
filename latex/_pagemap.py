"""Report the page-break map: section/table label -> PDF page, plus per-page
text-line fill (to spot float-gap whitespace)."""
import os
import re
import sys

LATEX = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LATEX)

# main_v2.tex is compiled from the repo ROOT, so the aux lands there.
aux = open(os.path.join(ROOT, "main_v2.aux")).read()
# \newlabel{name}{{refnum}{page}}...
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux):
    labels[m.group(1)] = (m.group(2), m.group(3))

order = [
    ("sec:intro", "1 Introduction"),
    ("sec:intro-motivation", "  1.1 Motivation"),
    ("sec:intro-contributions", "  1.4 Contributions"),
    ("sec:intro-findings", "  1.5 Findings"),
    ("sec:intro-roadmap", "  1.6 Roadmap"),
    ("sec:rw", "2 Related Work"),
    ("sec:rw-positioning", "  2.6 Positioning"),
    ("sec:datasets", "3 Datasets & Protocol"),
    ("sec:datasets-benchmarks", "  3.1 Benchmarks"),
    ("sec:datasets-protocol", "  3.2 Protocol"),
    ("sec:datasets-metrics", "  3.3 Metrics"),
    ("sec:datasets-models", "  3.4 Models"),
    ("sec:datasets-stats", "  3.5 Statistics"),
    ("sec:results", "4 Results"),
    ("sec:results-sota", "  4.1 SOTA leaderboard"),
    ("sec:results-ablation", "  4.2 Ablation"),
    ("sec:rob", "5 Robustness"),
    ("sec:rob-corr", "  5.1 Correction"),
    ("sec:rob-alpha", "  5.2 Threshold sweep"),
    ("sec:concl", "6 Conclusion"),
    ("sec:concl-limitations", "  6.2 Limitations"),
    ("tab:datasets", "Table 1"),
    ("tab:models", "Table 2"),
    ("tab:sota-agent", "Table 3"),
    ("tab:sota-cross", "Table 4"),
    ("tab:abl-agent", "Table 5"),
    ("tab:abl-cross", "Table 6"),
    ("tab:deltas", "Table 7"),
    ("tab:corr-counts", "Table 8"),
    ("tab:lost-pairs", "Table 9"),
]
print("=== label -> page (from main.aux) ===")
for lab, name in order:
    if lab in labels:
        num, pg = labels[lab]
        print(f"  {name:<22} {lab:<26} page {pg}  (ref {num})")
    else:
        print(f"  {name:<22} {lab:<26} MISSING")

# per-page fill from pdftotext output (arg or default)
txt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/paper.txt"
if os.path.exists(txt_path):
    pages = [p for p in open(txt_path).read().split("\f") if p.strip()]
    print(f"\n=== per-page fill ({len(pages)} real pages) ===")
    for i, pg in enumerate(pages, 1):
        lines = [l for l in pg.splitlines() if l.strip()]
        print(f"  page {i}: {len(lines):>3} text lines")
