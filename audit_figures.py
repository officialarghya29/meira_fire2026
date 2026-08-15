"""audit_figures.py - collision/clipping audit for the three paper figures.

Regenerates the three figures embedded in the paper (Fig 1 leaderboard bars,
Fig 2 ablation-delta heatmap, Fig 3 ordering-stability) through their real
generating functions, instruments matplotlib's Text objects, and measures:

  1. pairwise text-bbox intersections (text_text)
  2. text clipped at the canvas edge (clip)
  3. panel-title horizontal overflow beyond its own axes (title_overflow)
  4. data text extending beyond its axes (beyond_axes; tick/axis labels and
     legend texts are intentional and excluded)
  5. text drawn on top of bar patches (text_bar; multi-segment error-bar
     lines are excluded - their bounding box spans every bar)
  6. data text overlapping data lines in the same axes (text_line;
     e.g. a caption drawn through a polyline - only labeled Line2D
     artists are considered, so gridlines/errorbar caps are skipped)

Measurement happens on the normal canvas BEFORE the tight-bbox savefig re-
renders at a different size, and text records are reset per figure so stale
labels from one figure never pollute the next.

Usage (from the repo root):
    .venv/bin/python audit_figures.py

Exit code 0 = all figures clean; 1 = at least one issue found (so it can be
wired into CI / pre-commit).

NOTE: the gridspec margins in run_SOTA.py / compare_metric_orderings.py are
tuned to the current data and font sizes (y-tick label widths, 90-degree
x-label overhang, star headroom). Re-run this audit after ANY data, model,
seed, or figure change.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.text
from matplotlib.path import Path

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

TEXTS = []
CURRENT = {"name": ""}

_orig_init = matplotlib.text.Text.__init__
def spy_init(self, *a, **k):
    _orig_init(self, *a, **k)
    TEXTS.append(self)
matplotlib.text.Text.__init__ = spy_init


def _role_of(t, fig):
    """Human-readable location for a text object."""
    ax = getattr(t, "axes", None)
    if ax is not None and ax in fig.axes:
        aid = fig.axes.index(ax)
        if t is ax.title:
            return f"ax{aid} title"
        if t is ax.xaxis.label:
            return f"ax{aid} xlabel"
        if t is ax.yaxis.label:
            return f"ax{aid} ylabel"
        try:
            if t in set(ax.get_xticklabels()) | set(ax.get_yticklabels()):
                return f"ax{aid} tick"
        except Exception:
            pass
        leg = ax.get_legend()
        if leg is not None:
            try:
                if t in set(leg.get_texts()):
                    return f"ax{aid} legend"
            except Exception:
                pass
        return f"ax{aid} data"
    return "figure legend"


def audit(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    issues = {"text_text": [], "clip": [], "title_overflow": [],
              "beyond_axes": [], "text_bar": [], "text_line": []}

    # texts whose position outside the axes is intentional
    intentional = set()
    for ax in fig.axes:
        intentional.add(id(ax.title))
        intentional.add(id(ax.xaxis.label))
        intentional.add(id(ax.yaxis.label))
        try:
            intentional |= {id(t) for t in ax.get_xticklabels()}
            intentional |= {id(t) for t in ax.get_yticklabels()}
        except Exception:
            pass
        leg = ax.get_legend()
        if leg is not None:
            try:
                intentional |= {id(t) for t in leg.get_texts()}
            except Exception:
                pass

    boxes = [(t, t.get_window_extent(r)) for t in TEXTS if t.get_text() != ""]

    for a in range(len(boxes)):
        ta, ba = boxes[a]
        for b in range(a + 1, len(boxes)):
            tb, bb = boxes[b]
            ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if ix > 0.5 and iy > 0.5 and ix * iy > 4:
                issues["text_text"].append(
                    (repr(ta.get_text())[:22], _role_of(ta, fig),
                     repr(tb.get_text())[:22], _role_of(tb, fig),
                     round(ix, 1), round(iy, 1)))

    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            issues["clip"].append((repr(t.get_text())[:30], _role_of(t, fig),
                                   tuple(round(v) for v in (b.x0, b.y0, b.x1, b.y1))))

    for ax in fig.axes:
        tb = ax.title.get_window_extent(r)
        ab = ax.get_window_extent(r)
        over = max(tb.x1 - ab.x1, ab.x0 - tb.x0)
        if over > 0:
            issues["title_overflow"].append((repr(ax.title.get_text())[:36],
                                             round(over, 1)))

    for t, b in boxes:
        ax = getattr(t, "axes", None)
        if ax is None or id(t) in intentional:
            continue
        ab = ax.get_window_extent(r)
        over = max(b.x1 - ab.x1, ab.x0 - b.x0, b.y1 - ab.y1, ab.y0 - b.y0)
        if over > 2:
            issues["beyond_axes"].append((repr(t.get_text())[:30],
                                          round(over, 1)))

    for ax in fig.axes:
        for t, tb in boxes:
            if getattr(t, "axes", None) is not ax:
                continue
            for art in ax.patches:
                try:
                    ab = art.get_window_extent(r)
                except Exception:
                    continue
                ix = min(tb.x1, ab.x1) - max(tb.x0, ab.x0)
                iy = min(tb.y1, ab.y1) - max(tb.y0, ab.y0)
                if ix > 0.5 and iy > 0.5 and ix * iy > 9:
                    issues["text_bar"].append((repr(t.get_text())[:24],
                                               round(ix * iy, 1)))

    # data text overlapping data lines (labeled Line2D only: excludes
    # gridlines and errorbar cap segments)
    for ax in fig.axes:
        lines = [ln for ln in ax.lines
                 if isinstance(ln, matplotlib.lines.Line2D)
                 and ln.get_label() and not ln.get_label().startswith("_")]
        if not lines:
            continue
        for t, tb in boxes:
            if getattr(t, "axes", None) is not ax or id(t) in intentional:
                continue
            for ln in lines:
                try:
                    pts = ax.transData.transform(
                        np.column_stack([ln.get_xdata(), ln.get_ydata()]))
                except Exception:
                    continue
                for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                    seg = Path([(x0, y0), (x1, y1)])
                    if seg.intersects_bbox(tb, filled=False):
                        issues["text_line"].append(
                            (repr(t.get_text())[:24], _role_of(t, fig),
                             repr(ln.get_label())[:18],
                             round(tb.height, 1)))
                        break

    print(f"\n===== {name} =====  (canvas {W}x{H}, non-empty texts {len(boxes)})")
    n_bad = 0
    for k, v in issues.items():
        print(f"  {k}: {len(v)}")
        for row in v[:24]:
            print("    ", " | ".join(map(str, row)))
        n_bad += len(v)
    return n_bad


_orig_sf = Figure.savefig
def spy_sf(self, *a, **k):
    n = audit(self, CURRENT["name"])   # measure BEFORE tight-bbox re-render
    if n:
        CURRENT["bad"] = True
    return _orig_sf(self, *a, **k)
Figure.savefig = spy_sf


def run(name, fn):
    del TEXTS[:]
    CURRENT["name"] = name
    CURRENT["bad"] = False
    fn()
    return CURRENT["bad"]


def main():
    import run_SOTA, run_ablation, compare_metric_orderings as cmo
    os.makedirs("/tmp/audit_fig", exist_ok=True)
    run_SOTA.FIG_DIR = "/tmp/audit_fig"
    run_ablation.FIG_DIR = "/tmp/audit_fig"
    cmo.FIG_DIR = "/tmp/audit_fig"

    bad = False
    bad |= run("Fig1 sota1_leaderboard_bars", lambda: run_SOTA.fig_leaderboard_bars(
        json.load(open(os.path.join(BASE, "results/s10/sota.json")))["results"]))
    bad |= run("Fig2 abl2_delta_heatmap", lambda: run_ablation.fig_delta_heatmap(
        json.load(open(os.path.join(BASE, "results/s10/ablation.json")))["deltas"]))
    data = json.load(open(os.path.join(BASE, "results/k10_s10/"
                                       "metric_ordering_stability_holm.json")))
    bad |= run("Fig3 ord1_ordering_stability_holm",
               lambda: cmo.fig_ordering_stability(
                   data["datasets"], data["config"]["models"], "audit",
                   method="holm"))

    print("\n" + "=" * 60)
    if bad:
        print("RESULT: ISSUES FOUND - inspect the entries above")
        sys.exit(1)
    print("RESULT: ALL FIGURES CLEAN (0 collisions, 0 clipping, 0 overflow)")
    sys.exit(0)


if __name__ == "__main__":
    main()
