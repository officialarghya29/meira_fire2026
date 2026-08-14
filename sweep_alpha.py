"""
sweep_alpha.py – FIRE 2026 MEIRA
================================
Alpha-threshold sensitivity sweep for the pairwise significance analysis.

Loads the archived per-metric significance matrices (raw + Holm + Bonferroni
p-values) produced by run_significance.py and re-evaluates every verdict at
several significance thresholds (default α ∈ {0.01, 0.05, 0.10}):

  - significant-pair counts (of 28) per α × correction × metric × dataset;
  - "lost" pairs per α — Holm-significant but not Bonferroni-significant;
  - α-sensitive pairs — pairs whose Holm/Bonferroni verdict flips as the
    threshold moves (the comparisons reviewers will probe).

No models are re-run: p-values are α-independent, only the verdicts change.
At α = 0.05 the output matches `compare_corrections.py` by construction
(a runtime check compares the α=0.05 counts with correction_comparison.json).

Produces (into the config-named folder):
  results/k10_s10/alpha_sweep.json – source of truth
  results/k10_s10/alpha_sweep.md   – paper-ready sensitivity tables
  figures/k10_s10/sweep1_alpha_counts.png – significant-count curves per metric × dataset

Usage:
  python sweep_alpha.py --k 10 --seeds 10 [--alphas 0.01 0.05 0.10]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = None   # set in main() → results/<tag>/
FIG_DIR = None   # set in main() → figures/<tag>/

KEY_METRICS = ["F1", "nDCG@10", "MAP", "MRR"]
DEFAULT_ALPHAS = [0.01, 0.05, 0.10]

PALETTE = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
           "neutral":"#6B7280","bg":"#F8FAFC","grid":"#E5E7EB",
           "green":"#27AE60","orange":"#E67E22"}

plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
    "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
    "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
    "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
    "font.family":"DejaVu Sans","axes.titlesize":11,"axes.labelsize":9})


def make_output_dirs(tag: str):
    """Create and return config-named output dirs, e.g. results/k10_s10/."""
    fig = os.path.join(BASE, "figures", tag)
    res = os.path.join(BASE, "results", tag)
    os.makedirs(fig, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    return fig, res


def load_matrix(k: int, seeds: int, metric: str, correction: str) -> dict:
    """Load a run_significance.py matrix for the given metric × correction."""
    path = os.path.join(BASE, "results", f"k{k}_s{seeds}",
                        f"significance_matrix_{metric}_{correction}.json")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found.\n"
                 f"        Run `python run_significance.py --k {k} --seeds {seeds} "
                 f"--metric {metric} --correction {correction}` first.")
    with open(path) as f:
        return json.load(f)


def fmt_p(p):
    """Format a p-value for markdown (no star markers)."""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def main(args):
    global RES_DIR, FIG_DIR
    tag = f"k{args.k}_s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)
    alphas = sorted(set(args.alphas))       # dedupe 0.1 vs 0.10 etc.
    for a in alphas:
        if not 0.0 < a < 1.0:
            sys.exit(f"[ERROR] alpha must be in (0, 1), got {a}.")
    akeys = [f"{a:.2f}" for a in alphas]   # stable JSON keys

    print("\n" + "=" * 60)
    print("  Alpha-Threshold Sensitivity Sweep")
    print("  (Holm & Bonferroni verdicts at α = " + ", ".join(akeys) + ")")
    print("=" * 60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    models = None
    per_metric = {}      # metric → ds → {"pairs": [rec...]}  (rec holds raw/holm/bonf/t)
    for met in args.metrics:
        holm = load_matrix(args.k, args.seeds, met, "holm")
        bonf = load_matrix(args.k, args.seeds, met, "bonferroni")
        if models is None:
            models = holm["config"]["models"]
        m = holm["config"]["models"]
        n = len(m)
        ds_data = {}
        for ds in holm["datasets"]:
            Pr = np.array(holm["datasets"][ds]["p_matrix_raw"])
            Pbr = np.array(bonf["datasets"][ds]["p_matrix_raw"])
            if not np.allclose(Pr, Pbr):
                print(f"    [WARNING] {met} | {ds}: raw p matrices differ "
                      "between the holm/bonferroni files")
            Ph = np.array(holm["datasets"][ds]["p_matrix"])
            Pb = np.array(bonf["datasets"][ds]["p_matrix"])
            T = np.array(holm["datasets"][ds]["t_matrix"])
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    t = float(T[i, j])
                    hi, lo = (m[i], m[j]) if t > 0 else (m[j], m[i])
                    pairs.append({"higher": hi, "lower": lo,
                                  "raw": float(Pr[i, j]), "holm": float(Ph[i, j]),
                                  "bonferroni": float(Pb[i, j]), "t": abs(t)})
            ds_data[ds] = pairs
        per_metric[met] = ds_data

    # ---- per-α verdicts ----
    counts = {}        # akey → metric → ds → {raw, holm, bonferroni, lost}
    lost_by_a = {}     # akey → metric → ds → [pair records]
    for a in alphas:
        ak = f"{a:.2f}"
        counts[ak] = {}
        lost_by_a[ak] = {}
        for met in args.metrics:
            counts[ak][met] = {}
            lost_by_a[ak][met] = {}
            for ds, pairs in per_metric[met].items():
                n_pairs = len(pairs)
                c = {"raw": 0, "holm": 0, "bonferroni": 0, "lost": 0}
                lost = []
                for p in pairs:
                    c["raw"] += int(p["raw"] < a)
                    c["holm"] += int(p["holm"] < a)
                    s_bonf = p["bonferroni"] < a
                    c["bonferroni"] += int(s_bonf)
                    if p["holm"] < a and not s_bonf:
                        c["lost"] += 1
                        lost.append(p)
                counts[ak][met][ds] = c
                lost_by_a[ak][met][ds] = lost
                print(f"    [α={ak} | {met} | {ds}] raw {c['raw']}/{n_pairs} · "
                      f"Holm {c['holm']}/{n_pairs} · Bonferroni {c['bonferroni']}/{n_pairs} "
                      f"· lost {c['lost']}")

    # ---- α-sensitive pairs (verdict flips or lost at any α) ----
    sensitive = []
    for met in args.metrics:
        for ds, pairs in per_metric[met].items():
            for p in pairs:
                h_sig = {ak: False for ak in akeys}
                b_sig = {ak: False for ak in akeys}
                for a, ak in zip(alphas, akeys):
                    h_sig[ak] = p["holm"] < a
                    b_sig[ak] = p["bonferroni"] < a
                flips_holm = len(set(h_sig.values())) > 1
                flips_bonf = len(set(b_sig.values())) > 1
                lost_any = any(h_sig[ak] and not b_sig[ak] for ak in akeys)
                if flips_holm or flips_bonf or lost_any:
                    sensitive.append({"metric": met, "dataset": ds,
                                      "higher": p["higher"], "lower": p["lower"],
                                      "raw": p["raw"], "holm": p["holm"],
                                      "bonferroni": p["bonferroni"], "t": p["t"],
                                      "holm_sig": h_sig, "bonf_sig": b_sig,
                                      "flips_holm": flips_holm,
                                      "flips_bonf": flips_bonf,
                                      "lost_at": [ak for ak in akeys
                                                  if h_sig[ak] and not b_sig[ak]]})
    sensitive.sort(key=lambda s: (s["metric"], s["dataset"], s["raw"]))

    # ---- runtime sanity check vs compare_corrections.py at α=0.05 ----
    if "0.05" in akeys:
        cc_path = os.path.join(RES_DIR, "correction_comparison.json")
        if os.path.exists(cc_path):
            with open(cc_path) as f:
                cc = json.load(f)
            for met in args.metrics:
                for ds in cc["summary"][met]:
                    mine = counts["0.05"][met][ds]
                    theirs = cc["summary"][met][ds]
                    if (mine["raw"], mine["holm"], mine["bonferroni"], mine["lost"]) != \
                       (theirs["raw"], theirs["holm"], theirs["bonferroni"],
                        len(cc["lost_pairs"][met][ds])):
                        print(f"    [WARNING] α=0.05 {met} {ds}: sweep counts {mine} "
                              f"differ from compare_corrections {theirs}")

    # ---------------- JSON (source of truth) ----------------
    out = {"config": {"k": args.k, "seeds": list(range(42, 42 + args.seeds)),
                      "models": models, "metrics": args.metrics,
                      "alphas": alphas,
                      "note": "Lost = significant under Holm but not Bonferroni "
                              "at that α; verdicts are α-dependent, p-values are not."},
           "counts": counts,
           "lost_pairs": lost_by_a,
           "sensitive_pairs": sensitive}
    json_path = os.path.join(RES_DIR, "alpha_sweep.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"  ✓ {os.path.basename(json_path)}")

    # ---------------- Markdown (paper-ready) ----------------
    lines = ["# Alpha-Threshold Sensitivity Sweep — FIRE 2026 MEIRA",
             "",
             "> Source: `results/k10_s10/significance_matrix_{metric}_{holm|bonferroni}.json` ",
             "> (paired t-tests, family of all 28 pairwise tests per dataset × metric). "
             "p-values are α-independent; only the verdicts change with the threshold. "
             "Simulated evaluation-harness data — see `model_sim.py`.",
             "",
             f"Alphas swept: **{', '.join(akeys)}**. A pair is **lost** when it is "
             "significant under Holm-Bonferroni but not under the stricter "
             "Bonferroni at that α.",
             "",
             "## 1. Significant pairs at α (of 28)",
             "",
             "| α | Metric | Dataset | raw | Holm | Bonferroni | lost |",
             "|---|---|---|---|---|---|---|"]
    for ak in akeys:
        for met in args.metrics:
            for ds, c in counts[ak][met].items():
                lines.append(f"| {ak} | {met} | {ds} | {c['raw']}/28 | {c['holm']}/28 | "
                             f"{c['bonferroni']}/28 | {c['lost']} |")
    lines.append("")
    lines += ["## 2. Lost pairs per α (Holm-significant, not Bonferroni)",
              "",
              "| α | Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |",
              "|---|---|---|---|---|---|---|---|"]
    any_lost = False
    for ak in akeys:
        for met in args.metrics:
            for ds, lost in lost_by_a[ak][met].items():
                for p in lost:
                    any_lost = True
                    lines.append(f"| {ak} | {met} | {ds} | {p['higher']} > {p['lower']} | "
                                 f"{fmt_p(p['raw'])} | {fmt_p(p['holm'])} | "
                                 f"{fmt_p(p['bonferroni'])} | {p['t']:.2f} |")
    if not any_lost:
        lines.append("| — | — | — | *(none)* | — | — | — | — |")
    lines.append("")
    lines += ["## 3. α-sensitive pairs (verdict flips or lost at any α)",
              "",
              "✓ = significant, ✗ = not significant at that α.",
              "",
              "| Metric | Dataset | Pair | raw p | Holm p | Bonf p | "
              "Holm @0.01 | @0.05 | @0.10 | Bonf @0.01 | @0.05 | @0.10 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sensitive:
        cell = lambda d, ak: "✓" if d[ak] else "✗"
        lines.append(f"| {s['metric']} | {s['dataset']} | {s['higher']} > {s['lower']} | "
                     f"{fmt_p(s['raw'])} | {fmt_p(s['holm'])} | {fmt_p(s['bonferroni'])} | "
                     + " | ".join(cell(s["holm_sig"], ak) for ak in akeys) + " | "
                     + " | ".join(cell(s["bonf_sig"], ak) for ak in akeys) + " |")
    lines.append("")
    md_path = os.path.join(RES_DIR, "alpha_sweep.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(md_path)}")

    # ---------------- Figure: counts vs α per metric × dataset ----------------
    n_met = len(args.metrics)
    n_ds = len(list(counts[akeys[0]][args.metrics[0]].keys()))
    x = np.arange(len(alphas))
    fig, axes = plt.subplots(n_met, n_ds, figsize=(4.6 * n_ds, 3.6 * n_met),
                             squeeze=False)
    fig.suptitle(f"Significant pairs (of 28) vs significance threshold ({tag})",
                 fontsize=12, color=PALETTE["primary"], fontweight="bold")
    for ri, met in enumerate(args.metrics):
        for ci, ds in enumerate(counts[akeys[0]][met].keys()):
            ax = axes[ri, ci]
            for corr, color, ls in [("raw", PALETTE["neutral"], "--"),
                                    ("holm", PALETTE["secondary"], "-"),
                                    ("bonferroni", PALETTE["accent"], "-")]:
                ys = [counts[ak][met][ds][corr] for ak in akeys]
                ax.plot(x, ys, marker="o", ms=5, lw=2.2, ls=ls, color=color,
                        label=corr, zorder=3)
            if "0.05" in akeys:
                ax.axvline(x[akeys.index("0.05")], color=PALETTE["orange"],
                           lw=1.2, ls=":", zorder=1)
                ax.text(x[akeys.index("0.05")], 1, "α=0.05", fontsize=7,
                        color=PALETTE["orange"], ha="center")
            ax.set_xticks(x)
            ax.set_xticklabels(akeys)
            ax.set_ylim(0, 28.5)
            ax.set_yticks([0, 7, 14, 21, 28])
            ax.grid(True, axis="y")
            if ci == 0:
                ax.set_ylabel(f"{met}\nsig. pairs", color=PALETTE["primary"],
                              fontweight="bold")
            ax.set_title(ds if ri == 0 else "", color=PALETTE["primary"],
                         fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=3, frameon=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    png_path = os.path.join(FIG_DIR, "sweep1_alpha_counts.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(png_path)}")

    print("\n" + "=" * 70)
    print("  SUMMARY — α-sensitive pairs (verdict changes or lost at some α)")
    print("=" * 70)
    if not sensitive:
        print("  (none — every verdict is stable across all swept α levels)")
    for s in sensitive:
        mark = "lost@" + ",".join(s["lost_at"]) if s["lost_at"] else "flips"
        print(f"  [{s['metric']} | {s['dataset']}] {s['higher']} > {s['lower']} "
              f"(raw={s['raw']:.4f} holm={s['holm']:.4f} bonf={s['bonferroni']:.4f}) "
              f"→ {mark}")
    print(f"\n  Results → {json_path}, {md_path}")
    print("  Alpha sensitivity sweep complete ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="Folds of the source run (folder tag)")
    p.add_argument("--seeds", type=int, default=10, help="Seeds of the source run (folder tag)")
    p.add_argument("--metrics", nargs="+", default=KEY_METRICS, choices=KEY_METRICS,
                   help="Metrics to sweep (default: all four)")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                   help="Significance thresholds to sweep (default: 0.01 0.05 0.10)")
    main(p.parse_args())
