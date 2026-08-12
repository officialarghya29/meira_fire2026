"""
compare_corrections.py – FIRE 2026 MEIRA
========================================
Holm vs Bonferroni comparison for the paper's robustness appendix.

Loads the per-metric significance matrices produced by run_significance.py
under BOTH correction methods (default: results/k10_s10/
significance_matrix_{metric}_{holm|bonferroni}.json) and reports, per
dataset × metric:

  - how many of the n(n−1)/2 = 28 pairwise tests are significant at
    α=0.05 under raw / Holm / Bonferroni;
  - the "lost" pairs — significant under Holm but no longer under the
    stricter Bonferroni — with raw / Holm / Bonferroni p-values;
  - a full per-pair appendix table (raw, Holm, Bonferroni p + t).

The raw p-values are read from the p_matrix_raw stored by both variants,
so the two families are guaranteed to share the same underlying tests
(a runtime check asserts the raw matrices agree).

Produces (into the config-named folder):
  results/k10_s10/correction_comparison.json – source of truth
  results/k10_s10/correction_comparison.md   – paper-ready appendix tables
  figures/k10_s10/corr1_correction_comparison.png – status grids per metric × dataset

Usage:
  python compare_corrections.py --k 10 --seeds 10 [--metrics F1 nDCG@10 MAP MRR]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = None   # set in main() → results/<tag>/
FIG_DIR = None   # set in main() → figures/<tag>/

KEY_METRICS = ["F1", "nDCG@10", "MAP", "MRR"]
ALPHA = 0.05

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
    """Format a p-value for markdown (stars only, no extra markers)."""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def main(args):
    global RES_DIR, FIG_DIR
    tag = f"k{args.k}_s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)   # returns (fig, res)

    print("\n" + "=" * 60)
    print("  Holm vs Bonferroni — Correction-Robustness Comparison")
    print("  (appendix tables for the paper's robustness section)")
    print("=" * 60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    models = None
    per_metric = {}      # metric → dataset → dict(pairs, lost, counts)
    status_grids = {}    # metric → dataset → 8×8 status int matrix
    raw_mismatch_any = False

    for met in args.metrics:
        holm = load_matrix(args.k, args.seeds, met, "holm")
        bonf = load_matrix(args.k, args.seeds, met, "bonferroni")
        if models is None:
            models = holm["config"]["models"]
        m = holm["config"]["models"]
        if bonf["config"]["models"] != m:
            sys.exit(f"[ERROR] model lists differ between holm/bonferroni files "
                     f"for metric {met}.")
        n = len(m)
        n_pairs = n * (n - 1) // 2
        ds_data = {}
        status = {}
        for ds in holm["datasets"]:
            Pr = np.array(holm["datasets"][ds]["p_matrix_raw"])
            Pbr = np.array(bonf["datasets"][ds]["p_matrix_raw"])
            if not np.allclose(Pr, Pbr):
                raw_mismatch_any = True
            Ph = np.array(holm["datasets"][ds]["p_matrix"])
            Pb = np.array(bonf["datasets"][ds]["p_matrix"])
            T = np.array(holm["datasets"][ds]["t_matrix"])
            pairs = []
            S = np.zeros((n, n), dtype=int)   # 0 diag, 1 neither, 2 lost, 3 both
            for i in range(n):
                for j in range(i + 1, n):
                    t = float(T[i, j])
                    hi, lo = (m[i], m[j]) if t > 0 else (m[j], m[i])
                    # t is stored with the matrix convention (models[i] minus
                    # models[j]); report its magnitude since the pair name
                    # already encodes the direction (higher > lower).
                    rec = {"higher": hi, "lower": lo,
                           "raw": float(Pr[i, j]), "holm": float(Ph[i, j]),
                           "bonferroni": float(Pb[i, j]), "t": abs(t),
                           "sig_raw": bool(Pr[i, j] < ALPHA),
                           "sig_holm": bool(Ph[i, j] < ALPHA),
                           "sig_bonferroni": bool(Pb[i, j] < ALPHA)}
                    pairs.append(rec)
                    s_holm, s_bonf = rec["sig_holm"], rec["sig_bonferroni"]
                    S[i, j] = S[j, i] = 3 if (s_holm and s_bonf) else \
                                       (2 if s_holm else (1 if s_bonf else 0))
            lost = [p for p in pairs if p["sig_holm"] and not p["sig_bonferroni"]]
            # a pair that is Bonferroni-significant is always Holm-significant,
            # but flag it defensively if the invariant ever breaks
            unexpected = [p for p in pairs if p["sig_bonferroni"] and not p["sig_holm"]]
            if unexpected:
                print(f"    [WARNING] {met} | {ds}: {len(unexpected)} pair(s) "
                      f"Bonferroni-significant but NOT Holm-significant "
                      f"(invariant violation): "
                      + ", ".join(f"{u['higher']}>{u['lower']}" for u in unexpected))
            # Holm is always at least as powerful as Bonferroni (holm ≤ bonf
            # pointwise); break loudly if the correction math ever disagrees
            if np.any(Ph > Pb + 1e-9):
                print(f"    [WARNING] {met} | {ds}: Holm p > Bonferroni p for "
                      f"{int(np.sum(Ph > Pb + 1e-9))} pair(s) "
                      "(monotonicity violated)")
            counts = {c: sum(1 for p in pairs if p[f"sig_{c}"]) for c in
                      ("raw", "holm", "bonferroni")}
            ds_data[ds] = {"n_pairs": n_pairs, "counts": counts,
                           "pairs": pairs, "lost": lost,
                           "unexpected": unexpected}
            status[ds] = S
            print(f"    [{met} | {ds}] {counts['raw']}/{n_pairs} raw → "
                  f"{counts['holm']}/{n_pairs} Holm → "
                  f"{counts['bonferroni']}/{n_pairs} Bonferroni "
                  f"({len(lost)} pair(s) lost)")
        per_metric[met] = ds_data
        status_grids[met] = status

    if raw_mismatch_any:
        print("  [WARNING] raw p matrices disagree between holm/bonferroni files!")

    # ---------------- JSON (source of truth) ----------------
    out = {"config": {"k": args.k, "seeds": list(range(42, 42 + args.seeds)),
                      "models": models, "metrics": args.metrics,
                      "alpha": ALPHA,
                      "note": "Lost = significant under Holm, not under Bonferroni"},
           "summary": {met: {ds: {"n_pairs": d["n_pairs"], **d["counts"]}
                             for ds, d in dd.items()}
                       for met, dd in per_metric.items()},
           "lost_pairs": {met: {ds: d["lost"] for ds, d in dd.items()}
                          for met, dd in per_metric.items()},
           "per_pair": {met: {ds: d["pairs"] for ds, d in dd.items()}
                        for met, dd in per_metric.items()},
           "unexpected": {met: {ds: d["unexpected"] for ds, d in dd.items()}
                          for met, dd in per_metric.items()}}
    json_path = os.path.join(RES_DIR, "correction_comparison.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ {os.path.basename(json_path)}")

    # ---------------- Markdown (paper-ready appendix) ----------------
    lines = ["# Correction Robustness — Holm vs Bonferroni (FIRE 2026)",
             "",
             "> Source: `results/k10_s10/significance_matrix_{metric}_{holm|bonferroni}.json` ",
             "> (paired t-tests, family of all 28 pairwise comparisons per dataset × metric). "
             "Simulated evaluation-harness data — see `model_sim.py`.",
             "",
             "**Reading the tables:** a pair is **lost** when it is significant at "
             f"α={ALPHA} under Holm-Bonferroni but no longer under the stricter "
             "Bonferroni correction (×28). Holm is always at least as powerful as "
             "Bonferroni, so `raw ≥ Holm ≤ Bonferroni` per pair.",
             ""]
    # summary counts table
    lines += ["## 1. Significant pairs at α=0.05 (of 28)",
              "",
              "| Metric | Dataset | raw | Holm | Bonferroni | lost |",
              "|---|---|---|---|---|---|"]
    for met in args.metrics:
        for ds, d in per_metric[met].items():
            c = d["counts"]
            lines.append(f"| {met} | {ds} | {c['raw']}/28 | {c['holm']}/28 | "
                         f"{c['bonferroni']}/28 | {len(d['lost'])} |")
    lines.append("")
    # lost pairs per metric
    lines += ["## 2. Pairs lost under Bonferroni (significant under Holm)",
              "",
              "| Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |",
              "|---|---|---|---|---|---|---|"]
    any_lost = False
    for met in args.metrics:
        for ds, d in per_metric[met].items():
            for p in d["lost"]:
                any_lost = True
                lines.append(f"| {met} | {ds} | {p['higher']} > {p['lower']} | "
                             f"{fmt_p(p['raw'])} | {fmt_p(p['holm'])} | "
                             f"{fmt_p(p['bonferroni'])} | {p['t']:.2f} |")
    if not any_lost:
        lines.append("| — | — | *(none — every Holm-significant pair also passes Bonferroni)* | — | — | — | — |")
    lines.append("")
    # full per-pair appendix tables
    lines += ["## 3. Full per-pair appendix (raw / Holm / Bonferroni p-values)",
              "",
              "| Metric | Dataset | Pair | raw p | Holm p | Bonferroni p | t |",
              "|---|---|---|---|---|---|---|"]
    for met in args.metrics:
        for ds, d in per_metric[met].items():
            for p in d["pairs"]:
                lines.append(f"| {met} | {ds} | {p['higher']} > {p['lower']} | "
                             f"{fmt_p(p['raw'])} | {fmt_p(p['holm'])} | "
                             f"{fmt_p(p['bonferroni'])} | {p['t']:.2f} |")
    lines.append("")
    md_path = os.path.join(RES_DIR, "correction_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(md_path)}")

    # ---------------- Figure: status grids per metric × dataset ----------------
    n_met = len(args.metrics)
    n_ds = len(list(status_grids[args.metrics[0]].keys()))
    fig, axes = plt.subplots(n_met, n_ds, figsize=(4.2 * n_ds, 4.0 * n_met),
                             squeeze=False)
    fig.suptitle(f"Correction Robustness — status per model pair ({tag})\n"
                 "green = significant under Holm & Bonferroni · orange = lost under Bonferroni "
                 "· grey = not significant under Holm",
                 fontsize=12, color=PALETTE["primary"], fontweight="bold")
    cmap = ListedColormap(["#FFFFFF", "#D5DBDB", PALETTE["orange"], PALETTE["green"]])
    for ri, met in enumerate(args.metrics):
        for ci, (ds, S) in enumerate(status_grids[met].items()):
            ax = axes[ri, ci]
            ax.imshow(S, cmap=cmap, vmin=0, vmax=3, aspect="auto")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.replace("-", "\n") for m in models],
                               fontsize=6.5, rotation=45, ha="right")
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.replace("-", "\n") for m in models], fontsize=6.5)
            for i in range(len(models)):
                for j in range(len(models)):
                    if S[i, j] == 2:
                        ax.text(j, i, "lost", ha="center", va="center", fontsize=6,
                                color="black", fontweight="bold")
            if ci == 0:
                ax.set_ylabel(met, color=PALETTE["primary"], fontweight="bold")
            ax.set_title(ds if ri == 0 else "", color=PALETTE["primary"],
                         fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(FIG_DIR, "corr1_correction_comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(png_path)}")

    print("\n" + "=" * 70)
    print("  SUMMARY — pairs lost under Bonferroni (Holm → Bonferroni)")
    print("=" * 70)
    total_lost = sum(len(d["lost"])
                     for met in args.metrics for d in per_metric[met].values())
    if total_lost == 0:
        print("  (no pairs lost — Holm and Bonferroni agree on every pair)")
    for met in args.metrics:
        for ds, d in per_metric[met].items():
            if d["lost"]:
                for p in d["lost"]:
                    print(f"  [{met} | {ds}] {p['higher']} > {p['lower']}: "
                          f"raw={p['raw']:.4f} holm={p['holm']:.4f} "
                          f"bonf={p['bonferroni']:.4f} (t={p['t']:.2f})")
    print(f"\n  Results → {json_path}, {md_path}")
    print("  Correction comparison complete ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="Folds of the source run (folder tag)")
    p.add_argument("--seeds", type=int, default=10, help="Seeds of the source run (folder tag)")
    p.add_argument("--metrics", nargs="+", default=KEY_METRICS, choices=KEY_METRICS,
                   help="Metrics to compare (default: all four)")
    main(p.parse_args())
