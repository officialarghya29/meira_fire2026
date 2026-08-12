"""
run_significance.py – FIRE 2026 MEIRA
=====================================
Full pairwise significance analysis on archived multi-seed results.

This script does NOT run the models. It loads the multi-seed experiment
output produced by run_experiments.py (default: results/k10_s10/
experiments.json, i.e. the camera-ready 10-fold / 10-seed run) and reports:

  - ALL pairwise paired t-tests between the 8 models (not just MEIRA-full
    vs the best baseline), per dataset, for a chosen metric (default
    nDCG@10). Per-seed values for all models come from the same test split
    per seed, so the pairs are naturally paired across the 10 seeds.
  - A compact per-metric significance summary of MEIRA-full vs every
    other model (F1, nDCG@10, MAP, MRR).

Multiple-comparison correction: p-values are adjusted (default Holm-
Bonferroni, or Bonferroni) for the family of ALL pairwise tests within
each dataset × metric (m = 28), so the pairwise tables survive multiplicity.

Produces (into config-named subfolders, tag = k{k}_s{seeds}; metric and
correction method are part of the output filenames so several metric ×
correction combinations coexist):
  results/k10_s10/significance_matrix_nDCG@10_holm.json – p-value / t-stat matrices
  results/k10_s10/significance_matrix_nDCG@10_holm.md   – paper-ready tables
  figures/k10_s10/sig1_pairwise_heatmap_nDCG@10_holm.png – -log10(p) heatmaps

  (with `--correction none` the `_holm` suffix is omitted)

Usage:
  python run_significance.py --k 10 --seeds 10 [--metric nDCG@10] [--correction holm]
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_correction import correct

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = None  # set in main() → figures/<tag>/
RES_DIR = None  # set in main() → results/<tag>/

PALETTE = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
           "neutral":"#6B7280","bg":"#F8FAFC","grid":"#E5E7EB",
           "green":"#27AE60","orange":"#E67E22"}

plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
    "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
    "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
    "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
    "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10})

KEY_METRICS = ["F1", "nDCG@10", "MAP", "MRR"]


def make_output_dirs(tag: str):
    """Create and return config-named output dirs, e.g. results/k10_s10/."""
    fig = os.path.join(BASE, "figures", tag)
    res = os.path.join(BASE, "results", tag)
    os.makedirs(fig, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    return fig, res


def load_multiseed(k: int, seeds: int) -> dict:
    """Load the archived run_experiments.py output for the given config."""
    path = os.path.join(BASE, "results", f"k{k}_s{seeds}", "experiments.json")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found.\n"
                 f"        Run `python run_experiments.py --k {k} --seeds {seeds}` first.")
    with open(path) as f:
        data = json.load(f)
    return data["multiseed"], data["config"]["models"]


def per_model_series(results, models, metric):
    """dict model → list of per-seed values for `metric` (None entries dropped)."""
    out = {}
    for m in models:
        vals = [r.get(metric) for r in results.get(m, {}).get("per_seed", [])]
        out[m] = [v for v in vals if v is not None]
    return out


def ttest(a, b):
    """Paired t-test on two same-length series; returns (t, p)."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 1.0
    t, p = stats.ttest_rel(a[:n], b[:n])
    if not np.isfinite(t):   # identical series → zero variance
        return 0.0, 1.0
    return float(t), float(p)


def pairwise_matrices(series, models, method="holm"):
    """Raw + multiplicity-corrected p-value matrices and t-stats.

    Correction is applied to the family of ALL pairwise tests
    (m = n(n−1)/2) within one dataset × metric, so reported significance
    survives multiplicity.
    """
    n = len(models)
    raws = np.ones((n, n)); P = np.ones((n, n)); T = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for (i, j) in pairs:
        t, p = ttest(series[models[i]], series[models[j]])
        raws[i, j] = raws[j, i] = p
        T[i, j] = t; T[j, i] = -t
    if method != "none":
        adj = correct([raws[i, j] for (i, j) in pairs], method)
        for (i, j), a in zip(pairs, adj):
            P[i, j] = P[j, i] = a
    else:
        P = raws.copy()
    return raws, P, T


def fmt_p(p):
    if p < 0.0001:
        return "<0.0001***"
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    return f"{p:.4f}{stars}"


def stat_desc(vals):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "—"


def out_name(prefix, metric, correction):
    """Config-aware output stem; correction is part of the filename."""
    suffix = "" if correction == "none" else f"_{correction}"
    return f"{prefix}{metric}{suffix}"


def write_markdown(models, metric, ds_results, seeds, tag, correction):
    n_pairs = len(models) * (len(models) - 1) // 2
    if correction == "none":
        corr_note = "p-values are **raw** (no multiple-comparison correction)."
    else:
        corr_note = (f"p-values are **{correction}-corrected** for the family of all "
                     f"{n_pairs} pairwise tests per dataset.")
    lines = ["# Pairwise Significance Matrix — MEIRA (FIRE 2026)",
             "",
             "> Loaded from the archived multi-seed run "
             f"(`results/{tag}/experiments.json`, {len(seeds)} seeds). "
             "Simulated evaluation-harness data — see `model_sim.py`.",
             "",
             f"Metric for the matrices: **{metric}** (paired t-test, two-sided). "
             f"{corr_note} "
             "`*` p<0.05, `**` p<0.01, `***` p<0.001.",
             ""]
    for ds_name, res in ds_results.items():
        P = res["P"]
        raws = res["raws"]
        series = res["series"]
        n_sig_raw = int(np.sum(raws[np.triu_indices(len(models), 1)] < 0.05))
        n_sig = int(np.sum(P[np.triu_indices(len(models), 1)] < 0.05))
        after = "raw" if correction == "none" else f"{correction} correction"
        lines += [f"## {ds_name}\n",
                  f"Significant pairs at α=0.05: **{n_sig_raw}/{n_pairs} raw** → "
                  f"**{n_sig}/{n_pairs}** after {after}.\n",
                  "| Model | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        for i, m in enumerate(models):
            cells = []
            for j in range(len(models)):
                if i == j:
                    cells.append("—")              # diagonal
                elif i < j:
                    cells.append(fmt_p(P[i, j]))    # upper triangle (corrected)
                else:
                    cells.append("")               # lower triangle (mirrored)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"**{metric} (mean±std across seeds):** " +
                     ", ".join(f"{m}={stat_desc(series[m])}" for m in models))
        lines += ["",
                  "**MEIRA-full vs each model (corrected p-values):**",
                  "",
                  "| Model | " + " | ".join(KEY_METRICS) + " |",
                  "|---|" + "---|" * len(KEY_METRICS)]
        for m in models:
            if m == "MEIRA-full":
                continue
            cells = [fmt_p(res["vs_full"][m][met]) for met in KEY_METRICS]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    path = os.path.join(RES_DIR, f"{out_name('significance_matrix_', metric, correction)}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(path)}")


def fig_heatmaps(models, ds_results, metric, correction):
    fig, axes = plt.subplots(1, len(ds_results), figsize=(6.5 * len(ds_results), 5.5))
    if len(ds_results) == 1:
        axes = [axes]
    fig.suptitle(f"Pairwise Significance Heatmaps (–log10 p, metric: {metric}, "
                 f"{correction}-corrected) – FIRE 2026",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for ax, (ds_name, res) in zip(axes, ds_results.items()):
        P = res["P"]
        n = len(models)
        L = -np.log10(np.clip(P, 1e-12, 1.0))
        im = ax.imshow(L, cmap="YlOrRd", vmin=0, vmax=8, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels([m.replace("-", "\n") for m in models],
                                                     fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels([m.replace("-", "\n") for m in models],
                                                     fontsize=7)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ax.text(j, i, fmt_p(P[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if L[i, j] > 4 else "black")
        ax.set_title(ds_name, color=PALETTE["primary"], fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.85, label="-log10(p)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{out_name('sig1_pairwise_heatmap_', metric, correction)}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def main(args):
    global FIG_DIR, RES_DIR
    tag = f"k{args.k}_s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)

    print("\n" + "=" * 60)
    print("  MEIRA – FIRE 2026 Pairwise Significance Analysis")
    print("  (all 8 models, archived multi-seed data)")
    print("=" * 60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    print(f"\n[1/3] Loading results/k{args.k}_s{args.seeds}/experiments.json")
    multiseed, models = load_multiseed(args.k, args.seeds)
    seeds = list(range(42, 42 + args.seeds))

    print(f"\n[2/3] Paired t-tests (metric: {args.metric}, correction: {args.correction})")
    ds_results = {}
    full = "MEIRA-full"
    for ds_name, results in multiseed.items():
        series = per_model_series(results, models, args.metric)
        raws, P, T = pairwise_matrices(series, models, args.correction)
        # vs-full: corrected p for every key metric (same 28-test family per metric)
        vs_full = {m: {} for m in models if m != full}
        i_full = models.index(full)
        for met in KEY_METRICS:
            Pc = P if met == args.metric else \
                pairwise_matrices(per_model_series(results, models, met),
                                  models, args.correction)[1]
            for j, m in enumerate(models):
                if j != i_full:
                    vs_full[m][met] = round(float(Pc[i_full, j]), 6)
        ds_results[ds_name] = {"series": series, "raws": raws, "P": P,
                               "T": T, "vs_full": vs_full}
        n_pairs = len(models) * (len(models) - 1) // 2
        n_raw = int(np.sum(raws[np.triu_indices(len(models), 1)] < 0.05))
        n_corr = int(np.sum(P[np.triu_indices(len(models), 1)] < 0.05))
        print(f"    [{ds_name}] significant pairs at α=0.05 ({args.metric}): "
              f"{n_raw}/{n_pairs} raw → {n_corr}/{n_pairs} "
              f"{args.correction}-corrected")

    stem = out_name("significance_matrix_", args.metric, args.correction)
    print(f"\n[3/3] Writing outputs → {RES_DIR}, {FIG_DIR}")
    write_markdown(models, args.metric, ds_results, seeds, tag, args.correction)
    with open(os.path.join(RES_DIR, f"{stem}.json"), "w") as f:
        json.dump({"config": {"k": args.k, "seeds": seeds, "models": models,
                              "metric": args.metric, "correction": args.correction},
                   "datasets": {ds: {"p_matrix": res["P"].tolist(),
                                     "p_matrix_raw": res["raws"].tolist(),
                                     "t_matrix": res["T"].tolist(),
                                     "mean_std": {m: stat_desc(s) for m, s in res["series"].items()},
                                     "vs_full": res["vs_full"]}
                                for ds, res in ds_results.items()}},
                  f, indent=2)
    print(f"  ✓ {stem}.json")
    fig_heatmaps(models, ds_results, args.metric, args.correction)

    print("\n" + "=" * 70)
    print(f"  PAIRWISE SIGNIFICANCE SUMMARY ({args.metric}, paired t-test, "
          f"{args.correction}-corrected)")
    print("=" * 70)
    for ds_name, res in ds_results.items():
        P = res["P"]
        print(f"\n  {ds_name}:")
        for i, m in enumerate(models):
            cells = []
            for j in range(len(models)):
                if i == j:
                    cells.append("  —  ")
                else:
                    cells.append(fmt_p(P[i, j]).ljust(11))
            print(f"    {m:<18} " + " ".join(cells))
        print(f"    mean±std: " + " | ".join(
            f"{m}={stat_desc(res['series'][m])}" for m in models))
    print(f"\n  Results → {RES_DIR}/{stem}.json, {RES_DIR}/{stem}.md")
    print("  Significance analysis complete ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="Folds of the source run (folder tag)")
    p.add_argument("--seeds", type=int, default=10, help="Seeds of the source run (folder tag)")
    p.add_argument("--metric", default="nDCG@10", help="Metric for the pairwise matrix")
    p.add_argument("--correction", default="holm",
                   choices=["none", "holm", "bonferroni"],
                   help="Multiple-comparison correction for the family of all pairwise "
                        "tests per dataset (default: holm)")
    main(p.parse_args())
