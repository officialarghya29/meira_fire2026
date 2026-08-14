"""
compare_metric_orderings.py – FIRE 2026 MEIRA
=============================================
Cross-metric model-ordering stability analysis on archived multi-seed
results (results/k{k}_s{seeds}/experiments.json, i.e. the camera-ready
10-fold / 10-seed run).

For each dataset and metric (F1, nDCG@10, MAP, MRR) it:
  - ranks the 8 models by mean performance (1 = best);
  - computes pairwise paired t-tests so it can flag "close" adjacencies —
    adjacent pairs whose difference is NOT significant after multiple-
    comparison correction (default Holm-Bonferroni, family = all 28
    pairwise tests per metric × dataset), i.e. the places where the
    reported ordering could flip across runs/metrics;
  - measures agreement between metric orderings with Spearman rank
    correlation.

Produces (tag = k{k}_s{seeds}; correction method is part of the filenames,
omitted when `--correction none`):
  results/k10_s10/metric_ordering_stability_holm.json – rankings, adjacencies,
                                                       correlations
  results/k10_s10/metric_ordering_stability_holm.md   – paper-ready tables
  figures/k10_s10/ord1_ordering_stability_holm.png    – parallel-coordinates +
                                                       rank-matrix figure

Usage:
  python compare_metric_orderings.py --k 10 --seeds 10 [--correction holm]
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

METRICS = ["F1", "nDCG@10", "MAP", "MRR"]
FULL = "MEIRA-full"

PALETTE = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
           "neutral":"#6B7280","bg":"#F8FAFC","grid":"#E5E7EB"}
COLORS = ["#1B4F72","#2E86AB","#E84855","#27AE60","#E67E22",
          "#8E44AD","#2C3E50","#16A085"]

plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
    "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
    "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
    "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
    "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10})


def make_output_dirs(tag: str):
    """Create and return config-named output dirs, e.g. results/k10_s10/."""
    fig = os.path.join(BASE, "figures", tag)
    res = os.path.join(BASE, "results", tag)
    os.makedirs(fig, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    return fig, res


def load_multiseed(k: int, seeds: int) -> tuple:
    path = os.path.join(BASE, "results", f"k{k}_s{seeds}", "experiments.json")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found.\n"
                 f"        Run `python run_experiments.py --k {k} --seeds {seeds}` first.")
    with open(path) as f:
        data = json.load(f)
    return data["multiseed"], data["config"]["models"]


def series_for(results, model, metric):
    return [r.get(metric) for r in results.get(model, {}).get("per_seed", [])
            if r.get(metric) is not None]


def mean_std(vals):
    return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)) if vals else (None, None)


def rank_models(means):
    """Rank models by mean (1 = best); ties share the average rank."""
    ordered = sorted(means, key=lambda m: -(means[m] if means[m] is not None else -1))
    rank = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and means[ordered[j + 1]] == means[ordered[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            rank[ordered[t]] = avg
        i = j + 1
    return rank


def fmt_p(p):
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def ttest(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 1.0
    t, p = stats.ttest_rel(a[:n], b[:n])
    if not np.isfinite(t):
        return 0.0, 1.0
    return float(t), float(p)


def analyze_dataset(results, models, method="holm"):
    """Per-metric rankings + adjacency significance + cross-metric correlations.

    Adjacency significance is based on p-values corrected (Holm/Bonferroni)
    for the family of ALL pairwise tests (m = 28) within each metric × dataset.
    """
    means = {}
    stds = {}
    per_seed = {}
    n = len(models)
    for met in METRICS:
        vals = {m: series_for(results, m, met) for m in models}
        per_seed[met] = vals
        means[met] = {m: mean_std(vals[m])[0] for m in models}
        stds[met] = {m: mean_std(vals[m])[1] for m in models}

    # rankings: 1 = best, ties share the average rank
    rank = {met: rank_models(means[met]) for met in METRICS}

    # adjacent-pair significance per metric (corrected within the full 28-pair family)
    adjacency = {}
    for met in METRICS:
        ordered = sorted(models, key=lambda m: -(means[met][m] if means[met][m] is not None else -1))
        raws = np.ones((n, n)); adj = np.ones((n, n)); T = np.zeros((n, n))
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for (i, j) in pairs:
            t, p = ttest(per_seed[met][models[i]], per_seed[met][models[j]])
            raws[i, j] = raws[j, i] = p
            T[i, j] = t; T[j, i] = -t
        if method != "none":
            corr = correct([raws[i, j] for (i, j) in pairs], method)
            for (i, j), a in zip(pairs, corr):
                adj[i, j] = adj[j, i] = a
        else:
            adj = raws.copy()
        entries = []
        for k in range(len(ordered) - 1):
            a, b = ordered[k], ordered[k + 1]
            ia, ib = models.index(a), models.index(b)
            p_raw = float(raws[ia, ib]); p_adj = float(adj[ia, ib])
            diff = None if (means[met][a] is None or means[met][b] is None) \
                else round(means[met][a] - means[met][b], 4)
            entries.append({"higher": a, "lower": b,
                            "diff": diff,
                            "t": round(float(T[ia, ib]), 4),
                            "p_raw": round(p_raw, 4),
                            "p": round(p_adj, 4),
                            "significant_raw": bool(p_raw < 0.05),
                            "significant": bool(p_adj < 0.05)})
        adjacency[met] = entries

    # Spearman correlation between metric orderings (on mean ranks)
    corr = {}
    for i, m1 in enumerate(METRICS):
        for m2 in METRICS[i + 1:]:
            rho, p = stats.spearmanr([rank[m1][m] for m in models],
                                     [rank[m2][m] for m in models])
            corr[f"{m1} vs {m2}"] = {"spearman": round(float(rho), 4),
                                     "p": round(float(p), 4)}

    return {"means": means, "stds": stds, "rank": rank,
            "adjacency": adjacency, "corr": corr}


def fmt_mean(m, s):
    return "—" if m is None else f"{m:.3f}±{s:.3f}"


def out_stem(correction):
    """Config-aware output stem; correction is part of the filename."""
    suffix = "" if correction == "none" else f"_{correction}"
    return f"metric_ordering_stability{suffix}"


def fig_ordering_stability(ds_results, models, tag, method="holm"):
    """Parallel-coordinates (rank per metric) + rank-matrix 'barcode' panels."""
    n_ds = len(ds_results)
    n_met = len(METRICS)
    # camera-ready: figsize is the target print width (ACM sigconf full text
    # width) so fonts render at their nominal point size
    # Explicit gridspec: deterministic margins (mpl 3.11 layout engines leave
    # ~100px of slack between rows and crush the panels). Gaps sized for the
    # real text extents: bottom=0.17 reserves the legend band; the row gap
    # (hspace) clears the x labels and colorbar end ticks; explicit colorbar
    # ticks avoid 0.0/10.0 overhang labels. TUNED to the current data/fonts -
    # re-run audit_figures.py after any data, seed, or font change.
    fig = plt.figure(figsize=(7.2, 3.30))
    gs = fig.add_gridspec(n_ds, 2, left=0.09, right=0.975, top=0.87,
                          bottom=0.17, wspace=0.14, hspace=0.50)
    axes = [[fig.add_subplot(gs[di, ci]) for ci in range(2)]
            for di in range(n_ds)]
    fig.suptitle("Model-Ordering Stability Across Metrics (Holm-corrected)",
                 fontsize=11, color=PALETTE["primary"], fontweight="bold", y=0.99)
    x = np.arange(n_met)
    handles = labels = None
    for di, (ds_name, res) in enumerate(ds_results.items()):
        order = sorted(models, key=lambda m: res["rank"]["F1"][m])  # rank 1 first
        orders = {met: tuple(sorted(models, key=lambda m: res["rank"][met][m]))
                  for met in METRICS}
        stable = len(set(orders.values())) == 1
        # models involved in any non-significant adjacency (fragile — could flip)
        fragile_models = set()
        for met in METRICS:
            for pr in res["adjacency"][met]:
                if not pr["significant"]:
                    fragile_models.update([pr["higher"], pr["lower"]])

        ax = axes[di][0]
        for k, m in enumerate(order):
            ranks = [res["rank"][met][m] for met in METRICS]
            fragile = m in fragile_models
            color = PALETTE["accent"] if m == FULL else COLORS[k % len(COLORS)]
            ax.plot(x, ranks, marker="o", ms=4,
                    lw=2.6 if m == FULL else (1.6 if fragile else 1.1),
                    ls="--" if fragile else "-",
                    color=color, label=m, alpha=0.95,
                    zorder=5 if m == FULL else 3)
        if di == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xticks(x); ax.set_xticklabels(METRICS, fontsize=8)
        ax.set_yticks(range(1, len(models) + 1))
        ax.tick_params(axis="y", labelsize=8)   # 1..8 ranks in a short panel
        ax.set_ylim(len(models) + 0.5, 0.5)   # rank 1 on top
        ax.set_ylabel("Rank (1 = best)")
        ax.grid(True, axis="y", alpha=0.6)
        ds_short = ds_name.split('-')[1] if '-' in ds_name else ds_name
        ax.set_title(f"{ds_short} – parallel coordinates", fontsize=9,
                     color=PALETTE["primary"], fontweight="bold")
        msg = ("identical ranking across metrics — no crossings"
               if stable else "orderings differ across metrics")
        ax.text(0.02, 0.05, msg, transform=ax.transAxes, fontsize=7,
                color=PALETTE["neutral"], style="italic")

        ax2 = axes[di][1]
        mat = np.array([[res["rank"][met][m] for met in METRICS] for m in order])
        im = ax2.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=1, vmax=len(models))
        ax2.set_xticks(range(n_met)); ax2.set_xticklabels(METRICS, fontsize=8)
        ax2.set_yticks(range(len(order)))
        ax2.set_yticklabels(order, fontsize=7)
        for i in range(len(order)):
            for j in range(n_met):
                v = mat[i, j]
                ax2.text(j, i, f"{v:g}", ha="center", va="center", fontsize=7,
                         color="white" if v > len(models) / 2 else "black")
        ax2.set_title(f"{ds_short} – rank matrix", fontsize=9.5,
                      color=PALETTE["primary"], fontweight="bold")
        cbar = fig.colorbar(im, ax=ax2, shrink=0.85, label="Rank",
                            ticks=list(range(1, len(models) + 1)))
        cbar.ax.tick_params(labelsize=8)   # 8 ticks fit the short colorbar
    if handles is not None:
        # legend lives in the reserved bottom band (gridspec bottom=0.13)
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
                   framealpha=0.9)
    path = os.path.join(FIG_DIR, f"ord1_ordering_stability{'' if method == 'none' else '_' + method}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def write_markdown(ds_results, models, tag, seeds, method):
    lines = ["# Model-Ordering Stability Across Metrics — MEIRA (FIRE 2026)",
             "",
             "> Loaded from the archived multi-seed run "
             f"(`results/{tag}/experiments.json`, {len(seeds)} seeds). "
             "Simulated evaluation-harness data — see `model_sim.py`.",
             "",
             "Ranks: 1 = best mean. Adjacent pairs marked with `†` are "
             "**not** significantly different — p-values are "
             + ("**raw**" if method == "none" else f"**{method}-corrected**")
             + " for the family of all 28 pairwise tests per metric × dataset "
             "(p ≥ 0.05 after correction) — the places where the ordering "
             "could flip.",
             ""]
    for ds_name, res in ds_results.items():
        lines += [f"## {ds_name}\n",
                  "### Rankings (per metric)\n",
                  "| Model | " + " | ".join(f"{m}" for m in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        # order rows by F1 mean (reference ordering)
        rows = sorted(models, key=lambda m: -(res["means"]["F1"][m] or -1))
        for m in rows:
            cells = [f"{res['rank'][met][m]:g}" for met in METRICS]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["",
                  "### Mean performance (mean±std across seeds)\n",
                  "| Model | " + " | ".join(METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for m in rows:
            cells = [fmt_mean(res["means"][met][m], res["stds"][met][m]) for met in METRICS]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["",
                  "### Adjacency significance (per metric, adjacent pairs only)\n"]
        for met in METRICS:
            lines.append(f"**{met}:**")
            lines.append(f"| Higher | Lower | Δ mean | t | p raw | p ({method}) | sig |")
            lines.append("|---|---|---|---|---|---|---|")
            for pr in res["adjacency"][met]:
                mark = "†" if not pr["significant"] else ""
                diff = "—" if pr["diff"] is None else f"{pr['diff']:+.4f}"
                lines.append(f"| {pr['higher']} | {pr['lower']} | {diff} | "
                             f"{pr['t']:.3f} | {fmt_p(pr['p_raw'])} | "
                             f"{fmt_p(pr['p'])}{mark} | "
                             f"{'yes' if pr['significant'] else 'no'} |")
            lines.append("")
        lines += ["### Rank correlation between metric orderings (Spearman)\n",
                  "| Metric pair | ρ | p |",
                  "|---|---|---|"]
        for k, v in res["corr"].items():
            lines.append(f"| {k} | {v['spearman']:.4f} | {v['p']:.4f} |")
        lines.append("")
    path = os.path.join(RES_DIR, f"{out_stem(method)}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(path)}")


def main(args):
    global FIG_DIR, RES_DIR
    tag = f"k{args.k}_s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)

    print("\n" + "=" * 60)
    print("  MEIRA – FIRE 2026 Cross-Metric Ordering Stability")
    print("=" * 60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    print(f"\n[1/3] Loading results/k{args.k}_s{args.seeds}/experiments.json")
    multiseed, models = load_multiseed(args.k, args.seeds)
    seeds = list(range(42, 42 + args.seeds))

    print(f"\n[2/3] Ranking models per metric ({', '.join(METRICS)}), "
          f"correction: {args.correction})")
    ds_results = {}
    for ds_name, results in multiseed.items():
        res = analyze_dataset(results, models, args.correction)
        ds_results[ds_name] = res
        ref = sorted(models, key=lambda m: -(res["means"]["F1"][m] or -1))
        print(f"    [{ds_name}] F1 rank: " +
              ", ".join(f"{m}={res['rank']['F1'][m]}" for m in ref))
        for met in METRICS:
            entries = res["adjacency"][met]
            n_raw = sum(1 for pr in entries if pr["significant_raw"])
            n_corr = sum(1 for pr in entries if pr["significant"])
            print(f"      {met:<8} adjacent pairs significant at α=0.05: "
                  f"{n_raw}/{len(entries)} raw → {n_corr}/{len(entries)} "
                  f"{args.correction}-corrected")

    print(f"\n[3/3] Writing outputs → {RES_DIR}, {FIG_DIR}")
    write_markdown(ds_results, models, tag, seeds, args.correction)
    with open(os.path.join(RES_DIR, f"{out_stem(args.correction)}.json"), "w") as f:
        json.dump({"config": {"k": args.k, "seeds": seeds, "models": models,
                              "metrics": METRICS, "correction": args.correction},
                   "datasets": {ds: {"rank": res["rank"],
                                     "means": res["means"], "stds": res["stds"],
                                     "adjacency": res["adjacency"],
                                     "corr": res["corr"]}
                                for ds, res in ds_results.items()}},
                  f, indent=2)
    print(f"  ✓ {out_stem(args.correction)}.json")
    fig_ordering_stability(ds_results, models, tag, args.correction)

    print("\n" + "=" * 70)
    print("  ORDERING STABILITY SUMMARY")
    print("=" * 70)
    for ds_name, res in ds_results.items():
        print(f"\n  {ds_name}:")
        ref = sorted(models, key=lambda m: -(res["means"]["F1"][m] or -1))
        print(f"    F1 ranking:      " + " > ".join(ref))
        for met in METRICS[1:]:
            ordered = sorted(models, key=lambda m: -(res["means"][met][m] or -1))
            same = ordered == ref
            print(f"    {met:<16} " + " > ".join(ordered) +
                  ("" if same else "   ⚠ differs from F1"))
        close = set()
        for met in METRICS:
            for pr in res["adjacency"][met]:
                if not pr["significant"]:
                    close.add((pr["higher"], pr["lower"]))
        if close:
            print("    Close adjacencies (not significant in ≥1 metric): " +
                  "; ".join(f"{a}>{b}" for a, b in sorted(close)))
        else:
            print("    No close adjacencies — all adjacent pairs significant in every metric")
    stem = out_stem(args.correction)
    print(f"\n  Results → {RES_DIR}/{stem}.json, {RES_DIR}/{stem}.md")
    print("  Ordering comparison complete ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="Folds of the source run (folder tag)")
    p.add_argument("--seeds", type=int, default=10, help="Seeds of the source run (folder tag)")
    p.add_argument("--correction", default="holm",
                   choices=["none", "holm", "bonferroni"],
                   help="Multiple-comparison correction for the family of all pairwise "
                        "tests per metric × dataset (default: holm)")
    main(p.parse_args())
