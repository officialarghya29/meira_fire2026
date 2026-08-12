"""
run_SOTA.py – FIRE 2026 MEIRA
======================================
State-of-the-Art Baseline Comparison ("leaderboard" run) on BOTH
benchmark datasets:
  1. FIRE-AgentIR-2026
  2. FIRE-CrossLingIR-2026

Compares MEIRA-full against standard IR baselines (BM25, TF-IDF,
Dense-IR, ColBERT-like) using the FULL metric suite, including the two
novel metrics proposed in this paper:
  - XAIR@K  (eXplainability-Adjusted IR)   — only defined for models with
                                              an XAI head (MEIRA variants)
  - MDS     (Memory Diversity Score)       — only defined for models with
                                              an episodic memory bank

Reports significance of MEIRA-full's improvement over the strongest
baseline via a paired t-test across seeds, on each dataset separately.

NOTE ON DATA PROVENANCE
------------------------
Model outputs come from `model_sim.py::simulate_model()`, a calibrated
numpy stand-in for real trained-model inference. This script validates
the leaderboard/statistical-testing pipeline end-to-end. Swap in real
model checkpoints before treating these as reportable SOTA numbers.

Produces (into config-named subfolders, tag = s{seeds}):
  results/s10/sota.json      – per-model, per-seed metrics + significance tests
  results/s10/sota_table.md  – markdown leaderboard table (paper-ready)
  figures/s10/sota_*.png     – leaderboard figures

Usage:
  python run_SOTA.py --seeds 5
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from datasets_fire import (build_agent_ir_dataset, build_crossling_ir_dataset,
                            stratified_split, dataset_stats)
from ir_metrics import full_ir_metrics
from model_sim import simulate_model

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = None  # set in main() → figures/<tag>/
RES_DIR = None  # set in main() → results/<tag>/

def make_output_dirs(tag: str):
    """Create and return config-named output dirs, e.g. results/s10/."""
    fig = os.path.join(BASE, "figures", tag)
    res = os.path.join(BASE, "results", tag)
    os.makedirs(fig, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    return fig, res

PALETTE = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
           "neutral":"#6B7280","bg":"#F8FAFC","grid":"#E5E7EB",
           "green":"#27AE60","orange":"#E67E22"}
COLORS  = [PALETTE["neutral"],"#95A5A6",PALETTE["secondary"],PALETTE["orange"],PALETTE["accent"]]

plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
    "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
    "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
    "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
    "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10})

# ── Leaderboard entrants ─────────────────────────────────────────────────
SOTA_MODELS = ["BM25", "TF-IDF", "Dense-IR", "ColBERT-like", "MEIRA-full"]
LEADERBOARD_METRICS = ["F1", "AUC", "AP", "nDCG@5", "nDCG@10", "MAP",
                        "MAP@10", "MRR", "R-Prec", "P@5", "P@10",
                        "XAIR@10", "MDS"]


def run_one(test_samples, model_name, seed, dataset_name):
    out = simulate_model(test_samples, model_name=model_name,
                          seed=seed, dataset_name=dataset_name)
    m = full_ir_metrics(
        labels=out["labels"], preds=out["preds"], probs=out["probs"],
        conv_ids=out["conv_ids"], xai_conf=out["xai_conf"],
        ret_idx=out["ret_indices"], threshold=out["threshold"],
        memory_slots=64)
    flat = {}
    flat.update(m["classification"]); flat.update(m["ranked"]); flat.update(m["novel"])
    return flat


def aggregate(rows):
    keys = set(k for r in rows for k in r.keys())
    return {k: {"mean": round(float(np.mean([r[k] for r in rows if k in r])), 4),
                "std":  round(float(np.std([r[k] for r in rows if k in r])), 4)}
            for k in keys}


def run_sota(datasets, seeds, models=SOTA_MODELS):
    results = {}
    for ds_name, samples in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        results[ds_name] = {}
        for model_name in models:
            seed_rows = []
            for seed in seeds:
                _, _, test = stratified_split(samples, seed=seed)
                seed_rows.append(run_one(test, model_name, seed, ds_name))
            agg = aggregate(seed_rows)
            results[ds_name][model_name] = {"per_seed": seed_rows, "aggregate": agg}
            xair = agg.get("XAIR@10", {}).get("mean")
            mds  = agg.get("MDS", {}).get("mean")
            print(f"    {model_name:<14} F1={agg['F1']['mean']:.3f}  "
                  f"nDCG@10={agg.get('nDCG@10',{}).get('mean',0):.3f}  "
                  f"MAP={agg.get('MAP',{}).get('mean',0):.3f}  "
                  f"MRR={agg.get('MRR',{}).get('mean',0):.3f}"
                  + (f"  XAIR@10={xair:.3f}" if xair is not None else "")
                  + (f"  MDS={mds:.3f}" if mds is not None else ""))
    return results


def significance_vs_best_baseline(results, metric="nDCG@10", champion="MEIRA-full"):
    """Paired t-test: champion vs the strongest non-champion baseline, per dataset."""
    sig = {}
    for ds_name, models in results.items():
        baselines = [m for m in models if m != champion]
        best_baseline = max(baselines, key=lambda m: models[m]["aggregate"].get(metric,{}).get("mean",0))
        champ_vals = [r.get(metric,0) for r in models[champion]["per_seed"]]
        base_vals  = [r.get(metric,0) for r in models[best_baseline]["per_seed"]]
        n = min(len(champ_vals), len(base_vals))
        if n >= 2:
            t_stat, p_val = stats.ttest_rel(champ_vals[:n], base_vals[:n])
        else:
            t_stat, p_val = 0.0, 1.0
        sig[ds_name] = {
            "metric": metric, "champion": champion, "best_baseline": best_baseline,
            "champion_mean": round(float(np.mean(champ_vals)), 4),
            "baseline_mean": round(float(np.mean(base_vals)), 4),
            "t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 4),
            "significant": bool(p_val < 0.05),
        }
    return sig


# ── Figures ──────────────────────────────────────────────────────────────

def fig_leaderboard_bars(results, metrics=("nDCG@10","MAP","MRR","F1")):
    datasets = list(results.keys())
    # camera-ready: figsize is the target print width (full text width of
    # ACM sigconf two-column) so fonts render at their nominal point size
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(7.2, 3.0))
    if len(datasets) == 1: axes = [axes]
    fig.suptitle("SOTA Leaderboard – MEIRA-full vs Baselines (FIRE 2026)",
                 fontsize=11, color=PALETTE["primary"], fontweight="bold")
    for di, ds_name in enumerate(datasets):
        for mi, metric in enumerate(metrics):
            ax = axes[di][mi]
            means = [results[ds_name][m]["aggregate"].get(metric,{}).get("mean",0) for m in SOTA_MODELS]
            stds  = [results[ds_name][m]["aggregate"].get(metric,{}).get("std",0) for m in SOTA_MODELS]
            x = np.arange(len(SOTA_MODELS))
            bars = ax.bar(x, means, color=COLORS[:len(SOTA_MODELS)], edgecolor="white",
                          alpha=0.9, width=0.62)
            ax.errorbar(x, means, yerr=stds, fmt="none", color=PALETTE["neutral"], capsize=3, lw=1.2)
            ax.set_xticks(x); ax.set_xticklabels(SOTA_MODELS, fontsize=6.5, rotation=20, ha="right")
            ax.set_ylabel(metric if mi==0 else "")
            ax.set_title(f"{ds_name.split('-')[1] if '-' in ds_name else ds_name} – {metric}",
                         fontsize=9, color=PALETTE["primary"])
            ax.grid(True, axis="y")
            lo = max(0, min(means)-0.1); hi = min(1, max(means)+0.12)
            ax.set_ylim(lo, hi)
            best_i = int(np.argmax(means))
            ax.annotate("★", xy=(x[best_i], means[best_i]+stds[best_i]+0.01),
                        ha="center", fontsize=9, color=PALETTE["accent"])
            for bar, mu in zip(bars, means):
                ax.annotate(f"{mu:.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0,3), textcoords="offset points", ha="center", fontsize=6)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sota1_leaderboard_bars.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_novel_metrics(results):
    """XAIR@10 and MDS — only meaningful for MEIRA-full among these baselines."""
    datasets = list(results.keys())
    fig, axes = plt.subplots(1, len(datasets)*2, figsize=(7.2, 1.75))
    fig.suptitle("Novel Metrics on Two FIRE Datasets — XAIR@10 & MDS (MEIRA-full only)",
                 fontsize=10, color=PALETTE["primary"], fontweight="bold")
    idx = 0
    for ds_name in datasets:
        for metric, color in [("XAIR@10", PALETTE["secondary"]), ("MDS", PALETTE["orange"])]:
            ax = axes[idx]
            val = results[ds_name]["MEIRA-full"]["aggregate"].get(metric,{}).get("mean", 0)
            std = results[ds_name]["MEIRA-full"]["aggregate"].get(metric,{}).get("std", 0)
            ax.bar([0], [val], color=color, width=0.5, edgecolor="white", alpha=0.9)
            ax.errorbar([0], [val], yerr=[std], fmt="none", color=PALETTE["neutral"], capsize=5)
            ax.set_xticks([0]); ax.set_xticklabels(["MEIRA-full"])
            ax.set_ylim(0, 1); ax.set_ylabel(metric)
            ax.set_title(f"{ds_name.split('-')[1]}\n{metric}", fontsize=7.5, color=PALETTE["primary"])
            ax.annotate(f"{val:.3f}±{std:.3f}", xy=(0, val), xytext=(0,5),
                        textcoords="offset points", ha="center", fontsize=7)
            ax.grid(True, axis="y")
            idx += 1
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sota2_novel_metrics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


# ── Markdown table ────────────────────────────────────────────────────

def write_markdown_table(results, sig, path):
    lines = ["# SOTA Leaderboard — MEIRA vs Baselines (FIRE 2026)",
             "",
             "> Simulated evaluation-harness run (see `model_sim.py`). "
             "Replace `simulate_model()` with real trained-model inference "
             "before citing these numbers in a submission.",
             ""]
    for ds_name in results:
        lines.append(f"## {ds_name}\n")
        lines.append("| Model | " + " | ".join(LEADERBOARD_METRICS) + " |")
        lines.append("|---|" + "---|"*len(LEADERBOARD_METRICS))
        for model_name in SOTA_MODELS:
            agg = results[ds_name][model_name]["aggregate"]
            row = []
            for m in LEADERBOARD_METRICS:
                v = agg.get(m)
                row.append(f"{v['mean']:.3f}±{v['std']:.3f}" if v else "—")
            marker = " **(ours)**" if model_name == "MEIRA-full" else ""
            lines.append(f"| {model_name}{marker} | " + " | ".join(row) + " |")
        lines.append("")
        s = sig[ds_name]
        sig_str = "significant" if s["significant"] else "not significant"
        lines.append(f"**Significance test (paired t-test, {s['metric']}):** "
                      f"{s['champion']} ({s['champion_mean']:.3f}) vs best baseline "
                      f"{s['best_baseline']} ({s['baseline_mean']:.3f}) → "
                      f"t={s['t_stat']:.3f}, p={s['p_value']:.4f} ({sig_str} at α=0.05)\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    global FIG_DIR, RES_DIR
    tag = f"s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)

    print("\n" + "="*60)
    print("  MEIRA – FIRE 2026 SOTA Comparison")
    print("  Leaderboard on Two IR Benchmark Datasets")
    print("="*60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    print("\n[1/3] Building datasets")
    agent_samples     = build_agent_ir_dataset(n_convs=350, seed=42)
    crossling_samples = build_crossling_ir_dataset(n_convs=200, seed=42)
    datasets = {
        "FIRE-AgentIR-2026":     agent_samples,
        "FIRE-CrossLingIR-2026": crossling_samples,
    }
    for name, samps in datasets.items():
        st = dataset_stats(samps, name)
        print(f"  {name}: {st['total']:,} samples  pos={st['positives']}({100*st['pos_ratio']:.0f}%)  "
              f"hard_neg={st['hard_neg']}")

    seeds = list(range(42, 42 + args.seeds))
    print(f"\n[2/3] Running SOTA comparison ({args.seeds} seeds × {len(SOTA_MODELS)} models)")
    results = run_sota(datasets, seeds)
    sig = significance_vs_best_baseline(results, metric="nDCG@10")

    print(f"\n[3/3] Generating figures → {FIG_DIR}")
    fig_leaderboard_bars(results)
    fig_novel_metrics(results)

    with open(os.path.join(RES_DIR, "sota.json"), "w") as f:
        json.dump({"results": results, "significance": sig,
                    "config": {"seeds": seeds, "models": SOTA_MODELS}}, f, indent=2)
    write_markdown_table(results, sig, os.path.join(RES_DIR, "sota_table.md"))

    print("\n" + "="*70)
    print("  SOTA SUMMARY")
    print("="*70)
    for ds_name in datasets:
        s = sig[ds_name]
        print(f"\n  {ds_name}: MEIRA-full nDCG@10={s['champion_mean']:.3f} vs "
              f"best baseline {s['best_baseline']}={s['baseline_mean']:.3f}  "
              f"(p={s['p_value']:.4f}, {'sig.' if s['significant'] else 'n.s.'})")
    print(f"\n  Results → {RES_DIR}/sota.json, {RES_DIR}/sota_table.md")
    print("  SOTA comparison complete ✓")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    main(p.parse_args())
