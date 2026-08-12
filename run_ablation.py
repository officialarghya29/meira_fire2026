"""
run_ablation.py – FIRE 2026 MEIRA
======================================
Component Ablation Study on BOTH benchmark datasets:
  1. FIRE-AgentIR-2026
  2. FIRE-CrossLingIR-2026

Ablates each MEIRA component (episodic memory, XAI head, temporal decay)
by comparing the full model against variants with one component removed,
across multiple seeds, and reports the delta each component contributes
to standard IR metrics AND the two novel metrics proposed in this paper
(XAIR@10, MDS) — i.e. the new evaluation metrics are exercised on both
FIRE datasets as part of the ablation, not just the leaderboard run.

NOTE ON DATA PROVENANCE
------------------------
Model outputs come from `model_sim.py::simulate_model()`, a calibrated
numpy stand-in for the real MEIRA forward pass (see that file's docstring).
This script is an *evaluation-harness* ablation: it validates that the
pipeline correctly measures component contribution end-to-end. Before
these numbers are reported in an actual paper/leaderboard, swap
`simulate_model()` for real trained-model inference on real checkpoints.

Produces (into config-named subfolders, tag = s{seeds}):
  results/s10/ablation.json           – per-component, per-seed metrics
  results/s10/ablation_table.md       – markdown ablation table (paper-ready)
  figures/s10/abl_*.png               – ablation figures

Usage:
  python run_ablation.py --seeds 5
"""

import os, sys, json, argparse
import numpy as np
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
COLORS  = [PALETTE["primary"],PALETTE["secondary"],PALETTE["accent"],
           PALETTE["green"],PALETTE["orange"]]

plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
    "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
    "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
    "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
    "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10})

# ── Ablation ladder ─────────────────────────────────────────────────────
# Each variant removes exactly one MEIRA component relative to MEIRA-full.
ABLATION_VARIANTS = [
    ("MEIRA-full",       "Full model (memory + XAI + decay)"),
    ("MEIRA-no-memory",  "− Episodic memory"),
    ("MEIRA-no-xai",     "− XAI attribution head"),
    ("MEIRA-no-decay",   "− Temporal decay"),
]
KEY_METRICS = ["F1","AUC","nDCG@10","MAP","MRR","XAIR@10","MDS"]


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


def run_ablation(datasets, seeds):
    results = {}
    for ds_name, samples in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        results[ds_name] = {}
        for model_name, desc in ABLATION_VARIANTS:
            seed_rows = []
            for seed in seeds:
                _, _, test = stratified_split(samples, seed=seed)
                seed_rows.append(run_one(test, model_name, seed, ds_name))
            agg = aggregate(seed_rows)
            results[ds_name][model_name] = {"desc": desc, "per_seed": seed_rows, "aggregate": agg}
            print(f"    {model_name:<18} ({desc:<32})  "
                  f"F1={agg['F1']['mean']:.3f}±{agg['F1']['std']:.3f}  "
                  f"nDCG@10={agg.get('nDCG@10',{}).get('mean',0):.3f}  "
                  f"XAIR@10={agg.get('XAIR@10',{}).get('mean',0):.3f}  "
                  f"MDS={agg.get('MDS',{}).get('mean',0):.3f}")
    return results


def deltas_vs_full(results):
    """Compute how much each removed component costs, per dataset/metric."""
    deltas = {}
    for ds_name, variants in results.items():
        full_agg = variants["MEIRA-full"]["aggregate"]
        deltas[ds_name] = {}
        for model_name, desc in ABLATION_VARIANTS[1:]:
            v_agg = variants[model_name]["aggregate"]
            deltas[ds_name][model_name] = {
                m: round(full_agg.get(m,{}).get("mean",0) - v_agg.get(m,{}).get("mean",0), 4)
                for m in KEY_METRICS
            }
    return deltas


# ── Figures ──────────────────────────────────────────────────────────────

def fig_ablation_bars(results):
    """Component contribution bar chart per dataset (F1, nDCG@10, XAIR@10, MDS)."""
    datasets = list(results.keys())
    metrics_to_show = ["F1", "nDCG@10", "XAIR@10", "MDS"]
    # camera-ready: figsize is the target print width (ACM sigconf full text
    # width) so fonts render at their nominal point size
    fig, axes = plt.subplots(len(datasets), len(metrics_to_show),
                              figsize=(7.2, 3.2))
    if len(datasets) == 1: axes = [axes]
    fig.suptitle("Ablation Study – Component Contribution (FIRE 2026)",
                 fontsize=11, color=PALETTE["primary"], fontweight="bold")
    variant_names = [v for v, _ in ABLATION_VARIANTS]
    for di, ds_name in enumerate(datasets):
        for mi, metric in enumerate(metrics_to_show):
            ax = axes[di][mi]
            means = [results[ds_name][v]["aggregate"].get(metric,{}).get("mean",0) for v in variant_names]
            stds  = [results[ds_name][v]["aggregate"].get(metric,{}).get("std",0) for v in variant_names]
            x = np.arange(len(variant_names))
            bars = ax.bar(x, means, color=COLORS[:len(variant_names)], edgecolor="white",
                          alpha=0.88, width=0.6)
            ax.errorbar(x, means, yerr=stds, fmt="none", color=PALETTE["neutral"], capsize=3, lw=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace("MEIRA-","").replace("-","\n") for v in variant_names],
                               fontsize=6.5)
            ax.set_ylabel(metric if mi == 0 else "")
            ax.set_title(f"{ds_name.split('-')[1] if '-' in ds_name else ds_name} – {metric}",
                         fontsize=9, color=PALETTE["primary"])
            ax.grid(True, axis="y")
            lo = max(0, min(means) - 0.1); hi = min(1, max(means) + 0.1)
            ax.set_ylim(lo, hi)
            for bar, mu in zip(bars, means):
                ax.annotate(f"{mu:.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                            xytext=(0,3), textcoords="offset points", ha="center", fontsize=6)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "abl1_component_bars.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_delta_heatmap(deltas):
    """Heatmap: how much each ablated component costs, per dataset."""
    datasets = list(deltas.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.7))
    if len(datasets) == 1: axes = [axes]
    fig.suptitle("Performance Drop When Component Removed (Δ vs MEIRA-full)",
                 fontsize=11, color=PALETTE["primary"], fontweight="bold")
    variants = [v for v, _ in ABLATION_VARIANTS[1:]]
    for ax, ds_name in zip(axes, datasets):
        mat = np.array([[deltas[ds_name][v][m] for m in KEY_METRICS] for v in variants])
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=-0.05, vmax=0.15)
        ax.set_xticks(range(len(KEY_METRICS))); ax.set_xticklabels(KEY_METRICS, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels([v.replace("MEIRA-","−") for v in variants], fontsize=8)
        ax.set_title(ds_name, color=PALETTE["primary"], fontweight="bold", fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", fontsize=7,
                        color="black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Δ (full − ablated)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "abl2_delta_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


# ── Markdown table (paper-ready) ─────────────────────────────────────────

def write_markdown_table(results, deltas, path):
    lines = ["# Ablation Study — MEIRA (FIRE 2026)",
             "",
             "> Simulated evaluation-harness run (see `model_sim.py`). "
             "Replace `simulate_model()` with real trained-model inference "
             "before citing these numbers in a submission.",
             ""]
    for ds_name in results:
        lines.append(f"## {ds_name}\n")
        lines.append("| Variant | " + " | ".join(KEY_METRICS) + " |")
        lines.append("|---|" + "---|"*len(KEY_METRICS))
        for model_name, desc in ABLATION_VARIANTS:
            agg = results[ds_name][model_name]["aggregate"]
            row = [f"{agg.get(m,{}).get('mean',0):.3f}±{agg.get(m,{}).get('std',0):.3f}" for m in KEY_METRICS]
            lines.append(f"| **{model_name}** ({desc}) | " + " | ".join(row) + " |")
        lines.append("")
        lines.append("**Δ vs full (component contribution):**\n")
        lines.append("| Removed component | " + " | ".join(KEY_METRICS) + " |")
        lines.append("|---|" + "---|"*len(KEY_METRICS))
        for model_name, desc in ABLATION_VARIANTS[1:]:
            row = [f"{deltas[ds_name][model_name][m]:+.3f}" for m in KEY_METRICS]
            lines.append(f"| {desc} | " + " | ".join(row) + " |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    global FIG_DIR, RES_DIR
    tag = f"s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)

    print("\n" + "="*60)
    print("  MEIRA – FIRE 2026 Ablation Study")
    print("  Component Contribution on Two IR Benchmark Datasets")
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
    print(f"\n[2/3] Running ablation ({args.seeds} seeds × {len(ABLATION_VARIANTS)} variants)")
    results = run_ablation(datasets, seeds)
    deltas = deltas_vs_full(results)

    print(f"\n[3/3] Generating figures → {FIG_DIR}")
    fig_ablation_bars(results)
    fig_delta_heatmap(deltas)

    with open(os.path.join(RES_DIR, "ablation.json"), "w") as f:
        json.dump({"results": results, "deltas": deltas,
                    "config": {"seeds": seeds, "variants": [v for v,_ in ABLATION_VARIANTS]}}, f, indent=2)
    write_markdown_table(results, deltas, os.path.join(RES_DIR, "ablation_table.md"))

    print("\n" + "="*70)
    print("  ABLATION SUMMARY (largest component contribution wins)")
    print("="*70)
    for ds_name in datasets:
        print(f"\n  {ds_name}:")
        for model_name, desc in ABLATION_VARIANTS[1:]:
            d = deltas[ds_name][model_name]
            print(f"    {desc:<32} ΔF1={d['F1']:+.3f}  ΔnDCG@10={d['nDCG@10']:+.3f}  "
                  f"ΔXAIR@10={d['XAIR@10']:+.3f}  ΔMDS={d['MDS']:+.3f}")
    print(f"\n  Results → {RES_DIR}/ablation.json, {RES_DIR}/ablation_table.md")
    print("  Ablation complete ✓")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    main(p.parse_args())
