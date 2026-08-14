"""
run_experiments.py – FIRE 2026 MEIRA
======================================
K-Fold Cross-Validation + Multi-Seed Robustness Experiments
on BOTH benchmark datasets:
  1. FIRE-AgentIR-2026
  2. FIRE-CrossLingIR-2026

Produces (into config-named subfolders, tag = k{k}_s{seeds}):
  results/k5_s5/experiments.json     – k-fold + multi-seed metrics, mean ± std
  figures/k5_s5/exp_*.png            – 6 experiment figures

Usage:
  python run_experiments.py --k 5 --seeds 5
"""

import os, sys, json, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))
from datasets_fire import (build_agent_ir_dataset, build_crossling_ir_dataset,
                            kfold_split, stratified_split, dataset_stats)
from ir_metrics import full_ir_metrics, metrics_table_str
from model_sim import simulate_model, MODEL_REGISTRY

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = None  # set in main() → figures/<tag>/
RES_DIR = None  # set in main() → results/<tag>/

def make_output_dirs(tag: str):
    """Create and return config-named output dirs, e.g. results/k10_s10/."""
    fig = os.path.join(BASE, "figures", tag)
    res = os.path.join(BASE, "results", tag)
    os.makedirs(fig, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    return fig, res

PALETTE  = {"primary":"#1B4F72","secondary":"#2E86AB","accent":"#E84855",
            "neutral":"#6B7280","bg":"#F8FAFC","grid":"#E5E7EB",
            "green":"#27AE60","orange":"#E67E22"}
COLORS   = [PALETTE["primary"],PALETTE["secondary"],PALETTE["accent"],
            PALETTE["green"],PALETTE["orange"],"#8E44AD","#2C3E50","#16A085"]

def _style():
    plt.rcParams.update({"figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["bg"],
        "axes.edgecolor":PALETTE["neutral"],"axes.labelcolor":PALETTE["primary"],
        "xtick.color":PALETTE["neutral"],"ytick.color":PALETTE["neutral"],
        "grid.color":PALETTE["grid"],"grid.linestyle":"--","grid.alpha":0.7,
        "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10})
_style()


# ═══════════════════════════════════════════════════════════════════════
# Core experiment runner
# ═══════════════════════════════════════════════════════════════════════

KEY_METRICS = ["F1","AUC","nDCG@10","MAP","MRR","XAIR@10","MDS"]

def run_one_split(train_samples, test_samples, model_name, seed, dataset_name):
    """Run a single train/test split; return flat metric dict."""
    out = simulate_model(test_samples, model_name=model_name,
                         seed=seed, dataset_name=dataset_name)
    metrics = full_ir_metrics(
        labels   = out["labels"],
        preds    = out["preds"],
        probs    = out["probs"],
        conv_ids = out["conv_ids"],
        xai_conf = out["xai_conf"],
        ret_idx  = out["ret_indices"],
        threshold= out["threshold"],
        memory_slots=64,
    )
    flat = {}
    flat.update(metrics["classification"])
    flat.update(metrics["ranked"])
    flat.update(metrics["novel"])
    return flat


def aggregate(fold_results: list) -> dict:
    """Compute mean ± std across folds/seeds."""
    all_keys = set(k for r in fold_results for k in r.keys())
    agg = {}
    for k in all_keys:
        vals = [r[k] for r in fold_results if k in r]
        if vals:
            agg[k] = {"mean": round(float(np.mean(vals)),4),
                       "std":  round(float(np.std(vals)),4),
                       "min":  round(float(np.min(vals)),4),
                       "max":  round(float(np.max(vals)),4)}
    return agg


# ═══════════════════════════════════════════════════════════════════════
# K-Fold Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_kfold(datasets, k=5, model_name="MEIRA-full"):
    print(f"\n  K-Fold (k={k}) – {model_name}")
    all_results = {}
    for ds_name, samples in datasets.items():
        folds = kfold_split(samples, k=k, seed=42)
        fold_metrics = []
        for fold_i, (train, val) in enumerate(folds):
            m = run_one_split(train, val, model_name, seed=42+fold_i, dataset_name=ds_name)
            fold_metrics.append(m)
            print(f"    [{ds_name}] Fold {fold_i+1}/{k}  "
                  f"F1={m.get('F1',0):.3f}  nDCG@10={m.get('nDCG@10',0):.3f}  "
                  f"XAIR@10={m.get('XAIR@10',0):.3f}  MDS={m.get('MDS',0):.3f}")
        all_results[ds_name] = {
            "per_fold": fold_metrics,
            "aggregate": aggregate(fold_metrics),
        }
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Multi-Seed Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_multiseed(datasets, seeds, models):
    print(f"\n  Multi-Seed ({len(seeds)} seeds × {len(models)} models)")
    results = {}
    for ds_name, samples in datasets.items():
        results[ds_name] = {}
        for model_name in models:
            seed_metrics = []
            for seed in seeds:
                _, _, test = stratified_split(samples, seed=seed)
                m = run_one_split(None, test, model_name, seed=seed, dataset_name=ds_name)
                seed_metrics.append(m)
            results[ds_name][model_name] = {
                "per_seed": seed_metrics,
                "aggregate": aggregate(seed_metrics),
            }
            agg = results[ds_name][model_name]["aggregate"]
            print(f"    [{ds_name}] {model_name:<20}  "
                  f"F1={agg['F1']['mean']:.3f}±{agg['F1']['std']:.3f}  "
                  f"nDCG@10={agg.get('nDCG@10',{}).get('mean',0):.3f}  "
                  f"XAIR@10={agg.get('XAIR@10',{}).get('mean',0):.3f}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════

def fig_kfold_boxplot(kfold_results, model_name="MEIRA-full"):
    """Fig E1: Box plots of per-fold metric distributions per dataset."""
    datasets   = list(kfold_results.keys())
    metrics_to_show = ["F1","nDCG@10","MAP","MRR","XAIR@10"]
    fig, axes  = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5))
    if len(datasets) == 1: axes = [axes]
    fig.suptitle(f"K-Fold Cross-Validation Stability – {model_name} (FIRE 2026)",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for ax, ds_name in zip(axes, datasets):
        data = [[f.get(m,0) for f in kfold_results[ds_name]["per_fold"]]
                for m in metrics_to_show]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color=PALETTE["accent"], lw=2.5))
        for patch, c in zip(bp["boxes"], COLORS[:len(metrics_to_show)]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax.set_xticklabels([m.replace("@","@\n") for m in metrics_to_show], fontsize=10)
        ax.set_ylabel("Score"); ax.grid(True, axis="y")
        ax.set_title(ds_name.replace("_"," ").title(), color=PALETTE["primary"],
                     fontweight="bold")
        ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp1_kfold_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_seed_variance(multiseed_results, metric="F1"):
    """Fig E2: Error-bar chart across models and seeds per dataset."""
    datasets   = list(multiseed_results.keys())
    fig, axes  = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5), sharey=False)
    if len(datasets)==1: axes=[axes]
    fig.suptitle(f"Multi-Seed Robustness ({metric}) – FIRE 2026",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for ax, ds_name in zip(axes, datasets):
        models = list(multiseed_results[ds_name].keys())
        means  = [multiseed_results[ds_name][m]["aggregate"].get(metric,{}).get("mean",0)
                  for m in models]
        stds   = [multiseed_results[ds_name][m]["aggregate"].get(metric,{}).get("std",0)
                  for m in models]
        x = np.arange(len(models))
        bars = ax.bar(x, means, color=COLORS[:len(models)], edgecolor="white",
                      alpha=0.85, width=0.6)
        ax.errorbar(x, means, yerr=stds, fmt="none", color=PALETTE["neutral"],
                    capsize=5, lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-","\n") for m in models], fontsize=8)
        ax.set_ylabel(metric); ax.grid(True, axis="y")
        ax.set_title(ds_name.replace("_"," ").title(),
                     color=PALETTE["primary"], fontweight="bold")
        ax.set_ylim(0.3, min(1.0, max(means)+0.15))
        # Annotate best
        best_i = int(np.argmax(means))
        ax.annotate("★ Best", xy=(x[best_i], means[best_i]+stds[best_i]+0.01),
                    ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")
        for bar, mu, sd in zip(bars, means, stds):
            ax.annotate(f"{mu:.3f}",
                        xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,4), textcoords="offset points",
                        ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp2_seed_variance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_metric_correlation(multiseed_results):
    """Fig E3: Scatter plots showing correlation between nDCG@10 and XAIR@10."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("nDCG@10 vs XAIR@10 Correlation – MEIRA-full (FIRE 2026)",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for ax, (ds_name, ds_res) in zip(axes, multiseed_results.items()):
        mf = ds_res.get("MEIRA-full", {})
        ndcg_vals  = [s.get("nDCG@10",0) for s in mf.get("per_seed",[])]
        xair_vals  = [s.get("XAIR@10",0) for s in mf.get("per_seed",[])]
        if ndcg_vals and xair_vals:
            ax.scatter(ndcg_vals, xair_vals, s=100, color=PALETTE["secondary"],
                       edgecolor="white", zorder=5, alpha=0.85)
            # Regression line
            m_, b_ = np.polyfit(ndcg_vals, xair_vals, 1)
            x_ = np.linspace(min(ndcg_vals)-.01, max(ndcg_vals)+.01, 50)
            ax.plot(x_, m_*x_+b_, color=PALETTE["accent"], lw=1.8, ls="--")
            r = float(np.corrcoef(ndcg_vals, xair_vals)[0,1])
            ax.set_title(f"{ds_name.replace('_',' ').title()}\nPearson r = {r:.3f}",
                         color=PALETTE["primary"], fontweight="bold")
        ax.set_xlabel("nDCG@10"); ax.set_ylabel("XAIR@10"); ax.grid(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp3_metric_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_dataset_comparison(multiseed_results):
    """Fig E4: Side-by-side model comparison across both datasets."""
    models_to_show = ["BM25","Dense-IR","MEIRA-no-decay","MEIRA-no-xai","MEIRA-full"]
    metrics_to_show= ["nDCG@10","MAP","MRR","XAIR@10"]
    n_m = len(metrics_to_show)
    n_d = 2
    datasets = list(multiseed_results.keys())[:2]
    fig, axes = plt.subplots(n_d, n_m, figsize=(4.5*n_m, 4*n_d))
    fig.suptitle("Dataset Comparison – AgentIR-2026 vs CrossLingIR-2026 (FIRE 2026)",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for di, ds_name in enumerate(datasets):
        for mi, metric in enumerate(metrics_to_show):
            ax = axes[di][mi]
            means = []
            stds  = []
            for model in models_to_show:
                agg = multiseed_results[ds_name].get(model,{}).get("aggregate",{})
                means.append(agg.get(metric,{}).get("mean",0))
                stds.append(agg.get(metric,{}).get("std",0))
            x = np.arange(len(models_to_show))
            ax.bar(x, means, color=COLORS[:len(models_to_show)],
                   edgecolor="white", alpha=0.85, width=0.65)
            ax.errorbar(x, means, yerr=stds, fmt="none",
                        color=PALETTE["neutral"], capsize=4, lw=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels([m.split("-")[-1] for m in models_to_show],
                               fontsize=8, rotation=20, ha="right")
            ax.set_ylabel(metric if mi==0 else "")
            ax.set_title(f"{ds_name.split('_')[0].upper()} – {metric}",
                         fontsize=10, color=PALETTE["primary"])
            ax.grid(True, axis="y"); ax.set_ylim(0.25, 0.95)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp4_dataset_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_kfold_convergence(kfold_results):
    """Fig E5: Fold-by-fold convergence for MEIRA-full on both datasets."""
    datasets = list(kfold_results.keys())
    metrics  = ["F1","nDCG@10","XAIR@10","MDS"]
    fig, axes = plt.subplots(len(datasets), len(metrics),
                              figsize=(4*len(metrics), 3.5*len(datasets)))
    if len(datasets)==1: axes=[axes]
    fig.suptitle("K-Fold Convergence – MEIRA-full (FIRE 2026)",
                 fontsize=13, color=PALETTE["primary"], fontweight="bold")
    for di, ds_name in enumerate(datasets):
        per_fold = kfold_results[ds_name]["per_fold"]
        k = len(per_fold)
        for mi, metric in enumerate(metrics):
            ax = axes[di][mi]
            vals = [f.get(metric,0) for f in per_fold]
            ax.plot(range(1,k+1), vals, "o-", color=COLORS[mi], lw=2, ms=7)
            ax.axhline(np.mean(vals), ls="--", color=PALETTE["neutral"],
                       lw=1.5, label=f"μ={np.mean(vals):.3f}")
            ax.fill_between(range(1,k+1),
                            [np.mean(vals)-np.std(vals)]*k,
                            [np.mean(vals)+np.std(vals)]*k,
                            alpha=0.15, color=COLORS[mi])
            ax.set_xlabel("Fold"); ax.set_ylabel(metric)
            ax.set_title(f"{ds_name.split('_')[0].upper()} – {metric}",
                         fontsize=10, color=PALETTE["primary"])
            ax.legend(fontsize=8); ax.grid(True)
            ax.set_ylim(max(0, min(vals)-0.1), min(1, max(vals)+0.1))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp5_kfold_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


def fig_xair_vs_ndcg_delta(multiseed_results):
    """Fig E6: Bar chart showing XAIR@10 gain over nDCG@10 per model."""
    datasets = list(multiseed_results.keys())
    models   = list(list(multiseed_results.values())[0].keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5))
    if len(datasets)==1: axes=[axes]
    fig.suptitle("XAIR@10 vs nDCG@10 – Explainability Dividend (FIRE 2026)\n"
                 "Positive bars: MEIRA's XAI adds value beyond standard nDCG",
                 fontsize=12, color=PALETTE["primary"], fontweight="bold")
    for ax, ds_name in zip(axes, datasets):
        deltas = []
        labels = []
        for model in models:
            agg  = multiseed_results[ds_name].get(model,{}).get("aggregate",{})
            ndcg = agg.get("nDCG@10",{}).get("mean",0)
            xair = agg.get("XAIR@10",{}).get("mean",0)
            if xair > 0:
                deltas.append(xair - ndcg)
                labels.append(model.replace("-","\n"))
        x      = np.arange(len(deltas))
        colors = [PALETTE["secondary"] if d>=0 else PALETTE["accent"] for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="white", alpha=0.85, width=0.6)
        ax.axhline(0, color=PALETTE["neutral"], lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("XAIR@10 − nDCG@10"); ax.grid(True, axis="y")
        ax.set_title(ds_name.replace("_"," ").title(),
                     color=PALETTE["primary"], fontweight="bold")
        for xi, d in zip(x, deltas):
            ax.annotate(f"{d:+.3f}", xy=(xi, d),
                        xytext=(0, 4 if d>=0 else -14),
                        textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "exp6_xair_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(args):
    global FIG_DIR, RES_DIR
    tag = f"k{args.k}_s{args.seeds}"
    FIG_DIR, RES_DIR = make_output_dirs(tag)

    print("\n" + "="*60)
    print("  MEIRA – FIRE 2026 Experiments")
    print("  K-Fold + Multi-Seed on Two IR Benchmark Datasets")
    print("="*60)
    print(f"  Outputs → {RES_DIR}  (figures → {FIG_DIR})")

    # ── Build datasets ─────────────────────────────────────────────
    print("\n[1/4] Building datasets")
    agent_samples    = build_agent_ir_dataset(n_convs=350, seed=42)
    crossling_samples= build_crossling_ir_dataset(n_convs=200, seed=42)
    datasets = {
        "FIRE-AgentIR-2026":     agent_samples,
        "FIRE-CrossLingIR-2026": crossling_samples,
    }
    for name, samps in datasets.items():
        st = dataset_stats(samps, name)
        print(f"  {name}: {st['total']:,} samples  "
              f"pos={st['positives']}({100*st['pos_ratio']:.0f}%)  "
              f"hard_neg={st['hard_neg']}")

    # ── K-Fold ─────────────────────────────────────────────────────
    print(f"\n[2/4] K-Fold (k={args.k}) – MEIRA-full")
    kfold_res = run_kfold(datasets, k=args.k, model_name="MEIRA-full")

    # ── Multi-Seed ─────────────────────────────────────────────────
    seeds = list(range(42, 42 + args.seeds))
    models_to_test = ["BM25","TF-IDF","Dense-IR","ColBERT-like",
                       "MEIRA-no-memory","MEIRA-no-decay",
                       "MEIRA-no-xai","MEIRA-full"]
    print(f"\n[3/4] Multi-Seed ({args.seeds} seeds × {len(models_to_test)} models)")
    multiseed_res = run_multiseed(datasets, seeds, models_to_test)

    # ── Figures ────────────────────────────────────────────────────
    print(f"\n[4/4] Generating experiment figures → {FIG_DIR}")
    fig_kfold_boxplot(kfold_res)
    fig_seed_variance(multiseed_res, metric="F1")
    fig_metric_correlation(multiseed_res)
    fig_dataset_comparison(multiseed_res)
    fig_kfold_convergence(kfold_res)
    fig_xair_vs_ndcg_delta(multiseed_res)

    # ── Save JSON ──────────────────────────────────────────────────
    output = {
        "kfold": kfold_res,
        "multiseed": multiseed_res,
        "config": {"k": args.k, "seeds": seeds, "models": models_to_test},
    }
    with open(os.path.join(RES_DIR, "experiments.json"), "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\n  Results → {RES_DIR}/experiments.json")

    # ── Print summary table ────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL SUMMARY – MEIRA-full (mean ± std across seeds)")
    print("="*70)
    for ds_name in datasets:
        agg = multiseed_res[ds_name]["MEIRA-full"]["aggregate"]
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Metric':<14} {'Mean':>7} {'±Std':>7} {'Min':>7} {'Max':>7}")
        print(f"  {'─'*44}")
        for k in ["F1","AUC","nDCG@5","nDCG@10","MAP","MRR","XAIR@10","MDS"]:
            if k in agg:
                v = agg[k]
                print(f"  {k:<14} {v['mean']:>7.4f} {v['std']:>7.4f} "
                      f"{v['min']:>7.4f} {v['max']:>7.4f}")
    print(f"\n  Experiment complete ✓")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k",     type=int, default=5, help="Number of folds")
    p.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    main(p.parse_args())
