"""
ir_metrics.py – ACM SIGIR / FIRE Standard IR Evaluation Metrics
================================================================
Implements the full suite of IR evaluation metrics required for
FIRE 2026 and ACM SIGIR conference submissions:

  Standard Classification:
    F1, Precision, Recall, Accuracy, ROC-AUC, Average Precision

  Standard IR (ranked list):
    nDCG@K         – Normalised Discounted Cumulative Gain
    MAP            – Mean Average Precision
    MAP@K          – MAP at cutoff K
    MRR            – Mean Reciprocal Rank
    R-Precision    – Precision at rank R (number of relevant docs)
    Precision@K    – P@1, P@5, P@10

  New Metric (MEIRA contribution):
    XAIR@K         – eXplainability-Adjusted IR score at K
                     Penalises correct retrievals that have low
                     XAI attribution confidence (not explainable).
                     Novel metric proposed in this paper.
    MDS            – Memory Diversity Score
                     Fraction of distinct memory slots used per query.
                     Measures retrieval diversity across the memory bank.

All functions operate on numpy arrays for efficiency.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, average_precision_score,
                              accuracy_score)


# ═══════════════════════════════════════════════════════════════════════
# 1. Standard classification metrics
# ═══════════════════════════════════════════════════════════════════════

def classification_metrics(labels: np.ndarray, preds: np.ndarray,
                            probs: np.ndarray, threshold: float = 0.5) -> dict:
    if preds is None:
        preds = (probs >= threshold).astype(int)
    return {
        "F1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "Precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "Recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "Accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "AUC":       round(float(roc_auc_score(labels, probs))
                           if len(set(labels.tolist())) > 1 else 0.0, 4),
        "AP":        round(float(average_precision_score(labels, probs)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Ranked-list IR metrics (core FIRE/SIGIR metrics)
# ═══════════════════════════════════════════════════════════════════════

def _dcg(rels: List[int], k: int = None) -> float:
    """Discounted Cumulative Gain for a ranked list of relevance labels."""
    if k is not None:
        rels = rels[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def _ideal_dcg(rels: List[int], k: int = None) -> float:
    sorted_rels = sorted(rels, reverse=True)
    return _dcg(sorted_rels, k)


def ndcg_at_k(query_labels: List[List[int]],
              query_scores: List[List[float]],
              k: int = 10) -> float:
    """
    nDCG@K averaged over queries.
    query_labels : list of relevance label lists per query
    query_scores : list of predicted score lists per query (higher = more relevant)
    """
    ndcgs = []
    for labels, scores in zip(query_labels, query_scores):
        order = np.argsort(scores)[::-1]
        ranked = [labels[i] for i in order]
        ideal  = _ideal_dcg(labels, k)
        if ideal == 0:
            continue
        ndcgs.append(_dcg(ranked, k) / ideal)
    return round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4)


def mean_average_precision(query_labels: List[List[int]],
                           query_scores: List[List[float]],
                           k: int = None) -> float:
    """MAP or MAP@K averaged over queries."""
    aps = []
    for labels, scores in zip(query_labels, query_scores):
        order = np.argsort(scores)[::-1]
        if k: order = order[:k]
        ranked = [labels[i] for i in order]
        n_rel  = sum(labels)
        if n_rel == 0:
            continue
        hits, ap = 0, 0.0
        for i, r in enumerate(ranked):
            if r == 1:
                hits += 1
                ap   += hits / (i + 1)
        aps.append(ap / n_rel)
    return round(float(np.mean(aps)) if aps else 0.0, 4)


def mean_reciprocal_rank(query_labels: List[List[int]],
                         query_scores: List[List[float]]) -> float:
    """MRR – Mean Reciprocal Rank of the first relevant document."""
    rrs = []
    for labels, scores in zip(query_labels, query_scores):
        order = np.argsort(scores)[::-1]
        for rank, i in enumerate(order, 1):
            if labels[i] == 1:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return round(float(np.mean(rrs)) if rrs else 0.0, 4)


def r_precision(query_labels: List[List[int]],
                query_scores: List[List[float]]) -> float:
    """R-Precision: Precision at rank R = number of relevant docs."""
    rps = []
    for labels, scores in zip(query_labels, query_scores):
        R = sum(labels)
        if R == 0:
            continue
        order  = np.argsort(scores)[::-1][:R]
        ranked = [labels[i] for i in order]
        rps.append(sum(ranked) / R)
    return round(float(np.mean(rps)) if rps else 0.0, 4)


def precision_at_k(query_labels: List[List[int]],
                   query_scores: List[List[float]],
                   k: int = 10) -> float:
    """Precision@K averaged over queries."""
    pks = []
    for labels, scores in zip(query_labels, query_scores):
        order  = np.argsort(scores)[::-1][:k]
        ranked = [labels[i] for i in order]
        pks.append(sum(ranked) / k)
    return round(float(np.mean(pks)) if pks else 0.0, 4)


# ═══════════════════════════════════════════════════════════════════════
# 3. New Metrics (MEIRA paper contributions)
# ═══════════════════════════════════════════════════════════════════════

def xair_at_k(query_labels: List[List[int]],
              query_scores: List[List[float]],
              query_xai:   List[List[float]],
              k: int = 10,
              xai_weight: float = 0.25) -> float:
    """
    XAIR@K – eXplainability-Adjusted IR score at K (proposed metric).

    Motivation: A retrieval system that returns the correct document
    but cannot explain WHY (low XAI confidence) is less trustworthy
    than one that is both correct AND explainable.

    Formula:
        XAIR@K = (1 - xai_weight) * nDCG@K
                 + xai_weight * mean(xai_conf for top-K relevant docs)

    where xai_conf ∈ [0,1] is the normalised XAI attribution confidence
    for each retrieved document (max token attribution score).

    This penalises black-box correct retrievals and rewards
    interpretable correct retrievals.

    Args:
        query_xai : list of XAI confidence score lists per query (one per doc)
        xai_weight: weight for XAI component (default 0.25)
    """
    xair_scores = []
    for labels, scores, xai in zip(query_labels, query_scores, query_xai):
        order  = np.argsort(scores)[::-1][:k]
        ranked_labels = [labels[i] for i in order]
        ranked_xai    = [xai[i]    for i in order]

        ideal = _ideal_dcg(labels, k)
        if ideal == 0:
            continue
        ndcg_part = _dcg(ranked_labels, k) / ideal

        # XAI confidence for relevant docs only
        rel_xai = [ranked_xai[i] for i, r in enumerate(ranked_labels) if r == 1]
        xai_part = float(np.mean(rel_xai)) if rel_xai else 0.0

        xair_scores.append((1 - xai_weight) * ndcg_part + xai_weight * xai_part)

    return round(float(np.mean(xair_scores)) if xair_scores else 0.0, 4)


def memory_diversity_score(retrieved_indices: List[List[int]],
                            memory_slots: int = 64) -> float:
    """
    MDS – Memory Diversity Score (proposed metric).

    Measures what fraction of memory slots are actually used across
    all queries. Low MDS → memory bank under-utilised (mode collapse).
    High MDS → rich episodic coverage.

    MDS = |unique slots used| / total_slots

    Ranges [0, 1]. Target for a healthy memory bank: MDS > 0.3.
    """
    used = set()
    for indices in retrieved_indices:
        used.update(indices)
    return round(len(used) / max(1, memory_slots), 4)


def temporal_decay_effectiveness(scores_before_decay: np.ndarray,
                                  scores_after_decay: np.ndarray,
                                  labels: np.ndarray) -> dict:
    """
    Measures how much temporal decay improves ranking quality.
    Compares nDCG before and after applying the decay multiplier.
    Returns delta_nDCG and a p-value estimate.
    """
    from scipy import stats
    # Treat each sample as a 1-element query
    ndcg_before = [_dcg([l], 1) / max(_ideal_dcg([l], 1), 1e-9)
                    for l, s in zip(labels, scores_before_decay)]
    ndcg_after  = [_dcg([l], 1) / max(_ideal_dcg([l], 1), 1e-9)
                    for l, s in zip(labels, scores_after_decay)]
    t_stat, p_val = stats.ttest_rel(ndcg_after, ndcg_before)
    delta = float(np.mean(ndcg_after)) - float(np.mean(ndcg_before))
    return {
        "delta_nDCG": round(delta, 4),
        "t_stat":     round(float(t_stat), 4),
        "p_value":    round(float(p_val), 4),
        "significant": p_val < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Build ranked query pools from flat sample lists
# ═══════════════════════════════════════════════════════════════════════

def build_query_pools(conv_ids: np.ndarray,
                      labels: np.ndarray,
                      scores: np.ndarray,
                      xai_scores: Optional[np.ndarray] = None,
                      ret_indices: Optional[np.ndarray] = None
                      ) -> Tuple[List,List,List,List]:
    """
    Groups flat predictions back into per-query pools for ranked metrics.

    Returns:
        q_labels, q_scores, q_xai, q_ret_idx
    """
    pools: Dict[int, dict] = {}
    for i, (cid, lbl, sc) in enumerate(zip(conv_ids, labels, scores)):
        cid = int(cid)
        if cid not in pools:
            pools[cid] = {"labels":[], "scores":[], "xai":[], "ret":[]}
        pools[cid]["labels"].append(int(lbl))
        pools[cid]["scores"].append(float(sc))
        if xai_scores is not None:
            pools[cid]["xai"].append(float(xai_scores[i]))
        if ret_indices is not None:
            pools[cid]["ret"].append(list(ret_indices[i]) if hasattr(ret_indices[i],'__iter__')
                                     else [int(ret_indices[i])])

    q_labels = [v["labels"] for v in pools.values()]
    q_scores = [v["scores"] for v in pools.values()]
    q_xai    = [v["xai"]   for v in pools.values()] if xai_scores is not None else []
    q_ret    = [v["ret"]   for v in pools.values()] if ret_indices is not None else []
    return q_labels, q_scores, q_xai, q_ret


def full_ir_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray,  conv_ids: np.ndarray,
                    xai_conf: Optional[np.ndarray] = None,
                    ret_idx:  Optional[np.ndarray] = None,
                    threshold: float = 0.5,
                    memory_slots: int = 64) -> dict:
    """
    Compute the complete metric suite used in the FIRE 2026 paper.
    """
    clf = classification_metrics(labels, preds, probs, threshold)

    q_lbl, q_sc, q_xai, q_ret = build_query_pools(
        conv_ids, labels, probs, xai_conf, ret_idx)

    ranked = {
        "nDCG@5":   ndcg_at_k(q_lbl, q_sc, k=5),
        "nDCG@10":  ndcg_at_k(q_lbl, q_sc, k=10),
        "MAP":      mean_average_precision(q_lbl, q_sc),
        "MAP@10":   mean_average_precision(q_lbl, q_sc, k=10),
        "MRR":      mean_reciprocal_rank(q_lbl, q_sc),
        "R-Prec":   r_precision(q_lbl, q_sc),
        "P@5":      precision_at_k(q_lbl, q_sc, k=5),
        "P@10":     precision_at_k(q_lbl, q_sc, k=10),
    }

    novel = {}
    if q_xai and all(len(x) > 0 for x in q_xai):
        novel["XAIR@10"] = xair_at_k(q_lbl, q_sc, q_xai, k=10)
        novel["XAIR@5"]  = xair_at_k(q_lbl, q_sc, q_xai, k=5)

    if q_ret and ret_idx is not None:
        flat_ret = [idx for qr in q_ret for idx in qr]
        novel["MDS"] = memory_diversity_score(flat_ret, memory_slots)

    return {"classification": clf, "ranked": ranked, "novel": novel}


def metrics_table_str(metrics: dict, dataset_name: str = "") -> str:
    """Pretty-print metrics as a table string."""
    lines = [f"\n{'═'*54}",
             f"  {dataset_name} Evaluation Results",
             f"{'═'*54}"]

    for group, vals in metrics.items():
        if not vals: continue
        lines.append(f"\n  ── {group.upper()} ──")
        for k, v in vals.items():
            lines.append(f"    {k:<14} {v:.4f}")

    lines.append(f"{'═'*54}")
    return "\n".join(lines)
