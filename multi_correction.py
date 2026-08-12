"""
multi_correction.py – FIRE 2026 MEIRA
=====================================
Multiple-comparison corrections for families of pairwise p-values
(Bonferroni and Holm-Bonferroni step-down).

Used by run_significance.py and compare_metric_orderings.py to adjust
p-values for the family of ALL pairwise model comparisons within one
dataset × metric (m = n(n−1)/2 = 28 tests for the 8-model suite), so that
reported significance survives multiplicity.
"""

import numpy as np


def bonferroni(p_values):
    """Bonferroni: p_adj = min(1, p · m)."""
    p = np.asarray(p_values, dtype=float)
    return np.minimum(1.0, p * len(p))


def holm(p_values):
    """Holm-Bonferroni step-down.

    Sort p_(1) ≤ … ≤ p_(m); adjusted p_(k) = max_{j ≤ k} min(1, p_(j) · (m − j + 1)).
    More powerful than Bonferroni while still controlling FWER at α.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for k in range(m):
        running = max(running, min(1.0, p[order[k]] * (m - k)))
        adj[order[k]] = running
    return adj


def correct(p_values, method="holm"):
    """Apply the requested correction; 'none' returns the raw p-values."""
    if method is None or method == "none":
        return np.asarray(p_values, dtype=float)
    if method == "bonferroni":
        return bonferroni(p_values)
    if method == "holm":
        return holm(p_values)
    raise ValueError(f"Unknown correction method: {method}")
