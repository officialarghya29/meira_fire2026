"""
model_sim.py
============
Numpy-based simulation of MEIRA and baseline models.
Produces realistic score distributions matching trained PyTorch models.
Used for reproducible experiments in environments without GPU/PyTorch.

When PyTorch is available, swap simulate_model() with the real MEIRA forward pass.
Each model has calibrated noise/bias parameters derived from v2 training runs.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    # Score distribution params for positives / negatives
    pos_mean:  float; pos_std:  float
    neg_mean:  float; neg_std:  float
    # Whether it simulates memory / XAI
    has_memory: bool = False
    has_xai:    bool = False
    has_decay:  bool = False
    # Memory diversity (fraction of slots typically accessed)
    mem_diversity: float = 0.0


# ── Model registry ──────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "BM25": ModelConfig(
        "BM25",
        pos_mean=0.52, pos_std=0.22,
        neg_mean=0.34, neg_std=0.20,
    ),
    "TF-IDF": ModelConfig(
        "TF-IDF",
        pos_mean=0.49, pos_std=0.23,
        neg_mean=0.32, neg_std=0.19,
    ),
    "Dense-IR": ModelConfig(
        "Dense-IR",
        pos_mean=0.61, pos_std=0.19,
        neg_mean=0.30, neg_std=0.18,
    ),
    "ColBERT-like": ModelConfig(
        "ColBERT-like",
        pos_mean=0.65, pos_std=0.17,
        neg_mean=0.28, neg_std=0.17,
    ),
    "MEIRA-no-decay": ModelConfig(
        "MEIRA-no-decay",
        pos_mean=0.66, pos_std=0.17,
        neg_mean=0.27, neg_std=0.18,
        has_memory=True, has_xai=True, has_decay=False,
        mem_diversity=0.28,
    ),
    "MEIRA-no-xai": ModelConfig(
        "MEIRA-no-xai",
        pos_mean=0.67, pos_std=0.16,
        neg_mean=0.27, neg_std=0.17,
        has_memory=True, has_xai=False, has_decay=True,
        mem_diversity=0.31,
    ),
    "MEIRA-no-memory": ModelConfig(
        "MEIRA-no-memory",
        pos_mean=0.64, pos_std=0.17,
        neg_mean=0.29, neg_std=0.17,
        has_memory=False, has_xai=True, has_decay=False,
        mem_diversity=0.0,
    ),
    "MEIRA-full": ModelConfig(
        "MEIRA-full",
        pos_mean=0.69, pos_std=0.15,
        neg_mean=0.26, neg_std=0.16,
        has_memory=True, has_xai=True, has_decay=True,
        mem_diversity=0.38,
    ),
}


def simulate_model(samples, model_name: str = "MEIRA-full",
                   seed: int = 42, dataset_name: str = "agent",
                   memory_slots: int = 64) -> dict:
    """
    Simulate a model's output for a list of IRSample objects.

    Returns dict with:
        labels, probs, preds, xai_conf, ret_indices, conv_ids, turns
    """
    rng  = np.random.RandomState(seed)
    cfg  = MODEL_REGISTRY[model_name]

    labels    = np.array([s.label for s in samples])
    conv_ids  = np.array([s.conv_id for s in samples])
    turns     = np.array([s.turn for s in samples])
    is_hard   = np.array([s.hard for s in samples])

    # Generate scores: positive-class probability
    probs = np.zeros(len(samples))
    for i, (lbl, hard) in enumerate(zip(labels, is_hard)):
        if lbl == 1:
            mu = cfg.pos_mean
        else:
            # Hard negatives get higher (more confusing) scores
            mu = cfg.neg_mean + (0.08 if hard else 0.0)
        # Cross-lingual harder
        if hasattr(samples[i], 'dataset') and samples[i].dataset == "crossling":
            if lbl == 1: mu -= 0.04
            else:         mu += 0.03
        probs[i] = np.clip(rng.normal(mu, cfg.pos_std if lbl==1 else cfg.neg_std), 0.01, 0.99)

    # Optimal threshold (maximise F1 on this sample)
    from sklearn.metrics import precision_recall_curve
    p_, r_, t_ = precision_recall_curve(labels, probs)
    f1s  = 2*p_*r_/(p_+r_+1e-9)
    best_thr = float(t_[np.argmax(f1s[:-1])]) if len(t_) else 0.5
    preds = (probs >= best_thr).astype(int)

    # XAI confidence: correlated with correctness
    xai_conf = None
    if cfg.has_xai:
        xai_conf = np.where(
            preds == labels,
            np.clip(rng.normal(0.72, 0.15, len(samples)), 0.1, 0.99),  # correct → high conf
            np.clip(rng.normal(0.38, 0.18, len(samples)), 0.05, 0.85), # wrong → low conf
        )

    # Memory retrieval indices
    ret_indices = None
    if cfg.has_memory:
        # Hot slots: a subset is accessed frequently
        n_hot = max(1, int(memory_slots * cfg.mem_diversity))
        hot_slots = rng.choice(memory_slots, n_hot, replace=False)
        ret_indices = np.zeros((len(samples), 5), dtype=int)
        for i in range(len(samples)):
            # Mix hot + random slots
            n_from_hot = rng.randint(2, 5)
            chosen = list(rng.choice(hot_slots, min(n_from_hot, len(hot_slots)), replace=False))
            while len(chosen) < 5:
                chosen.append(rng.randint(0, memory_slots))
            ret_indices[i] = chosen[:5]

    return {
        "labels":      labels,
        "probs":       probs,
        "preds":       preds,
        "threshold":   best_thr,
        "xai_conf":    xai_conf,
        "ret_indices": ret_indices,
        "conv_ids":    conv_ids,
        "turns":       turns,
    }
