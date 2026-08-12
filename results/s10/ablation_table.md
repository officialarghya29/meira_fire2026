# Ablation Study — MEIRA (FIRE 2026)

> Simulated evaluation-harness run (see `model_sim.py`). Replace `simulate_model()` with real trained-model inference before citing these numbers in a submission.

## FIRE-AgentIR-2026

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (Full model (memory + XAI + decay)) | 0.826±0.015 | 0.957±0.006 | 0.969±0.008 | 0.954±0.011 | 0.644±0.020 | 0.894±0.007 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.716±0.014 | 0.893±0.011 | 0.930±0.014 | 0.900±0.019 | 0.612±0.021 | 0.859±0.012 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.777±0.013 | 0.932±0.008 | 0.951±0.010 | 0.929±0.013 | 0.628±0.020 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.748±0.012 | 0.914±0.009 | 0.941±0.011 | 0.915±0.015 | 0.620±0.019 | 0.869±0.011 | 1.000±0.000 |

**Δ vs full (component contribution):**

| Removed component | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| − Episodic memory | +0.110 | +0.064 | +0.038 | +0.055 | +0.032 | +0.035 | +1.000 |
| − XAI attribution head | +0.049 | +0.025 | +0.018 | +0.025 | +0.015 | +0.894 | +0.000 |
| − Temporal decay | +0.078 | +0.043 | +0.028 | +0.040 | +0.024 | +0.025 | +0.000 |

## FIRE-CrossLingIR-2026

| Variant | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| **MEIRA-full** (Full model (memory + XAI + decay)) | 0.780±0.030 | 0.923±0.015 | 0.962±0.008 | 0.947±0.011 | 0.529±0.012 | 0.886±0.010 | 1.000±0.000 |
| **MEIRA-no-memory** (− Episodic memory) | 0.655±0.030 | 0.834±0.024 | 0.922±0.015 | 0.891±0.021 | 0.501±0.017 | 0.850±0.017 | 0.000±0.000 |
| **MEIRA-no-xai** (− XAI attribution head) | 0.726±0.029 | 0.888±0.019 | 0.945±0.008 | 0.922±0.011 | 0.517±0.014 | 0.000±0.000 | 1.000±0.000 |
| **MEIRA-no-decay** (− Temporal decay) | 0.694±0.027 | 0.866±0.021 | 0.937±0.011 | 0.911±0.015 | 0.511±0.015 | 0.865±0.011 | 1.000±0.000 |

**Δ vs full (component contribution):**

| Removed component | F1 | AUC | nDCG@10 | MAP | MRR | XAIR@10 | MDS |
|---|---|---|---|---|---|---|---|
| − Episodic memory | +0.124 | +0.089 | +0.040 | +0.056 | +0.028 | +0.036 | +1.000 |
| − XAI attribution head | +0.054 | +0.036 | +0.017 | +0.025 | +0.012 | +0.886 | +0.000 |
| − Temporal decay | +0.086 | +0.058 | +0.025 | +0.036 | +0.018 | +0.021 | +0.000 |
