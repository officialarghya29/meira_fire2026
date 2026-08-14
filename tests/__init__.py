"""Test suite for the MEIRA FIRE 2026 pipeline.

Run from the repo root with:

    python -m unittest discover -s tests -v

or with the venv interpreter:

    .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys

# make repo-root modules (ir_metrics, multi_correction, datasets_fire,
# model_sim, run_significance, md2tex, ...) importable from the tests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
