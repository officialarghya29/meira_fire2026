"""Tests for model_sim.py - the numpy model simulation: determinism, class
separation, and the memory/XAI capability flags.
"""

import unittest

import numpy as np

from datasets_fire import build_agent_ir_dataset, build_crossling_ir_dataset
from model_sim import simulate_model, MODEL_REGISTRY


class TestRegistry(unittest.TestCase):
    def test_eight_models(self):
        self.assertEqual(len(MODEL_REGISTRY), 8)
        for name in ("BM25", "TF-IDF", "Dense-IR", "ColBERT-like",
                     "MEIRA-no-memory", "MEIRA-no-decay", "MEIRA-no-xai",
                     "MEIRA-full"):
            self.assertIn(name, MODEL_REGISTRY)

    def test_full_capabilities(self):
        cfg = MODEL_REGISTRY["MEIRA-full"]
        self.assertTrue(cfg.has_memory and cfg.has_xai and cfg.has_decay)

    def test_baselines_lack_memory_xai(self):
        cfg = MODEL_REGISTRY["BM25"]
        self.assertFalse(cfg.has_memory or cfg.has_xai)


class TestSimulateModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = build_agent_ir_dataset(n_convs=15, seed=11)
        cls.cross = build_crossling_ir_dataset(n_convs=8, seed=12)

    def test_deterministic(self):
        a = simulate_model(self.agent, "MEIRA-full", seed=42)
        b = simulate_model(self.agent, "MEIRA-full", seed=42)
        np.testing.assert_array_equal(a["probs"], b["probs"])
        np.testing.assert_array_equal(a["preds"], b["preds"])
        np.testing.assert_array_equal(a["xai_conf"], b["xai_conf"])

    def test_different_seeds_differ(self):
        a = simulate_model(self.agent, "MEIRA-full", seed=1)
        b = simulate_model(self.agent, "MEIRA-full", seed=2)
        self.assertFalse(np.array_equal(a["probs"], b["probs"]))

    def test_positives_beat_negatives(self):
        out = simulate_model(self.agent, "ColBERT-like", seed=3)
        pos = out["probs"][out["labels"] == 1]
        neg = out["probs"][out["labels"] == 0]
        self.assertGreater(pos.mean(), neg.mean())
        # margin should be clearly positive (well-separated simulation)
        self.assertGreater(pos.mean() - neg.mean(), 0.15)

    def test_meira_full_outranks_baseline(self):
        full = simulate_model(self.agent, "MEIRA-full", seed=5)["probs"]
        bm25 = simulate_model(self.agent, "BM25", seed=5)["probs"]
        self.assertGreater(full[full_labels(self.agent) == 1].mean(),
                           bm25[full_labels(self.agent) == 1].mean())

    def test_memory_xai_presence(self):
        full = simulate_model(self.agent, "MEIRA-full", seed=4)
        self.assertIsNotNone(full["xai_conf"])
        self.assertIsNotNone(full["ret_indices"])
        self.assertEqual(full["ret_indices"].shape[1], 5)
        bm25 = simulate_model(self.agent, "BM25", seed=4)
        self.assertIsNone(bm25["xai_conf"])
        self.assertIsNone(bm25["ret_indices"])
        nm = simulate_model(self.agent, "MEIRA-no-memory", seed=4)
        self.assertIsNone(nm["ret_indices"])

    def test_xai_correlates_with_correctness(self):
        out = simulate_model(self.agent, "MEIRA-full", seed=6)
        correct = out["preds"] == out["labels"]
        wrong = out["xai_conf"][~correct]
        self.assertGreater(len(wrong), 0, "simulation should have some errors")
        self.assertGreater(out["xai_conf"][correct].mean(), wrong.mean())

    def test_memory_coverage(self):
        out = simulate_model(self.cross, "MEIRA-full", seed=6, memory_slots=64)
        flat = {int(x) for row in out["ret_indices"] for x in row}
        self.assertLessEqual(len(flat), 64)
        # 64*0.38 = 24 hot slots + random draws -> broad coverage, not a single slot
        self.assertGreater(len(flat), 10)

    def test_output_shapes(self):
        out = simulate_model(self.agent, "MEIRA-no-decay", seed=8)
        n = len(self.agent)
        self.assertEqual(out["labels"].shape, (n,))
        self.assertEqual(out["probs"].shape, (n,))
        self.assertEqual(out["conv_ids"].shape, (n,))
        self.assertEqual(out["turns"].shape, (n,))
        self.assertTrue(0.0 <= out["threshold"] <= 1.0)
        self.assertTrue(np.all((out["probs"] >= 0.01) & (out["probs"] <= 0.99)))

    def test_crossling_penalty_applied(self):
        # cross-lingual is harder: positive scores shifted down vs agent data
        agent_pos = simulate_model(self.agent, "Dense-IR", seed=9)["probs"]
        cross_pos = simulate_model(self.cross, "Dense-IR", seed=9)["probs"]
        self.assertLess(cross_pos[full_labels(self.cross) == 1].mean(),
                        agent_pos[full_labels(self.agent) == 1].mean())


def full_labels(samples):
    return np.array([s.label for s in samples])


if __name__ == "__main__":
    unittest.main()
