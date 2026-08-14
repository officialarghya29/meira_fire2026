"""Tests for multi_correction.py - Bonferroni and Holm-Bonferroni
step-down corrections used by the significance pipeline.
"""

import unittest

import numpy as np

import multi_correction as mc


class TestBonferroni(unittest.TestCase):
    def test_formula(self):
        p = np.array([0.01, 0.1, 0.5])
        np.testing.assert_allclose(mc.bonferroni(p), np.minimum(1.0, p * 3))

    def test_caps_at_one(self):
        p = np.array([0.6, 0.9])
        np.testing.assert_allclose(mc.bonferroni(p), [1.0, 1.0])

    def test_empty(self):
        np.testing.assert_allclose(mc.bonferroni([]), [])


class TestHolm(unittest.TestCase):
    def test_hand_computed(self):
        # p = [0.01, 0.04, 0.20]; m = 3
        # rank1: 0.01*3 = 0.03; rank2: max(0.03, 0.04*2) = 0.08
        # rank3: max(0.08, 0.20*1) = 0.20
        np.testing.assert_allclose(mc.holm(np.array([0.01, 0.04, 0.20])),
                                   [0.03, 0.08, 0.20])

    def test_monotone_in_rank(self):
        rng = np.random.RandomState(0)
        p = rng.uniform(0, 1, 30)
        adj = mc.holm(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(adj[order]) >= -1e-12))

    def test_never_exceeds_bonferroni(self):
        rng = np.random.RandomState(1)
        p = rng.uniform(0, 1, 25)
        self.assertTrue(np.all(mc.holm(p) <= mc.bonferroni(p) + 1e-12))

    def test_bounds(self):
        p = np.array([0.9, 0.8, 0.7])
        adj = mc.holm(p)
        self.assertTrue(np.all(adj >= p - 1e-12))
        self.assertTrue(np.all(adj <= 1.0 + 1e-12))

    def test_permutation_invariant(self):
        p1 = np.array([0.02, 0.01, 0.10])
        p2 = np.array([0.10, 0.02, 0.01])
        np.testing.assert_allclose(np.sort(mc.holm(p1)), np.sort(mc.holm(p2)))


class TestCorrect(unittest.TestCase):
    def test_none_returns_raw(self):
        p = [0.1, 0.2]
        np.testing.assert_allclose(mc.correct(p, "none"), p)
        np.testing.assert_allclose(mc.correct(p, None), p)

    def test_dispatches(self):
        p = [0.01, 0.04, 0.20]
        np.testing.assert_allclose(mc.correct(p, "holm"), mc.holm(p))
        np.testing.assert_allclose(mc.correct(p, "bonferroni"), mc.bonferroni(p))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            mc.correct([0.1], "fdr")


if __name__ == "__main__":
    unittest.main()
