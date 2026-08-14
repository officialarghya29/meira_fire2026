"""Tests for run_significance.py helpers - series extraction, paired
t-tests, pairwise matrices with multiplicity correction, and formatting.
These test the pure functions; the CLI itself is exercised end-to-end
by the pipeline verification step.
"""

import os
import shutil
import unittest

import numpy as np

import run_significance as rs


class TestPerModelSeries(unittest.TestCase):
    def test_drops_none(self):
        results = {"M1": {"per_seed": [{"nDCG@10": 0.5}, {"nDCG@10": None},
                                       {"nDCG@10": 0.6}]}}
        s = rs.per_model_series(results, ["M1"], "nDCG@10")
        self.assertEqual(s["M1"], [0.5, 0.6])

    def test_unknown_model_empty(self):
        s = rs.per_model_series({}, ["M1"], "F1")
        self.assertEqual(s["M1"], [])


class TestTtest(unittest.TestCase):
    def test_identical_series(self):
        self.assertEqual(rs.ttest([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), (0.0, 1.0))

    def test_short_series(self):
        self.assertEqual(rs.ttest([0.5], [0.5]), (0.0, 1.0))

    def test_separated_series_significant(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0.55, 0.02, 10)
        b = rng.normal(0.45, 0.02, 10)
        t, p = rs.ttest(a, b)
        self.assertLess(p, 0.001)
        self.assertGreater(t, 0)

    def test_reversed_pair_flips_t(self):
        rng = np.random.RandomState(1)
        a = rng.normal(0.55, 0.02, 10)
        b = rng.normal(0.45, 0.02, 10)
        t_ab, _ = rs.ttest(a, b)
        t_ba, _ = rs.ttest(b, a)
        self.assertAlmostEqual(t_ab, -t_ba, places=10)


class TestPairwiseMatrices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(3)
        cls.series = {
            "A": rng.normal(0.60, 0.02, 10).tolist(),
            "B": rng.normal(0.50, 0.02, 10).tolist(),
            "C": rng.normal(0.55, 0.02, 10).tolist(),
        }
        cls.models = ["A", "B", "C"]

    def test_shapes_and_symmetry(self):
        raws, P, T = rs.pairwise_matrices(self.series, self.models, "holm")
        n = 3
        self.assertEqual(raws.shape, (n, n))
        np.testing.assert_allclose(raws, raws.T)
        np.testing.assert_allclose(P, P.T)
        np.testing.assert_allclose(T, -T.T)
        np.testing.assert_allclose(np.diag(raws), 1.0)
        np.testing.assert_allclose(np.diag(P), 1.0)
        np.testing.assert_allclose(np.diag(T), 0.0)

    def test_holm_does_not_decrease_p(self):
        raws, P, _ = rs.pairwise_matrices(self.series, self.models, "holm")
        self.assertTrue(np.all(P >= raws - 1e-12))

    def test_none_returns_raw(self):
        raws, P, _ = rs.pairwise_matrices(self.series, self.models, "none")
        np.testing.assert_allclose(P, raws)

    def test_significant_vs_insignificant(self):
        rng = np.random.RandomState(4)
        series = {
            "X": rng.normal(0.6, 0.02, 10).tolist(),
            "Y": rng.normal(0.6, 0.02, 10).tolist(),   # tied -> not significant
            "Z": rng.normal(0.4, 0.02, 10).tolist(),
        }
        _, P, _ = rs.pairwise_matrices(series, ["X", "Y", "Z"], "holm")
        iX, iY, iZ = 0, 1, 2
        self.assertLess(P[iX, iZ], 0.05)
        self.assertGreater(P[iX, iY], 0.05)


class TestFormatting(unittest.TestCase):
    def test_fmt_p(self):
        self.assertEqual(rs.fmt_p(0.00001), "<0.0001***")
        self.assertEqual(rs.fmt_p(0.5), "0.5000")
        self.assertIn("***", rs.fmt_p(0.0005))
        self.assertIn("**", rs.fmt_p(0.005))
        self.assertIn("*", rs.fmt_p(0.03))
        self.assertNotIn("*", rs.fmt_p(0.2))

    def test_stat_desc(self):
        self.assertEqual(rs.stat_desc([0.123, 0.129]), "0.126±0.003")
        self.assertEqual(rs.stat_desc([]), "—")

    def test_out_name(self):
        self.assertEqual(rs.out_name("significance_matrix_", "nDCG@10", "holm"),
                         "significance_matrix_nDCG@10_holm")
        self.assertEqual(rs.out_name("significance_matrix_", "nDCG@10", "none"),
                         "significance_matrix_nDCG@10")


class TestOutputDirs(unittest.TestCase):
    def setUp(self):
        self._paths = [os.path.join(rs.BASE, "results", "k1_s1"),
                       os.path.join(rs.BASE, "figures", "k1_s1")]
        self._pre_existing = [os.path.exists(p) for p in self._paths]

    def tearDown(self):
        # only remove dirs this test created (never touch pre-existing data)
        for p, existed in zip(self._paths, self._pre_existing):
            if not existed and os.path.isdir(p):
                shutil.rmtree(p)

    def test_dirs_inside_project(self):
        fig, res = rs.make_output_dirs("k1_s1")
        base = os.path.abspath(rs.BASE)
        self.assertTrue(os.path.abspath(fig).startswith(base))
        self.assertTrue(os.path.abspath(res).startswith(base))
        self.assertTrue(os.path.isdir(fig))
        self.assertTrue(os.path.isdir(res))


if __name__ == "__main__":
    unittest.main()
