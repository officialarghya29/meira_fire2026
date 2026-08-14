"""Tests for ir_metrics.py - every ranked/novel metric is checked against a
hand-computed example, not just for smoke.
"""

import unittest

import numpy as np

import ir_metrics as im


class TestNDCG(unittest.TestCase):
    def test_hand_computed(self):
        # labels [1,0,1] ranked by scores [3,2,1] -> ranked [1,0,1]
        # dcg  = 1/log2(2) + 0 + 1/log2(4) = 1.5
        # ideal= 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
        expected = 1.5 / (1 + 1 / np.log2(3))
        val = im.ndcg_at_k([[1, 0, 1]], [[3, 2, 1]], k=3)
        self.assertAlmostEqual(val, round(expected, 4), places=4)

    def test_perfect_ranking_is_one(self):
        self.assertEqual(im.ndcg_at_k([[1, 1, 0]], [[3, 2, 1]], k=3), 1.0)

    def test_no_relevant_returns_zero(self):
        self.assertEqual(im.ndcg_at_k([[0, 0, 0]], [[3, 2, 1]], k=3), 0.0)

    def test_cutoff_honoured(self):
        # k=1 only looks at the top hit
        self.assertEqual(im.ndcg_at_k([[0, 1]], [[2, 1]], k=1), 0.0)   # top hit irrelevant
        self.assertEqual(im.ndcg_at_k([[0, 1]], [[1, 2]], k=1), 1.0)   # top hit relevant
        self.assertEqual(im.ndcg_at_k([[1, 0]], [[2, 1]], k=1), 1.0)


class TestMAP(unittest.TestCase):
    def test_hand_computed(self):
        # labels [1,0,1,0], scores [4,3,2,1] -> ranked [1,0,1,0]
        # ap = (1/1 + 2/3) / 2 = 0.8333
        self.assertEqual(im.mean_average_precision([[1, 0, 1, 0]], [[4, 3, 2, 1]]), 0.8333)

    def test_map_at_k(self):
        # labels [1,0,1,0], scores [1,2,3,4] -> top-2 [0,1]
        # ap = (1/2) / 2 = 0.25
        self.assertEqual(im.mean_average_precision([[1, 0, 1, 0]], [[1, 2, 3, 4]], k=2), 0.25)


class TestMRR(unittest.TestCase):
    def test_first_relevant_at_rank_3(self):
        # labels [0,1,0], scores [3,1,2] -> order [0,2,1] -> relevant at rank 3
        self.assertEqual(im.mean_reciprocal_rank([[0, 1, 0]], [[3, 1, 2]]), round(1 / 3, 4))

    def test_no_relevant_is_zero(self):
        self.assertEqual(im.mean_reciprocal_rank([[0, 0]], [[2, 1]]), 0.0)


class TestPrecision(unittest.TestCase):
    def test_p_at_k(self):
        # labels [1,0,1,0], scores [1,2,3,4] -> top-2 [0,1] -> P@2 = 0.5
        self.assertEqual(im.precision_at_k([[1, 0, 1, 0]], [[1, 2, 3, 4]], k=2), 0.5)

    def test_r_precision(self):
        # R = 2 relevant; top-2 by score: idx0 (lbl1), idx2 (lbl1) -> 1.0
        self.assertEqual(im.r_precision([[1, 0, 1, 0]], [[4, 1, 3, 2]]), 1.0)


class TestXAIR(unittest.TestCase):
    def test_hand_computed(self):
        # labels [1,0,1], scores [3,2,1], xai [0.8,0.2,0.6], k=3, w=0.25
        ndcg = 1.5 / (1 + 1 / np.log2(3))      # 0.9197
        xai_part = (0.8 + 0.6) / 2             # 0.7 (relevant docs only)
        expected = 0.75 * ndcg + 0.25 * xai_part
        val = im.xair_at_k([[1, 0, 1]], [[3, 2, 1]], [[0.8, 0.2, 0.6]], k=3, xai_weight=0.25)
        self.assertAlmostEqual(val, round(expected, 4), places=4)

    def test_zero_xai_lowers_score(self):
        plain = im.ndcg_at_k([[1, 0, 1]], [[3, 2, 1]], k=3)
        penalised = im.xair_at_k([[1, 0, 1]], [[3, 2, 1]], [[0.0, 0.0, 0.0]], k=3)
        self.assertLess(penalised, plain)

    def test_high_xai_matches_ndcg_weight(self):
        # full XAI confidence on relevant docs -> xair == ndcg at w=0
        v = im.xair_at_k([[1, 0, 1]], [[3, 2, 1]], [[1.0, 0.0, 1.0]], k=3, xai_weight=0.0)
        self.assertEqual(v, im.ndcg_at_k([[1, 0, 1]], [[3, 2, 1]], k=3))


class TestMDS(unittest.TestCase):
    def test_hand_computed(self):
        # 4 distinct slots used out of 64
        self.assertEqual(im.memory_diversity_score([[0, 1], [1, 2], [2, 3]], 64), 0.0625)

    def test_bounds(self):
        self.assertEqual(im.memory_diversity_score([[5]] * 10, 64), round(1 / 64, 4))


class TestClassification(unittest.TestCase):
    def test_hand_computed(self):
        labels = np.array([1, 0, 1, 1, 0])
        preds = np.array([1, 0, 1, 0, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
        m = im.classification_metrics(labels, preds, probs)
        # tp=2 (idx0,idx2), fp=0, fn=1 (idx3) -> P=1.0, R=2/3, F1=2PR/(P+R)=0.8
        self.assertEqual(m["F1"], 0.8)
        self.assertEqual(m["Precision"], 1.0)
        self.assertEqual(m["Recall"], 0.6667)
        self.assertEqual(m["Accuracy"], 0.8)       # 4/5 correct
        self.assertEqual(m["AUC"], 1.0)            # all pos probs above all neg
        self.assertEqual(m["AP"], 1.0)

    def test_auc_single_class_zero(self):
        labels = np.array([1, 1, 1])
        m = im.classification_metrics(labels, None, np.array([0.9, 0.8, 0.7]))
        self.assertEqual(m["AUC"], 0.0)


class TestQueryPools(unittest.TestCase):
    def test_grouping(self):
        conv = np.array([0, 0, 1, 1, 1])
        lbl = np.array([1, 0, 1, 0, 1])
        sc = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        ql, qs, qx, qr = im.build_query_pools(conv, lbl, sc)
        self.assertEqual(len(ql), 2)
        self.assertEqual(ql[0], [1, 0])
        self.assertEqual(qs[1], [0.8, 0.2, 0.7])
        self.assertEqual(qx, [])      # no xai supplied
        self.assertEqual(qr, [])

    def test_optional_arrays(self):
        conv = np.array([0, 1])
        lbl = np.array([1, 0])
        sc = np.array([0.9, 0.1])
        xai = np.array([0.7, 0.2])
        ret = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        ql, qs, qx, qr = im.build_query_pools(conv, lbl, sc, xai, ret)
        self.assertEqual(qx[0], [0.7])
        self.assertEqual(qr[1], [[6, 7, 8, 9, 10]])


class TestFullMetrics(unittest.TestCase):
    def test_smoke_and_keys(self):
        n = 40
        conv = np.repeat(np.arange(10), 4)
        lbl = np.tile([1, 0, 0, 0], 10)
        probs = np.random.RandomState(0).uniform(0, 1, n)
        xai = np.random.RandomState(1).uniform(0, 1, n)
        ret = np.random.RandomState(2).randint(0, 64, (n, 5))
        m = im.full_ir_metrics(lbl, (probs >= 0.5).astype(int), probs, conv,
                               xai_conf=xai, ret_idx=ret, memory_slots=64)
        for key in ("nDCG@5", "nDCG@10", "MAP", "MAP@10", "MRR", "R-Prec", "P@5", "P@10"):
            self.assertIn(key, m["ranked"])
        self.assertIn("XAIR@10", m["novel"])
        self.assertIn("MDS", m["novel"])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in m["ranked"].values()))

    def test_table_str(self):
        m = {"classification": {"F1": 0.5}, "ranked": {"MRR": 0.3}, "novel": {}}
        s = im.metrics_table_str(m, "test")
        self.assertIn("test Evaluation Results", s)
        self.assertIn("0.5000", s)


class TestTemporalDecay(unittest.TestCase):
    def test_keys_and_ranges(self):
        out = im.temporal_decay_effectiveness(np.array([0.5, 0.5]),
                                              np.array([0.6, 0.6]),
                                              np.array([1, 1]))
        self.assertIn("delta_nDCG", out)
        self.assertIn("p_value", out)
        self.assertIn("significant", out)


if __name__ == "__main__":
    unittest.main()
