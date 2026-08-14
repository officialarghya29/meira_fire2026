"""Tests for datasets_fire.py - deterministic dataset builders, tokeniser,
and the stratified/k-fold split utilities.
"""

import unittest

from datasets_fire import (build_agent_ir_dataset, build_crossling_ir_dataset,
                           stratified_split, kfold_split, dataset_stats,
                           tokenise, IRSample)


class TestTokenise(unittest.TestCase):
    def test_lengths_and_special_tokens(self):
        ids, mask = tokenise("retrieval query document", 32)
        self.assertEqual(len(ids), 32)
        self.assertEqual(len(mask), 32)
        self.assertEqual(ids[0], 1)      # CLS
        self.assertEqual(ids[-1], 0)     # padding
        self.assertEqual(mask[-1], 0)
        self.assertEqual(sum(mask), sum(1 for i in ids if i != 0))

    def test_deterministic_vocab(self):
        ids1, _ = tokenise("episodic memory decay", 16)
        ids2, _ = tokenise("episodic memory decay", 16)
        self.assertEqual(ids1, ids2)


class TestAgentDataset(unittest.TestCase):
    def test_deterministic_and_size(self):
        d1 = build_agent_ir_dataset(n_convs=20, seed=7)
        d2 = build_agent_ir_dataset(n_convs=20, seed=7)
        self.assertEqual(len(d1), 20 * 6 * 4)     # convs x turns x (1 pos + 3 neg)
        for a, b in zip(d1, d2):
            self.assertEqual(a.input_ids, b.input_ids)
            self.assertEqual(a.label, b.label)
            self.assertEqual(a.conv_id, b.conv_id)

    def test_stats(self):
        d = build_agent_ir_dataset(n_convs=10, seed=3)
        st = dataset_stats(d)
        self.assertEqual(st["total"], 10 * 6 * 4)
        self.assertGreater(st["pos_ratio"], 0.15)
        self.assertLess(st["pos_ratio"], 0.35)
        self.assertGreater(st["hard_neg"], 0)
        self.assertTrue(all(s.dataset == "agent" for s in d))
        self.assertTrue(all(0 <= s.turn < 6 for s in d))

    def test_sample_interface(self):
        d = build_agent_ir_dataset(n_convs=2, seed=1)
        s = d[0]
        self.assertIsInstance(s, IRSample)
        self.assertEqual(len(s.input_ids), len(s.attention_mask))


class TestCrosslingDataset(unittest.TestCase):
    def test_deterministic_and_size(self):
        d1 = build_crossling_ir_dataset(n_convs=10, seed=8)
        d2 = build_crossling_ir_dataset(n_convs=10, seed=8)
        self.assertEqual(len(d1), 10 * 4 * 4)
        for a, b in zip(d1, d2):
            self.assertEqual(a.input_ids, b.input_ids)
        self.assertTrue(all(s.dataset == "crossling" for s in d1))

    def test_label_balance(self):
        st = dataset_stats(build_crossling_ir_dataset(n_convs=10, seed=2))
        self.assertGreater(st["pos_ratio"], 0.15)
        self.assertLess(st["pos_ratio"], 0.35)


class TestSplits(unittest.TestCase):
    def test_stratified_ratios_and_disjoint(self):
        d = build_agent_ir_dataset(n_convs=30, seed=5)
        tr, vl, te = stratified_split(d, seed=9)
        self.assertAlmostEqual(len(tr) / len(d), 0.70, delta=0.05)
        self.assertAlmostEqual(len(vl) / len(d), 0.15, delta=0.05)
        self.assertAlmostEqual(len(te) / len(d), 0.15, delta=0.05)
        ids_tr = {id(s) for s in tr}
        ids_vl = {id(s) for s in vl}
        ids_te = {id(s) for s in te}
        self.assertEqual(ids_tr & ids_vl, set())
        self.assertEqual(ids_tr & ids_te, set())
        self.assertEqual(ids_vl & ids_te, set())

    def test_stratified_preserves_class_balance(self):
        d = build_agent_ir_dataset(n_convs=30, seed=5)
        tr, _, te = stratified_split(d, seed=9)
        r_tr = sum(s.label == 1 for s in tr) / len(tr)
        r_te = sum(s.label == 1 for s in te) / len(te)
        self.assertAlmostEqual(r_tr, r_te, delta=0.03)

    def test_kfold_disjoint_and_exhaustive(self):
        d = build_agent_ir_dataset(n_convs=40, seed=2)
        k = 5
        folds = kfold_split(d, k=k, seed=11)
        self.assertEqual(len(folds), k)
        n = len(d)
        for tr, vl in folds:
            # per-class n//k chunking drops each class's remainder (<= k-1
            # samples per class), so coverage is n - drop, drop <= 2*(k-1)
            self.assertGreaterEqual(len(tr) + len(vl), n - 2 * (k - 1))
            self.assertLessEqual(len(tr) + len(vl), n)
            self.assertEqual({id(s) for s in tr} & {id(s) for s in vl}, set())
        # a sample appears as validation in at most one fold
        val_ids = [frozenset(id(s) for s in vl) for _, vl in folds]
        for i in range(len(val_ids)):
            for j in range(i + 1, len(val_ids)):
                self.assertEqual(val_ids[i] & val_ids[j], frozenset())


if __name__ == "__main__":
    unittest.main()
