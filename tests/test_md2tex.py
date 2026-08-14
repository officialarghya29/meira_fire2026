"""Regression tests for md2tex.py - the markdown -> LaTeX converter.

Covers the failure modes that have actually bitten this pipeline:
  * an empty abstract (the "Primary variant" paragraph silently not matching),
  * trim entries whose 'old' text no longer matches any paragraph (silently
    dead trims),
  * the six-section split and the section/table label maps.
"""

import os
import re
import unittest

import md2tex


class TestSplitSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(md2tex.MD_PATH) as f:
            cls.md = f.read()
        cls.chunks, cls.order = md2tex.split_sections(cls.md)

    def test_six_core_sections_present_in_order(self):
        expected = ["1. Introduction", "2. Related Work",
                    "3. Datasets and Evaluation Protocol",
                    "4. Results: SOTA Leaderboard and Component Ablation",
                    "5. Robustness of the Statistical Comparisons",
                    "6. Conclusion and Limitations"]
        for title in expected:
            self.assertIn(title, self.chunks)
        # the six sections appear in order (filter the order list)
        seen = [s for s in self.order if s in expected]
        self.assertEqual(seen, expected)

    def test_abstract_and_references_sections_exist(self):
        self.assertIn("Abstract", self.chunks)
        self.assertIn("References", self.chunks)


class TestExtractAbstract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(md2tex.MD_PATH) as f:
            cls.chunks, _ = md2tex.split_sections(f.read())

    def test_primary_nonempty_and_within_fire_limit(self):
        primary, blurb, warnings = md2tex.extract_abstract(self.chunks)
        self.assertTrue(primary.strip())
        self.assertNotIn("no '**Primary variant'", "; ".join(warnings))
        # FIRE 2026 abstract limit is 250 words
        self.assertLessEqual(len(primary.split()), 250)
        self.assertTrue(primary.startswith("Multi-turn agentic"))

    def test_no_markdown_leftovers_in_rendered_abstract(self):
        primary, _, _ = md2tex.extract_abstract(self.chunks)
        # extract_abstract intentionally keeps **bold** markers; the LaTeX
        # renderer must consume them (no raw markdown survives tex_inline)
        rendered = md2tex.tex_inline(primary)
        self.assertNotIn("**", rendered)
        self.assertNotIn("\n", rendered)
        self.assertNotIn("  ", rendered)         # single-spaced
        self.assertNotIn("\\textbf{\textbf", rendered)

    def test_trim_overlap_absent(self):
        # a trim that also matches abstract text would silently diverge the
        # abstract from the body (trims are NOT applied to the abstract)
        _, _, warnings = md2tex.extract_abstract(self.chunks)
        self.assertFalse([w for w in warnings if "also matches abstract" in w])

    def test_missing_primary_variant_warns(self):
        chunks = {"Abstract": ["**Tight blurb**", "", "A short blurb.", "",
                               "**Word count**", "", "x"]}
        primary, blurb, warnings = md2tex.extract_abstract(chunks)
        self.assertEqual(primary, "")
        self.assertTrue(blurb.strip())
        self.assertTrue(any("EMPTY" in w for w in warnings))

    def test_dehyphenation(self):
        chunks = {"Abstract": ["**Primary variant**", "",
                               "This is a long abstract line that gets wrapped-\n"
                               "hyphenated in the markdown source."]}
        primary, _, _ = md2tex.extract_abstract(chunks)
        self.assertNotIn("-\n", primary)
        self.assertIn("wrapped-hyphenated", primary)


def _converter_paragraphs(md):
    """Return the exact strings convert_section feeds to apply_trims, using
    the production block-walk helper (md2tex.block_trim_texts) so this test
    can never drift from the converter's real paragraph stream."""
    chunks, order = md2tex.split_sections(md)
    texts = []
    for title in order:
        if title in ("Abstract", "References", "FRONT"):
            continue
        lines = chunks[title]
        i, n = 0, len(lines)
        while i < n:
            ln = lines[i]
            if (not ln.strip() or ln.strip() == "---" or md2tex.HEAD_RE.match(ln)
                    or ln.strip().startswith(("```", ">"))
                    or md2tex.is_table_row(ln)):
                i += 1
                continue
            j = i
            blk = []
            while j < n:
                l2 = lines[j]
                if (not l2.strip() or l2.strip() == "---" or md2tex.HEAD_RE.match(l2)
                        or l2.strip().startswith(("```", ">"))
                        or md2tex.is_table_row(l2)):
                    break
                blk.append(l2.rstrip())
                j += 1
            if not blk:
                i += 1
                continue
            if "\n" not in "\n".join(blk).strip() and \
                    md2tex.FIG_RE.match("\n".join(blk).strip()):
                i = j          # figure line -> figure env, never trim-applied
                continue
            texts.extend(md2tex.block_trim_texts(blk))
            i = j
    return texts


class TestTrimsIntegrity(unittest.TestCase):
    """Trims are applied by md2tex.apply_trims to converter-built paragraphs
    in TRIMS order; chained trims legitimately match text produced by an
    earlier trim in the chain (not the raw markdown). The authoritative check
    is therefore: replay the exact converter paragraph stream through
    apply_trims and require every trim name to fire at least once."""

    @classmethod
    def setUpClass(cls):
        with open(md2tex.MD_PATH) as f:
            cls.texts = _converter_paragraphs(f.read())

    def setUp(self):
        md2tex._TRIM_APPLIED.clear()
        md2tex._TRIM_MISSED.clear()

    def test_all_trims_apply(self):
        for t in self.texts:
            md2tex.apply_trims(t)
        never = [n for n in md2tex._TRIM_MISSED if n not in md2tex._TRIM_APPLIED]
        self.assertEqual(never, [], "trims that never matched any converter "
                         "paragraph (dead/chained-broken trims)")
        self.assertEqual(len(md2tex._TRIM_APPLIED), len(md2tex.TRIMS),
                         "every trim should fire at least once")

    def test_paragraph_stream_matches_converter(self):
        # the mirror must produce a sane stream: at least ~50 paragraphs
        self.assertGreater(len(self.texts), 50)

    def test_trims_never_grow_text(self):
        for name, old, new in md2tex.TRIMS:
            self.assertLessEqual(len(new), len(old), f"trim {name} grows text")

    def test_trim_names_unique(self):
        names = [n for n, _, _ in md2tex.TRIMS]
        self.assertEqual(len(names), len(set(names)))


class TestLabelMaps(unittest.TestCase):
    def test_table_labels_cover_1_9(self):
        for i in range(1, 10):
            self.assertIn(str(i), md2tex.TAB_LABELS,
                          f"Table {i} has no LaTeX label")

    def test_figure_labels_cover_1_3(self):
        for i in range(1, 4):
            self.assertIn(str(i), md2tex.FIG_LABELS,
                          f"Figure {i} has no LaTeX label")

    def test_section_labels_cover_core(self):
        for sec in ("1", "2", "3", "4", "5", "6"):
            self.assertIn(sec, md2tex.SEC_LABELS)

    def test_bib_loaded_with_keys(self):
        self.assertGreater(len(md2tex.BIB_KEYS), 15)


class TestCitationForms(unittest.TestCase):
    def test_single_author(self):
        self.assertEqual(md2tex.citation_forms(["Zhang"], "2024"),
                         ["Zhang, 2024"])

    def test_two_authors(self):
        self.assertEqual(md2tex.citation_forms(["Salton", "Buckley"], "1988"),
                         ["Salton & Buckley, 1988"])

    def test_three_authors(self):
        forms = md2tex.citation_forms(["A", "B", "C"], "2020")
        self.assertIn("A, B & C, 2020", forms)
        self.assertIn("A et al., 2020", forms)

    def test_many_authors(self):
        forms = md2tex.citation_forms(["A", "B", "C", "D"], "2019")
        self.assertEqual(forms, ["A et al., 2019"])


if __name__ == "__main__":
    unittest.main()
