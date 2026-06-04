"""Tests for :mod:`onnx_light.backend.inference_coverage`."""

import unittest

from onnx_light.backend.inference_coverage import (
    InferenceCaseReport,
    InferenceCoverageReport,
    ValueComparison,
    ValueShape,
    compute_inference_coverage,
    render_rst_case,
    render_rst_report,
    render_rst_summary,
)
from onnx_light.ext_test_case import ExtTestCase


class TestInferenceCoverage(ExtTestCase):
    """Validates the inference coverage report helpers."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute_inference_coverage()

    def test_report_is_populated(self):
        self.assertIsInstance(self.report, InferenceCoverageReport)
        # The C++ test registry ships at least three ``"inference"``-tagged
        # cases (``add_concat_reshape``, ``nonzero_chain_*``,
        # ``shape_identity_unsqueeze``).
        self.assertGreaterEqual(self.report.total, 3)
        for case in self.report.cases:
            self.assertIsInstance(case, InferenceCaseReport)
            self.assertTrue(case.name.startswith("test_"))
            self.assertIn("flowchart", case.mermaid)

    def test_known_case_present(self):
        names = {c.name for c in self.report.cases}
        self.assertIn("test_cc_shape_inference_add_concat_reshape", names)

    def test_value_comparison_match_semantics(self):
        cmp = ValueComparison(
            name="x",
            role="input",
            expected=ValueShape(elem_type=1, shape=[1, 3]),
            computed=ValueShape(elem_type=1, shape=[1, 3]),
        )
        self.assertTrue(cmp.match)
        cmp_diff_shape = ValueComparison(
            name="x",
            role="input",
            expected=ValueShape(elem_type=1, shape=[1, 3]),
            computed=ValueShape(elem_type=1, shape=[1, 4]),
        )
        self.assertFalse(cmp_diff_shape.match)
        cmp_no_expected = ValueComparison(
            name="x", role="value_info", expected=None, computed=None
        )
        # No expected shape ⇒ always considered a match.
        self.assertTrue(cmp_no_expected.match)
        cmp_missing_computed = ValueComparison(
            name="x",
            role="value_info",
            expected=ValueShape(elem_type=1, shape=[1, 3]),
            computed=None,
        )
        self.assertFalse(cmp_missing_computed.match)

    def test_render_rst_summary(self):
        text = render_rst_summary(self.report)
        self.assertIn("list-table", text)
        self.assertIn("onnx_optim shape inference", text)
        self.assertIn(str(self.report.total), text)

    def test_render_rst_case(self):
        case = self.report.cases[0]
        text = render_rst_case(case)
        self.assertIn(case.name, text)
        self.assertIn("runmermaid", text)
        # Either the comparison table or the warning block must be present.
        self.assertTrue(("list-table" in text) or ("warning" in text))

    def test_render_rst_report_contains_every_case(self):
        text = render_rst_report(self.report)
        for case in self.report.cases:
            self.assertIn(case.name, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
