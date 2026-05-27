"""Tests for :mod:`onnx_light.backend.runtime_coverage`."""

import unittest

from onnx_light.backend.runtime_coverage import (
    DomainSummary,
    RuntimeCoverageReport,
    TestCaseStatus,
    compute_runtime_coverage,
    render_rst_domain_sections,
    render_rst_domain_summary,
    render_rst_domain_tabs,
    render_rst_summary,
    render_rst_table_for_domain,
)
from onnx_light.ext_test_case import ExtTestCase


class TestRuntimeCoverage(ExtTestCase):
    """Validates the runtime coverage report helpers."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute_runtime_coverage()

    def test_report_is_populated(self):
        self.assertIsInstance(self.report, RuntimeCoverageReport)
        self.assertGreater(len(self.report.statuses), 0)
        self.assertEqual(
            self.report.overall.total,
            sum(s.total for s in self.report.summaries.values()),
        )
        self.assertEqual(self.report.overall.total, len(self.report.statuses))

    def test_status_fields(self):
        status = self.report.statuses[0]
        self.assertIsInstance(status, TestCaseStatus)
        self.assertIsInstance(status.name, str)
        self.assertIsInstance(status.op_type, str)
        self.assertIsInstance(status.domain, str)
        self.assertIsInstance(status.static_shape, bool)
        self.assertIsInstance(status.dynamic_shapes, bool)
        # onnxruntime_cpu is either None (cannot run) or a non-negative float.
        if status.onnxruntime_cpu is not None:
            self.assertGreaterEqual(status.onnxruntime_cpu, 0.0)

    def test_static_and_dynamic_shape_pass_overall(self):
        # Static and dynamic shape inference are expected to succeed on every
        # collected test case; regressions in either should fail this test.
        self.assertEqual(
            self.report.overall.static_shape_ok,
            self.report.overall.total,
        )
        self.assertEqual(
            self.report.overall.dynamic_shapes_ok,
            self.report.overall.total,
        )

    def test_domain_summary_percent(self):
        s = DomainSummary(domain="", total=4, onnxruntime_ok=2)
        self.assertAlmostEqual(s.percent("onnxruntime_ok"), 50.0)
        self.assertAlmostEqual(DomainSummary(domain="").percent("onnxruntime_ok"), 0.0)

    def test_render_rst_summary(self):
        text = render_rst_summary(self.report)
        self.assertIn(".. list-table::", text)
        self.assertIn("onnxruntime (CPU)", text)
        self.assertIn("Static shape inference", text)
        self.assertIn("Dynamic shape inference", text)

    def test_render_rst_domain_summary(self):
        text = render_rst_domain_summary(self.report)
        self.assertIn(".. list-table::", text)
        # The default ai.onnx domain (empty string) should always be present.
        self.assertIn("ai.onnx (default)", text)

    def test_render_rst_table_for_domain(self):
        text = render_rst_table_for_domain(self.report, domain="", css_class="sphinx-datatable")
        self.assertIn(".. list-table::", text)
        self.assertIn(":class: sphinx-datatable", text)
        # Header columns required by the issue.
        self.assertIn("test op", text)
        self.assertIn("test name", text)
        self.assertIn("discrepancies (onnxruntime CPU)", text)
        self.assertIn("static shape", text)
        self.assertIn("dynamic_shapes", text)

    def test_render_rst_domain_sections(self):
        text = render_rst_domain_sections(self.report)
        self.assertNotIn(".. tab-set::", text)
        self.assertIn(":class: sphinx-datatable", text)
        for domain in self.report.summaries:
            label = domain or "ai.onnx (default)"
            self.assertIn(f".. rubric:: {label}", text)

    def test_render_rst_domain_tabs_compat(self):
        self.assertEqual(
            render_rst_domain_tabs(self.report),
            render_rst_domain_sections(self.report),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
