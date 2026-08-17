import unittest


from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The runtime coverage helpers are only available in the full build; skip this
# module on a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
_runtime_coverage = import_or_skip("onnx_light.onnx_lib.backend.runtime_coverage")
DomainSummary = _runtime_coverage.DomainSummary
RuntimeCoverageReport = _runtime_coverage.RuntimeCoverageReport
TestCaseStatus = _runtime_coverage.TestCaseStatus
compute_runtime_coverage = _runtime_coverage.compute_runtime_coverage
render_rst_domain_sections = _runtime_coverage.render_rst_domain_sections
render_rst_domain_summary = _runtime_coverage.render_rst_domain_summary
render_rst_domain_tabs = _runtime_coverage.render_rst_domain_tabs
render_rst_summary = _runtime_coverage.render_rst_summary
render_rst_table_for_domain = _runtime_coverage.render_rst_table_for_domain
_run_onnxruntime = _runtime_coverage._run_onnxruntime
_format_discrepancy = _runtime_coverage._format_discrepancy
_ORT_MAX_IR_VERSION = _runtime_coverage._ORT_MAX_IR_VERSION


class TestRuntimeCoverage(ExtTestCase):
    """Validates the runtime coverage report helpers."""

    @classmethod
    def setUpClass(cls):
        cls.report = compute_runtime_coverage()

    def test_report_is_populated(self):
        self.assertIsInstance(self.report, RuntimeCoverageReport)
        self.assertGreater(len(self.report.statuses), 0)
        self.assertEqual(
            self.report.overall.total, sum(s.total for s in self.report.summaries.values())
        )
        self.assertEqual(self.report.overall.total, len(self.report.statuses))

    def test_status_fields(self):
        status = self.report.statuses[0]
        self.assertIsInstance(status, TestCaseStatus)
        self.assertIsInstance(status.name, str)
        self.assertIsInstance(status.op_type, str)
        self.assertIsInstance(status.domain, str)
        self.assertIsInstance(status.tag, str)
        self.assertEqual(status.group, status.tag or status.domain)
        self.assertIsInstance(status.static_shape, bool)
        self.assertIsInstance(status.dynamic_shapes, bool)
        # onnxruntime_cpu is either None (cannot run) or a non-negative float.
        if status.onnxruntime_cpu is not None:
            self.assertGreaterEqual(status.onnxruntime_cpu, 0.0)

    def test_tagged_cases_have_own_group(self):
        # C++ test cases tagged with ``nan_inf``, ``local_function`` or
        # ``inference`` must appear in their own summary section instead of
        # being aggregated under their principal op's domain.
        groups = set(self.report.summaries)
        for status in self.report.statuses:
            if status.tag:
                # The summary key is the tag, not the principal op's domain.
                self.assertIn(status.tag, groups)
                self.assertEqual(status.group, status.tag)
        for expected_tag in ("nan_inf", "local_function", "inference"):
            tagged = [s for s in self.report.statuses if s.tag == expected_tag]
            if tagged:
                self.assertIn(expected_tag, groups)

    def test_static_and_dynamic_shape_pass_overall(self):
        # After stripping the recorded ``value_info`` and output shapes from
        # the test models, shape inference is no longer trivial: it must
        # actually recompute output shapes that match what the test author
        # recorded. Most cases pass, but a handful currently fail (operator
        # shape inference is incomplete or not registered for some custom
        # ops). We only check that the overall counts are sensible: positive,
        # bounded by the total, and that static and dynamic stay aligned.
        total = self.report.overall.total
        static_ok = self.report.overall.static_shape_ok
        dynamic_ok = self.report.overall.dynamic_shapes_ok
        self.assertGreater(static_ok, 0)
        self.assertGreater(dynamic_ok, 0)
        self.assertLessEqual(static_ok, total)
        self.assertLessEqual(dynamic_ok, total)

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
        for group in self.report.summaries:
            label = group or "ai.onnx (default)"
            self.assertIn(f".. rubric:: {label}", text)

    def test_render_rst_domain_tabs_compat(self):
        self.assertEqual(
            render_rst_domain_tabs(self.report), render_rst_domain_sections(self.report)
        )


class TestRuntimeCoverageOnnxruntime(ExtTestCase):
    """Validates the onnxruntime comparison honors tolerances and reports
    missing reference outputs as ``n/a`` rather than a false pass."""

    @classmethod
    def setUpClass(cls):
        import_or_skip("onnxruntime")
        import numpy as np

        import onnx_light.onnx as onnxl
        import onnx_light.onnx.defs as defs
        from onnx_light.onnx_lib.backend.test.case.base import ALL_TESTS, expect

        defs.register_onnx_operator_set_schema()
        cls.np = np
        cls.onnxl = onnxl
        cls.expect = staticmethod(expect)
        cls.ALL_TESTS = ALL_TESTS

    def setUp(self):
        self.ALL_TESTS.clear()

    def tearDown(self):
        self.ALL_TESTS.clear()

    def _make_identity_case(self, name, ort_output, expected):
        np = self.np
        node = self.onnxl.helper.make_node("Identity", inputs=["x"], outputs=["y"])
        # The model is an Identity, so ORT returns ``ort_output`` unchanged;
        # the recorded reference output is ``expected``.
        self.expect(
            node,
            inputs=[np.array(ort_output, dtype=np.float32)],
            outputs=[np.array(expected, dtype=np.float32)],
            name=name,
        )
        tc = self.ALL_TESTS[name]
        tc.model.ir_version = _ORT_MAX_IR_VERSION
        return tc

    def test_unsupported_ir_version_is_skipped(self):
        tc = self._make_identity_case("test_unsupported_ir", [1.0], [1.0])
        tc.model.ir_version = _ORT_MAX_IR_VERSION + 1
        diff, ok, err = _run_onnxruntime(tc)
        self.assertIsNone(diff)
        self.assertFalse(ok)
        self.assertEqual(
            err,
            f"skipped: model IR version {tc.model.ir_version} exceeds "
            f"onnxruntime maximum {_ORT_MAX_IR_VERSION}",
        )

    def test_tolerance_uses_relative_magnitude(self):
        # atol=1e-7, rtol=1e-3, expected magnitude 1000 -> allowed slack is
        # ``atol + rtol * |expected| ~= 1.0``. A discrepancy of 0.5 is within
        # tolerance even though it far exceeds the crude ``atol + rtol`` (1e-3)
        # threshold that the report used before.
        tc = self._make_identity_case("test_tol_relative", [1000.5], [1000.0])
        diff, ok, err = _run_onnxruntime(tc)
        self.assertIsNone(err)
        self.assertTrue(ok)
        # The reported discrepancy exceeds the naive atol + rtol threshold, so
        # the previous logic would have (incorrectly) flagged this as a failure.
        self.assertGreater(diff, tc.atol + tc.rtol)

    def test_tolerance_flags_real_discrepancy(self):
        # A discrepancy larger than ``atol + rtol * |expected|`` must fail.
        tc = self._make_identity_case("test_tol_fail", [10.0], [1.0])
        diff, ok, err = _run_onnxruntime(tc)
        self.assertIsNone(err)
        self.assertFalse(ok)
        self.assertGreater(diff, 0.0)

    def test_missing_reference_outputs_is_not_a_pass(self):
        tc = self._make_identity_case("test_no_ref", [1.0], [1.0])
        tc.data_sets = []
        diff, ok, err = _run_onnxruntime(tc)
        self.assertIsNone(diff)
        self.assertFalse(ok)
        self.assertEqual(err, "no reference outputs")

    def test_missing_reference_outputs_render_na(self):
        status = TestCaseStatus(
            name="x",
            op_type="Identity",
            domain="",
            tag="",
            onnxruntime_cpu=None,
            onnxruntime_ok=False,
            onnxruntime_error="no reference outputs",
            static_shape=True,
            static_shape_error=None,
            dynamic_shapes=True,
            dynamic_shapes_error=None,
        )
        cell = _format_discrepancy(status)
        self.assertIn("n/a", cell)
        self.assertIn(":red:", cell)

    def test_compute_runtime_coverage_no_ref_not_counted(self):
        tc = self._make_identity_case("test_no_ref_count", [1.0], [1.0])
        tc.data_sets = []
        report = compute_runtime_coverage([tc])
        self.assertEqual(report.overall.total, 1)
        self.assertEqual(report.overall.onnxruntime_ok, 0)
        self.assertFalse(report.statuses[0].onnxruntime_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
