import unittest

import numpy as np
import onnx_light.onnx as onnxl
from onnx_light.backend.coverage import (
    CoverageReport,
    OperatorCoverage,
    compute_test_case_coverage,
)
from onnx_light.backend.test.case.base import ALL_TESTS, expect
from onnx_light.ext_test_case import ExtTestCase


class TestCoverage(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        onnxl.defs.register_onnx_operator_set_schema()

    def setUp(self):
        ALL_TESTS.clear()

    def tearDown(self):
        ALL_TESTS.clear()

    def test_default_collects_cc_cases(self):
        """compute_test_case_coverage() returns a non-trivial report."""
        report = compute_test_case_coverage()
        self.assertIsInstance(report, CoverageReport)
        self.assertGreater(report.total_signatures, 0)
        self.assertGreaterEqual(report.covered_signatures, 0)
        self.assertLessEqual(report.covered_signatures, report.total_signatures)
        self.assertEqual(
            report.covered_signatures,
            sum(oc.covered for oc in report.operator_coverages),
        )
        self.assertEqual(
            report.total_signatures,
            sum(oc.total for oc in report.operator_coverages),
        )
        self.assertGreaterEqual(report.ratio, 0.0)
        self.assertLessEqual(report.ratio, 1.0)
        # uncovered_operators must match operators with no covered type.
        zero = {(oc.domain, oc.name) for oc in report.operator_coverages if oc.covered == 0}
        self.assertEqual(set(report.uncovered_operators), zero)

    def test_explicit_test_cases_abs_float(self):
        """A single Abs/float32 test case covers exactly that signature."""
        node = onnxl.helper.make_node("Abs", inputs=["x"], outputs=["y"])
        x = np.array([-1.0, 2.0], dtype=np.float32)
        y = np.abs(x)
        expect(node, inputs=[x], outputs=[y], name="test_abs_float")

        report = compute_test_case_coverage(dict(ALL_TESTS))
        # Find the Abs operator entry.
        abs_entry = next(
            oc for oc in report.operator_coverages if oc.name == "Abs" and oc.domain == "ai.onnx"
        )
        self.assertIn("tensor(float)", abs_entry.supported_types)
        self.assertEqual(abs_entry.covered_types, ["tensor(float)"])
        self.assertEqual(abs_entry.covered, 1)
        self.assertGreater(abs_entry.total, 1)
        self.assertNotIn("tensor(float)", abs_entry.missing_types)
        self.assertNotIn(("ai.onnx", "Abs"), report.uncovered_operators)

    def test_empty_test_cases(self):
        """With no test cases nothing is covered but the baseline stays."""
        report = compute_test_case_coverage({})
        self.assertGreater(report.total_signatures, 0)
        self.assertEqual(report.covered_signatures, 0)
        self.assertEqual(report.ratio, 0.0)
        # Every operator with a non-empty supported_types is uncovered.
        for oc in report.operator_coverages:
            if oc.total > 0:
                self.assertIn((oc.domain, oc.name), report.uncovered_operators)
                self.assertEqual(oc.covered_types, [])

    def test_operator_coverage_properties(self):
        oc = OperatorCoverage(
            domain="ai.onnx",
            name="Foo",
            supported_types=["tensor(float)", "tensor(int64)"],
            covered_types=["tensor(float)"],
        )
        self.assertEqual(oc.total, 2)
        self.assertEqual(oc.covered, 1)
        self.assertEqual(oc.missing_types, ["tensor(int64)"])
        self.assertAlmostEqual(oc.ratio, 0.5)

    def test_unknown_operator_ignored(self):
        """A test case for an operator absent from the baseline is ignored."""
        from onnx_light.backend.test.case.base import TestCase

        node = onnxl.helper.make_node(
            "DoesNotExist", inputs=["x"], outputs=["y"], domain="custom.domain"
        )
        x = np.array([1.0], dtype=np.float32)
        graph = onnxl.helper.make_graph(
            nodes=[node],
            name="g",
            inputs=[onnxl.helper.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [1])],
            outputs=[onnxl.helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])],
        )
        model = onnxl.helper.make_model(
            graph,
            opset_imports=[onnxl.helper.make_opsetid("custom.domain", 1)],
            producer_name="test",
        )
        tc = TestCase(
            name="test_custom",
            model_name="test_custom",
            url=None,
            model_dir=None,
            model=model,
            data_sets=[([x], [x])],
            kind="node",
            rtol=1e-3,
            atol=1e-7,
        )
        report = compute_test_case_coverage({"test_custom": tc})
        self.assertEqual(report.covered_signatures, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
