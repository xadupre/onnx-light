import unittest

import numpy as np
import onnx_light.onnx as onnxl
from onnx_light.backend.coverage import (
    CoverageReport,
    OperatorCoverage,
    compute_test_case_coverage,
)
from onnx_light.backend.test.case.base import ALL_TESTS, TestCase
from onnx_light.ext_test_case import ExtTestCase


def _make_test_case(
    op_type: str,
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    name: str,
    domain: str = "",
    opset_version: int = 13,
) -> TestCase:
    """Builds a single-node TestCase without relying on the full ONNX defs registry."""
    node = onnxl.helper.make_node(
        op_type,
        inputs=[f"in{i}" for i in range(len(inputs))],
        outputs=[f"out{i}" for i in range(len(outputs))],
        domain=domain,
    )
    in_vis = [
        onnxl.helper.make_tensor_value_info(
            f"in{i}", int(onnxl.helper.np_dtype_to_tensor_dtype(arr.dtype)), list(arr.shape)
        )
        for i, arr in enumerate(inputs)
    ]
    out_vis = [
        onnxl.helper.make_tensor_value_info(
            f"out{i}", int(onnxl.helper.np_dtype_to_tensor_dtype(arr.dtype)), list(arr.shape)
        )
        for i, arr in enumerate(outputs)
    ]
    graph = onnxl.helper.make_graph(nodes=[node], name=name, inputs=in_vis, outputs=out_vis)
    model = onnxl.helper.make_model(
        graph,
        opset_imports=[onnxl.helper.make_opsetid(domain, opset_version)],
        producer_name="test",
    )
    return TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(list(inputs), list(outputs))],
        kind="node",
        rtol=1e-3,
        atol=1e-7,
    )


class TestCoverage(ExtTestCase):
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
            report.covered_signatures, sum(oc.covered for oc in report.operator_coverages)
        )
        self.assertEqual(
            report.total_signatures, sum(oc.total for oc in report.operator_coverages)
        )
        self.assertGreaterEqual(report.ratio, 0.0)
        self.assertLessEqual(report.ratio, 1.0)
        # uncovered_operators must match operators with no covered type
        # (excluding operators whose schema declares no type constraint).
        zero = {
            (oc.domain, oc.name)
            for oc in report.operator_coverages
            if oc.covered == 0 and oc.total > 0
        }
        self.assertEqual(set(report.uncovered_operators), zero)

    def test_explicit_test_cases_abs_float(self):
        """A single Abs/float32 test case covers exactly that signature."""
        x = np.array([-1.0, 2.0], dtype=np.float32)
        y = np.abs(x)
        tc = _make_test_case("Abs", [x], [y], "test_abs_float")

        report = compute_test_case_coverage({"test_abs_float": tc})
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
        x = np.array([1.0], dtype=np.float32)
        tc = _make_test_case(
            "DoesNotExist", [x], [x], "test_custom", domain="custom.domain", opset_version=1
        )
        report = compute_test_case_coverage({"test_custom": tc})
        self.assertEqual(report.covered_signatures, 0)

    def test_multi_node_test_case_credits_every_op(self):
        """Each node in a multi-node model contributes to coverage of its operator."""
        # Two-node graph: Identity -> Abs. Both operators must be credited
        # with tensor(float).
        a = np.array([-1.0, 2.0], dtype=np.float32)
        node1 = onnxl.helper.make_node("Identity", ["in0"], ["mid"], domain="")
        node2 = onnxl.helper.make_node("Abs", ["mid"], ["out0"], domain="")
        in_vi = onnxl.helper.make_tensor_value_info("in0", int(onnxl.TensorProto.FLOAT), [2])
        out_vi = onnxl.helper.make_tensor_value_info("out0", int(onnxl.TensorProto.FLOAT), [2])
        graph = onnxl.helper.make_graph(
            nodes=[node1, node2], name="test_multi", inputs=[in_vi], outputs=[out_vi]
        )
        model = onnxl.helper.make_model(
            graph,
            opset_imports=[onnxl.helper.make_opsetid("", 13)],
            producer_name="test",
        )
        tc = TestCase(
            name="test_multi",
            model_name="test_multi",
            url=None,
            model_dir=None,
            model=model,
            data_sets=[([a], [np.abs(a)])],
            kind="node",
            rtol=1e-3,
            atol=1e-7,
        )
        report = compute_test_case_coverage({"test_multi": tc})
        ident = next(
            oc for oc in report.operator_coverages if oc.name == "Identity" and oc.domain == "ai.onnx"
        )
        abs_oc = next(
            oc for oc in report.operator_coverages if oc.name == "Abs" and oc.domain == "ai.onnx"
        )
        # Identity sees its input (graph input, float) and Abs sees its
        # output (graph output, float). Both must be credited.
        self.assertIn("tensor(float)", ident.covered_types)
        self.assertIn("tensor(float)", abs_oc.covered_types)
        self.assertNotIn(("ai.onnx", "Identity"), report.uncovered_operators)
        self.assertNotIn(("ai.onnx", "Abs"), report.uncovered_operators)


if __name__ == "__main__":
    unittest.main(verbosity=2)
