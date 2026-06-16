"""Validate the ``onnx_light`` runtime against every ONNX backend node test.

For every node-level test shipped by ``onnx.backend.test`` this module loads
the model and its reference input/output tensors, executes the model through
:class:`onnx_light.onnx.reference.ReferenceEvaluator` (which wraps the C++
``RunModel`` dispatcher exercised by ``onnx_light/onnx_backend_test``) and
compares the produced outputs with the reference outputs computed by ONNX.

This is the ``onnx`` vs ``onnx_light`` runtime counterpart of the serialization
round-trip checks in ``test_backend_onnx_vs_onnxlight.py``. A test case is
*skipped* (not run) when the ``onnx_light`` runtime cannot legitimately execute
it today:

* the model cannot be parsed by ``onnx_light``;
* the graph has a non-tensor input/output (sequence/optional/map), which the
  Python ``ReferenceEvaluator`` feed/return boundary represents differently;
* the model uses a non-deterministic operator (its outputs cannot be compared
  bit-for-bit against a stored reference); or
* the runtime reports the operator as unsupported.

The set of ONNX node tests whose results ``onnx_light`` does *not* reproduce is
maintained as a snapshot in ``_backend_runtime_known_discrepancies.txt`` so the
test fails when a new discrepancy appears (a regression) or when a recorded
discrepancy is resolved (a stale entry that must be removed).
"""

from __future__ import annotations

import glob
import os
import unittest

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.backend.test.loader import load_model_tests

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx import helper as onnxl_helper
from onnx_light.onnx.reference import ReferenceEvaluator

_KNOWN_DISCREPANCIES_FILE = os.path.join(
    os.path.dirname(__file__), "_backend_runtime_known_discrepancies.txt"
)

# Operators whose outputs are drawn from a pseudo-random distribution; their
# results legitimately differ from the stored reference outputs and therefore
# cannot be validated by a value comparison.
_NON_DETERMINISTIC_OPS: frozenset[str] = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)


def _load_known_discrepancies() -> set[str]:
    with open(_KNOWN_DISCREPANCIES_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _model_op_types(model: onnxl.ModelProto) -> set[str]:
    """Returns every ``op_type`` referenced (recursively) by ``model``."""
    ops: set[str] = set()

    def _walk(graph) -> None:
        for node in graph.node:
            ops.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnxl.AttributeProto.GRAPH:
                    _walk(attr.g)
                elif attr.type == onnxl.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        _walk(sub)

    _walk(model.graph)
    return ops


def _has_only_tensor_io(model: onnxl.ModelProto) -> bool:
    """Returns ``True`` when every graph input and output is a plain tensor."""
    for value_info in list(model.graph.input) + list(model.graph.output):
        if not value_info.type.has_tensor_type():
            return False
    return True


def _load_data_set(data_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Loads the input/output tensors of one ``test_data_set_*`` directory."""
    inputs = [
        numpy_helper.to_array(onnx.load_tensor(os.path.join(data_dir, f"input_{i}.pb")))
        for i in range(len(glob.glob(os.path.join(data_dir, "input_*.pb"))))
    ]
    outputs = [
        numpy_helper.to_array(onnx.load_tensor(os.path.join(data_dir, f"output_{i}.pb")))
        for i in range(len(glob.glob(os.path.join(data_dir, "output_*.pb"))))
    ]
    return inputs, outputs


def _values_match(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> bool:
    """Returns ``True`` when ``actual`` reproduces ``expected``.

    Floating-point tensors are compared with ``rtol``/``atol`` tolerances while
    integer, boolean and string tensors are compared for exact equality. Sub-byte
    and reduced-precision ``ml_dtypes`` values are widened to ``float64``/``int64``
    so the comparison does not depend on the storage representation.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return False

    kind = expected.dtype.kind
    if kind in ("U", "S", "O"):
        return np.array_equal(actual.astype(str), expected.astype(str))
    if kind == "b":
        return np.array_equal(actual, expected)
    if "float" in expected.dtype.name or kind == "f":
        return np.allclose(
            actual.astype(np.float64),
            expected.astype(np.float64),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    return np.array_equal(actual.astype(np.int64), expected.astype(np.int64))


class TestBackendRuntimeOnnxVsOnnxLight(ExtTestCase):
    """Runs every ONNX backend node test through the ``onnx_light`` runtime."""

    def test_model_op_types_walks_subgraphs(self):
        # Guards the recursive traversal in ``_model_op_types`` so operators
        # nested inside ``If``/``Loop``/``Scan`` subgraphs (GRAPH and GRAPHS
        # attributes) are taken into account when filtering cases.
        then_graph = onnxl_helper.make_graph(
            [onnxl_helper.make_node("Relu", ["x"], ["y"])],
            "then",
            [],
            [onnxl_helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])],
        )
        else_graph = onnxl_helper.make_graph(
            [onnxl_helper.make_node("Neg", ["x"], ["y"])],
            "else",
            [],
            [onnxl_helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])],
        )
        if_node = onnxl_helper.make_node(
            "If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph
        )
        graph = onnxl_helper.make_graph(
            [if_node],
            "g",
            [onnxl_helper.make_tensor_value_info("cond", onnxl.TensorProto.BOOL, [])],
            [onnxl_helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])],
        )
        model = onnxl_helper.make_model(graph)
        self.assertEqual(_model_op_types(model), {"If", "Relu", "Neg"})

    def _run_one(self, model_file: str) -> str:
        """Executes one backend test and returns its outcome.

        Returns ``"pass"`` when the runtime reproduces the reference outputs,
        ``"fail"`` when it does not (or raises an unexpected error) and
        ``"skip"`` when the case cannot be run by the runtime today.
        """
        try:
            model = onnxl.load(model_file)
        except RuntimeError:
            return "skip"

        if not _has_only_tensor_io(model):
            return "skip"
        if _model_op_types(model) & _NON_DETERMINISTIC_OPS:
            return "skip"

        model_dir = os.path.dirname(model_file)
        data_dirs = sorted(glob.glob(os.path.join(model_dir, "test_data_set*")))
        if not data_dirs:
            return "skip"

        session = ReferenceEvaluator(model)
        for data_dir in data_dirs:
            inputs, expected = _load_data_set(data_dir)
            feeds = dict(zip(session.input_names, inputs))
            # Executing the model is an external runtime boundary that can fail.
            # The runtime has no Python-side API to query the set of registered
            # kernels, so an operator it does not implement can only be detected
            # from the ``ValueError`` it raises ("unsupported op_type"). Such a
            # case is skipped; any other failure is a genuine discrepancy.
            try:
                outputs = session.run(None, feeds)
            except ValueError as e:
                if "unsupported op_type" in str(e):
                    return "skip"
                return "fail"
            if len(outputs) != len(expected):
                return "fail"
            for got, ref in zip(outputs, expected):
                if not _values_match(got, ref, rtol=1e-3, atol=1e-7):
                    return "fail"
        return "pass"

    def test_backend_runtime_onnx_vs_onnxlight(self):
        failed: set[str] = set()
        ran = 0
        for test in load_model_tests(kind="node"):
            model_file = os.path.join(test.model_dir, "model.onnx")
            if not os.path.exists(model_file):
                continue
            name = os.path.basename(test.model_dir)
            outcome = self._run_one(model_file)
            if outcome == "skip":
                continue
            ran += 1
            if outcome == "fail":
                failed.add(name)

        # Guard against a misconfigured environment where nothing executes.
        self.assertGreater(ran, 0, "No ONNX backend node test could be executed.")

        known = _load_known_discrepancies()
        new_failures = sorted(failed - known)
        stale_entries = sorted(known - failed)

        messages = []
        if new_failures:
            messages.append(
                "ONNX backend tests the onnx-light runtime no longer reproduces "
                f"({len(new_failures)}): {new_failures[:20]}"
                + (" ..." if len(new_failures) > 20 else "")
                + "\nFix the runtime or append these names to "
                + os.path.basename(_KNOWN_DISCREPANCIES_FILE)
                + "."
            )
        if stale_entries:
            messages.append(
                "Entries in the known-discrepancies snapshot now reproduced by "
                f"onnx-light ({len(stale_entries)}): {stale_entries[:20]}"
                + (" ..." if len(stale_entries) > 20 else "")
                + "\nRemove these names from "
                + os.path.basename(_KNOWN_DISCREPANCIES_FILE)
                + "."
            )
        if messages:
            self.fail("\n\n".join(messages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
