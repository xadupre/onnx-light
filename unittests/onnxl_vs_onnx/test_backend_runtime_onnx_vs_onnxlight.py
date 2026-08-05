"""Validate the ``onnx_light`` runtime against every ONNX backend node test.

For every node-level test shipped by ``onnx.backend.test`` this module loads
the model and its reference input/output tensors, executes the model through
:class:`onnx_light.onnx.reference.ReferenceEvaluator` (which wraps the
``ExecutionPlan`` / ``RuntimeSession`` execution machinery exercised by
``onnx_light/onnx_extensions/backend_test``) and
compares the produced outputs with the reference outputs computed by ONNX.

This is the ``onnx`` vs ``onnx_light`` runtime counterpart of the serialization
round-trip checks in ``test_backend_onnx_vs_onnxlight.py`` and, like that module,
generates one ``test_vs_<name>`` method per ONNX backend node test. A test case
is *skipped* (not run) when the ``onnx_light`` runtime cannot legitimately
execute it today:

* the model cannot be parsed by ``onnx_light``;
* the graph has an ``optional``/``map``/``sparse_tensor`` input/output, which is
  not supported by the Python ``ReferenceEvaluator`` feed/return boundary (plain
  tensors and ``seq(tensor)`` values *are* supported and run);
* the model uses a non-deterministic operator (its outputs cannot be compared
  bit-for-bit against a stored reference);
* the model uses an operator whose result is implementation-defined (image
  decoding depends on the platform's available codecs); or
* the runtime reports the operator as unsupported.

The set of ONNX node tests whose results ``onnx_light`` does *not* reproduce is
maintained as a snapshot in ``_backend_runtime_known_discrepancies.txt``. A
generated ``test_vs_<name>`` fails when a new discrepancy appears (a regression)
or when a recorded discrepancy is resolved (a stale entry that must be removed).
"""

from __future__ import annotations

import glob
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.backend.test.loader import load_model_tests

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
from onnx_light.onnx import helper as onnxl_helper
from onnx_light.onnx import numpy_helper as onnxl_numpy_helper

# The reference runtime is only available in the full build; skip this module on
# a reduced build (ONNX_LIGHT_BUILD_KERNELS=OFF).
ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")

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

# Operators whose result is implementation-defined: the ONNX specification lets
# ``ImageDecoder`` return platform-dependent pixel data (decoding relies on the
# codecs available on the host), so onnx-light cannot be compared against the
# stored reference outputs in a portable way.
_IMPLEMENTATION_DEFINED_OPS: frozenset[str] = frozenset({"ImageDecoder"})

# Per-test tolerances for ONNX backend cases whose stored reference outputs
# retain tiny floating-point residuals that onnx-light rounds closer to zero.
_CUSTOM_FLOAT_TOLERANCES: dict[str, tuple[float, float]] = {
    "test_dft_inverse": (1e-3, 1e-5),
    "test_dft_inverse_opset19": (1e-3, 1e-5),
}


def _should_exclude_runtime_test_name(test_name: str) -> bool:
    """Returns whether a backend runtime test name should be excluded from comparison."""
    return "_expanded" in test_name


def _load_known_discrepancies() -> set[str]:
    with open(_KNOWN_DISCREPANCIES_FILE, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not _should_exclude_runtime_test_name(line.strip())
        }


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


def _value_info_is_sequence(value_info: onnxl.ValueInfoProto) -> bool:
    """Returns ``True`` when ``value_info`` is declared as ``seq(tensor)``."""
    t = value_info.type
    return t.has_sequence_type() and t.sequence_type.elem_type.has_tensor_type()


def _value_info_is_optional_tensor(value_info: onnxl.ValueInfoProto) -> bool:
    """Returns ``True`` when ``value_info`` is declared as ``optional(tensor)``."""
    t = value_info.type
    return (
        t.has_optional_type()
        and t.optional_type.has_elem_type()
        and t.optional_type.elem_type.has_tensor_type()
    )


def _value_info_is_optional_sequence(value_info: onnxl.ValueInfoProto) -> bool:
    """Returns ``True`` when ``value_info`` is declared as ``optional(seq(tensor))``."""
    t = value_info.type
    return (
        t.has_optional_type()
        and t.optional_type.has_elem_type()
        and t.optional_type.elem_type.has_sequence_type()
        and t.optional_type.elem_type.sequence_type.elem_type.has_tensor_type()
    )


def _value_info_kind(value_info: onnxl.ValueInfoProto) -> str | None:
    """Returns the supported boundary kind for ``value_info``.

    Returns one of ``"tensor"``, ``"sequence"``, ``"optional_tensor"``,
    ``"optional_sequence"`` or ``None`` when the value cannot be exercised
    through the Python ``ReferenceEvaluator`` boundary.
    """
    t = value_info.type
    if t.has_tensor_type():
        return "tensor"
    if t.has_sequence_type() and t.sequence_type.elem_type.has_tensor_type():
        return "sequence"
    if t.has_optional_type() and t.optional_type.has_elem_type():
        elem_type = t.optional_type.elem_type
        if elem_type.has_tensor_type():
            return "optional_tensor"
        if elem_type.has_sequence_type() and elem_type.sequence_type.elem_type.has_tensor_type():
            return "optional_sequence"
    return None


def _has_supported_io(model: onnxl.ModelProto) -> bool:
    """Returns ``True`` when every graph input/output uses a supported boundary type.

    The Python ``ReferenceEvaluator`` feed/return boundary represents plain
    tensors as :class:`numpy.ndarray` and ``seq(tensor)`` values as a ``list``
    of arrays. Present ``optional(tensor)`` / ``optional(seq(tensor))`` values
    are represented by the underlying tensor / list through the runtime's
    passthrough optional convention. Any other structured type (``map``,
    ``sparse_tensor`` or unsupported optional element kinds) cannot be
    exercised through that boundary.
    """
    for value_info in list(model.graph.input) + list(model.graph.output):
        if _value_info_kind(value_info) is None:
            return False
    return True


def _load_tensor_value(path: str) -> np.ndarray:
    """Loads a serialised ``TensorProto`` as a :class:`numpy.ndarray`."""
    return numpy_helper.to_array(onnx.load_tensor(path))


def _load_sequence_value(path: str) -> list[np.ndarray]:
    """Loads a serialised ``SequenceProto`` as a list of :class:`numpy.ndarray`."""
    sequence = onnx.SequenceProto()
    with open(path, "rb") as f:
        sequence.ParseFromString(f.read())
    return [numpy_helper.to_array(t) for t in sequence.tensor_values]


def _load_optional_value(path: str) -> np.ndarray | list[np.ndarray] | None:
    """Loads a serialised ``OptionalProto`` as a Python value or ``None``."""
    optional = onnxl.OptionalProto()
    with open(path, "rb") as f:
        optional.ParseFromString(f.read())
    return onnxl_numpy_helper.to_optional(optional)


def _load_value(path: str, kind: str) -> np.ndarray | list[np.ndarray] | None:
    """Loads one backend input/output value according to ``kind``."""
    if kind == "tensor":
        return _load_tensor_value(path)
    if kind == "sequence":
        return _load_sequence_value(path)
    if kind in {"optional_tensor", "optional_sequence"}:
        return _load_optional_value(path)
    raise AssertionError(f"Unexpected backend value kind {kind!r}.")


def _load_data_set(
    data_dir: str, input_kinds: list[str], output_kinds: list[str]
) -> tuple[list, list]:
    """Loads the input/output values of one ``test_data_set_*`` directory.

    Each input/output is loaded as the Python value corresponding to its
    declared boundary kind (tensor, ``seq(tensor)``, ``optional(tensor)``,
    ``optional(seq(tensor))``).
    """
    inputs: list = []
    for i, kind in enumerate(input_kinds):
        path = os.path.join(data_dir, f"input_{i}.pb")
        inputs.append(_load_value(path, kind))
    outputs: list = []
    for i, kind in enumerate(output_kinds):
        path = os.path.join(data_dir, f"output_{i}.pb")
        outputs.append(_load_value(path, kind))
    return inputs, outputs


def _comparison_tolerances(name: str) -> tuple[float, float]:
    """Returns the comparison tolerances for one backend runtime test."""
    return _CUSTOM_FLOAT_TOLERANCES.get(name, (1e-3, 1e-7))


def _describe_mismatch(
    actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float
) -> str | None:
    """Returns a description of why ``actual`` differs from ``expected``.

    Returns ``None`` when ``actual`` reproduces ``expected``. Floating-point
    tensors are compared with ``rtol``/``atol`` tolerances while integer, boolean
    and string tensors are compared for exact equality. Sub-byte and
    reduced-precision ``ml_dtypes`` values are widened to ``float64``/``int64`` so
    the comparison does not depend on the storage representation. When the values
    differ the returned message names the first mismatching position together with
    its expected and actual values, which is far more actionable than a bare
    pass/fail flag when diagnosing a runtime discrepancy.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return f"shape mismatch: got {actual.shape}, expected {expected.shape}"

    kind = expected.dtype.kind
    if kind in ("U", "S", "O"):
        equal = actual.astype(str) == expected.astype(str)
    elif kind == "b":
        equal = actual == expected
    elif "float" in expected.dtype.name or kind == "f":
        equal = np.isclose(
            actual.astype(np.float64),
            expected.astype(np.float64),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    else:
        equal = actual.astype(np.int64) == expected.astype(np.int64)

    if np.all(equal):
        return None

    mismatched = np.argwhere(~np.asarray(equal))
    first = tuple(int(i) for i in mismatched[0])
    index = first[0] if actual.ndim == 1 else first
    return (
        f"{int(mismatched.shape[0])} of {expected.size} values differ; "
        f"first at index {index}: got {actual[first].item()!r}, "
        f"expected {expected[first].item()!r}"
    )


def _describe_output_mismatch(actual, expected, rtol: float, atol: float) -> str | None:
    """Returns a description of why output ``actual`` differs from ``expected``.

    Handles both tensor outputs (a :class:`numpy.ndarray`) and ``seq(tensor)``
    outputs (a ``list`` of arrays). Empty optionals are represented as
    ``None``. For sequences the element counts must match and each element is
    compared in turn. Returns ``None`` when they match.
    """
    if expected is None or actual is None:
        return (
            None
            if expected is actual
            else f"type mismatch: got {actual!r}, expected {expected!r}"
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"type mismatch: got {type(actual).__name__}, expected a sequence"
        if len(actual) != len(expected):
            return f"sequence length mismatch: got {len(actual)}, expected {len(expected)}"
        for j, (got_elem, ref_elem) in enumerate(zip(actual, expected)):
            detail = _describe_mismatch(got_elem, ref_elem, rtol=rtol, atol=atol)
            if detail is not None:
                return f"element {j}: {detail}"
        return None
    if isinstance(actual, list):
        return f"type mismatch: got a sequence, expected {type(expected).__name__}"
    return _describe_mismatch(actual, expected, rtol=rtol, atol=atol)


class TestBackendRuntimeOnnxVsOnnxLight(ExtTestCase):
    """Runs every ONNX backend node test through the ``onnx_light`` runtime."""

    def test_should_exclude_runtime_test_name(self) -> None:
        self.assertTrue(_should_exclude_runtime_test_name("elu_example_expanded_ver18"))
        self.assertTrue(_should_exclude_runtime_test_name("attention_4d_fp16_expanded"))
        self.assertFalse(_should_exclude_runtime_test_name("attention_4d_fp16"))

    def test_load_known_discrepancies_excludes_expanded_names(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("test_attention_4d_fp16_expanded\n")
            f.write("test_training_dropout\n")
            path = f.name
        try:
            with patch(f"{__name__}._KNOWN_DISCREPANCIES_FILE", path):
                self.assertEqual(_load_known_discrepancies(), {"test_training_dropout"})
        finally:
            os.unlink(path)

    def test_comparison_tolerances_defaults_and_overrides(self):
        self.assertEqual(_comparison_tolerances("test_abs"), (1e-3, 1e-7))
        self.assertEqual(_comparison_tolerances("test_dft_inverse"), (1e-3, 1e-5))
        self.assertEqual(_comparison_tolerances("test_dft_inverse_opset19"), (1e-3, 1e-5))

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

    def test_run_one_loop16_seq_none(self):
        model_file = None
        for test in load_model_tests(kind="node"):
            if os.path.basename(test.model_dir) == "test_loop16_seq_none":
                model_file = os.path.join(test.model_dir, "model.onnx")
                break
        self.assertIsNotNone(
            model_file, "ONNX backend case 'test_loop16_seq_none' was not found."
        )
        self.assertTrue(os.path.exists(model_file))
        outcome, detail = self._run_one(model_file)
        self.assertEqual(outcome, "pass")
        self.assertIsNone(detail)

    def _run_one(self, model_file: str) -> tuple[str, str | None]:
        """Executes one backend test and returns its outcome and a detail.

        Returns ``("pass", None)`` when the runtime reproduces the reference
        outputs, ``("fail", detail)`` when it does not (or raises an unexpected
        error) and ``("skip", None)`` when the case cannot be run by the runtime
        today. ``detail`` describes the discrepancy (which output mismatched and
        the first differing value, or the unexpected error raised) so a failure
        is actionable without re-running the case by hand.
        """
        try:
            model = onnxl.load(model_file)
        except RuntimeError:
            return "skip", None

        if not _has_supported_io(model):
            return "skip", None
        if _model_op_types(model) & _NON_DETERMINISTIC_OPS:
            return "skip", None
        if _model_op_types(model) & _IMPLEMENTATION_DEFINED_OPS:
            return "skip", None

        model_dir = os.path.dirname(model_file)
        test_name = os.path.basename(model_dir)
        rtol, atol = _comparison_tolerances(test_name)
        data_dirs = sorted(glob.glob(os.path.join(model_dir, "test_data_set*")))
        if not data_dirs:
            return "skip", None

        input_kinds = [_value_info_kind(vi) for vi in model.graph.input]
        output_kinds = [_value_info_kind(vi) for vi in model.graph.output]
        assert all(kind is not None for kind in input_kinds + output_kinds)

        session = ReferenceEvaluator(model)
        for data_dir in data_dirs:
            inputs, expected = _load_data_set(data_dir, input_kinds, output_kinds)
            if any(value is None for value in inputs + expected):
                return "skip", None
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
                    return "skip", None
                return "fail", f"runtime raised {type(e).__name__}: {e}"
            if len(outputs) != len(expected):
                return "fail", (
                    f"output count mismatch: got {len(outputs)}, expected {len(expected)}"
                )
            for i, (got, ref) in enumerate(zip(outputs, expected)):
                detail = _describe_output_mismatch(got, ref, rtol=rtol, atol=atol)
                if detail is not None:
                    return "fail", f"output {i} ({os.path.basename(data_dir)}): {detail}"
        return "pass", None

    def _run_model_test(self, model_file: str, name: str) -> None:
        """Runs one backend node test and checks it against the snapshot.

        ``name`` is the test's directory basename (e.g. ``test_abs``). When it
        appears in ``_backend_runtime_known_discrepancies.txt`` the runtime is
        expected *not* to reproduce the reference outputs; any other outcome
        (the case now passes or can no longer be executed) means the snapshot
        entry is stale and must be removed. Otherwise the case is skipped when
        the runtime cannot execute it and must pass when it can.
        """
        known = _load_known_discrepancies()
        outcome, detail = self._run_one(model_file)
        snapshot = os.path.basename(_KNOWN_DISCREPANCIES_FILE)
        if name in known:
            if outcome != "fail":
                self.fail(
                    f"{name!r} is listed in {snapshot} but the onnx-light runtime "
                    f"now handles it (outcome={outcome!r}). Remove this entry from "
                    f"{snapshot}."
                )
            return
        if outcome == "skip":
            self.skipTest(f"onnx-light runtime cannot execute {name!r} today")
        if outcome == "fail":
            self.fail(
                f"The onnx-light runtime does not reproduce {name!r} ({detail}). "
                f"Fix the runtime or append {name!r} to {snapshot}."
            )

    def test_known_discrepancies_all_have_tests(self):
        # A name recorded in the snapshot that no longer corresponds to any ONNX
        # backend node test is a stale entry that must be removed.
        known = _load_known_discrepancies()
        names = {
            os.path.basename(test.model_dir)
            for test in load_model_tests(kind="node")
            if os.path.exists(os.path.join(test.model_dir, "model.onnx"))
        }
        # Guard against a misconfigured environment where nothing is discovered.
        self.assertGreater(len(names), 0, "No ONNX backend node test could be found.")
        stale_entries = sorted(known - names)
        if stale_entries:
            self.fail(
                "Entries in the known-discrepancies snapshot no longer match any "
                f"ONNX backend node test ({len(stale_entries)}): {stale_entries[:20]}"
                + (" ..." if len(stale_entries) > 20 else "")
                + "\nRemove these names from "
                + os.path.basename(_KNOWN_DISCREPANCIES_FILE)
                + "."
            )

    @classmethod
    def add_test_methods(cls):
        for test in load_model_tests(kind="node"):
            model_file = os.path.join(test.model_dir, "model.onnx")
            if not os.path.exists(model_file):
                continue
            name = os.path.basename(test.model_dir)
            if _should_exclude_runtime_test_name(name):
                continue

            def _test_(self, model_file=model_file, name=name):
                self._run_model_test(model_file, name)

            short_name = name.replace("test_", "", 1)
            setattr(cls, f"test_vs_{short_name}", _test_)


TestBackendRuntimeOnnxVsOnnxLight.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
