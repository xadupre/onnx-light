"""Test-case coverage report against the light ONNX operator schemas.

This module measures how well the backend test cases collected by
:func:`onnx_light.backend.test.case.base.collect_test_case` exercise the ONNX
operators described by the lightweight schemas exposed in the C++
``onnx_op`` extension (``LightOpSchema``).

The baseline is the set of light op schemas: for every supported operator and
each tensor type it accepts, one *signature* (``(domain, op_name, type)``) is
counted. A signature is considered *covered* when at least one collected test
case instantiates a model whose single node has that ``op_type`` and uses that
ONNX type for one of its (typed) graph inputs or outputs.

The result is reported by :func:`compute_test_case_coverage` as a
:class:`CoverageReport`:

* :attr:`CoverageReport.total_signatures` — total number of ``(op, type)``
  signatures to cover (the baseline);
* :attr:`CoverageReport.covered_signatures` — signatures actually covered by a
  test case;
* :attr:`CoverageReport.ratio` — coverage ratio in ``[0, 1]``;
* :attr:`CoverageReport.uncovered_operators` — list of ``(domain, op_name)``
  pairs for which no test case is registered at all;
* :attr:`CoverageReport.operator_coverages` — per-operator breakdown
  (:class:`OperatorCoverage`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import onnx_light.onnx as onnxl

# Map TensorProto enum value -> ONNX type-string used in light op schema
# type constraints (e.g. "tensor(float)").
_TP = onnxl.TensorProto
_TENSOR_PROTO_TO_TYPE_STRING: dict[int, str] = {
    int(_TP.FLOAT): "tensor(float)",
    int(_TP.UINT8): "tensor(uint8)",
    int(_TP.INT8): "tensor(int8)",
    int(_TP.UINT16): "tensor(uint16)",
    int(_TP.INT16): "tensor(int16)",
    int(_TP.INT32): "tensor(int32)",
    int(_TP.INT64): "tensor(int64)",
    int(_TP.STRING): "tensor(string)",
    int(_TP.BOOL): "tensor(bool)",
    int(_TP.FLOAT16): "tensor(float16)",
    int(_TP.DOUBLE): "tensor(double)",
    int(_TP.UINT32): "tensor(uint32)",
    int(_TP.UINT64): "tensor(uint64)",
    int(_TP.COMPLEX64): "tensor(complex64)",
    int(_TP.COMPLEX128): "tensor(complex128)",
    int(_TP.BFLOAT16): "tensor(bfloat16)",
    int(_TP.FLOAT8E4M3FN): "tensor(float8e4m3fn)",
    int(_TP.FLOAT8E4M3FNUZ): "tensor(float8e4m3fnuz)",
    int(_TP.FLOAT8E5M2): "tensor(float8e5m2)",
    int(_TP.FLOAT8E5M2FNUZ): "tensor(float8e5m2fnuz)",
    int(_TP.FLOAT8E8M0): "tensor(float8e8m0)",
    int(_TP.FLOAT4E2M1): "tensor(float4e2m1)",
    int(_TP.UINT4): "tensor(uint4)",
    int(_TP.INT4): "tensor(int4)",
    int(_TP.UINT2): "tensor(uint2)",
    int(_TP.INT2): "tensor(int2)",
}


@dataclass
class OperatorCoverage:
    """Per-operator coverage entry of a :class:`CoverageReport`.

    :param domain: Operator domain (e.g. ``"ai.onnx"``).
    :param name: Operator name (e.g. ``"Add"``).
    :param supported_types: ONNX type strings the operator accepts (union of
        all its type constraints in the latest schema version).
    :param covered_types: Subset of ``supported_types`` for which at least one
        test case uses that type on a graph input or output.
    """

    domain: str
    name: str
    supported_types: list[str] = field(default_factory=list)
    covered_types: list[str] = field(default_factory=list)

    @property
    def missing_types(self) -> list[str]:
        """ONNX type strings supported by the operator but never exercised."""
        covered = set(self.covered_types)
        return [t for t in self.supported_types if t not in covered]

    @property
    def total(self) -> int:
        """Number of supported types (baseline for this operator)."""
        return len(self.supported_types)

    @property
    def covered(self) -> int:
        """Number of supported types covered by at least one test case."""
        return len(self.covered_types)

    @property
    def ratio(self) -> float:
        """Coverage ratio for this operator in ``[0, 1]`` (``1`` if total is 0)."""
        return 1.0 if self.total == 0 else self.covered / self.total


@dataclass
class CoverageReport:
    """Aggregated test-case coverage report.

    :param total_signatures: Total number of ``(op, type)`` signatures from the
        baseline (sum of ``supported_types`` over every operator).
    :param covered_signatures: Number of signatures covered by at least one
        test case.
    :param operator_coverages: Per-operator breakdown, sorted by
        ``(domain, name)``.
    :param uncovered_operators: ``(domain, op_name)`` pairs of operators that
        have **no** test case at all (a strict subset of ``operator_coverages``
        entries whose ``covered == 0`` and ``total > 0``).
    """

    total_signatures: int
    covered_signatures: int
    operator_coverages: list[OperatorCoverage]
    uncovered_operators: list[tuple[str, str]]

    @property
    def ratio(self) -> float:
        """Coverage ratio in ``[0, 1]`` (``1`` if ``total_signatures`` is 0)."""
        if self.total_signatures == 0:
            return 1.0
        return self.covered_signatures / self.total_signatures

    def __str__(self) -> str:  # pragma: no cover - human-readable summary
        return (
            f"CoverageReport(total={self.total_signatures}, "
            f"covered={self.covered_signatures}, "
            f"ratio={self.ratio:.3f}, "
            f"uncovered_operators={len(self.uncovered_operators)})"
        )


def _load_light_schemas() -> list[Any]:
    """Loads ``LightOpSchema`` objects from the C++ ``onnx_op`` extension."""
    from ..onnx_py._onnxpy import onnx_op as _op  # type: ignore[attr-defined]

    return list(_op.GetAllOnnxOpSchemasWithHistory())


def _to_type_string(allowed_type: Any) -> str:
    """Converts a ``LightOpSchema`` allowed type entry to its ONNX type string.

    ``allowed_type`` is typically an ``onnx_op.TensorType`` enum value but
    plain strings are accepted as well for forward compatibility.
    """
    if isinstance(allowed_type, str):
        return allowed_type
    from ..onnx_py._onnxpy import onnx_op as _op  # type: ignore[attr-defined]

    return _op.ToTypeString(allowed_type)


def _latest_schema_supported_types() -> dict[tuple[str, str], list[str]]:
    """Returns ``{(domain, name): supported_type_strings}`` from the latest schemas.

    For each operator, picks the schema with the highest ``since_version`` and
    returns the (de-duplicated, order-preserving) union of allowed type strings
    across all its type constraints.
    """
    schemas = _load_light_schemas()
    latest: dict[tuple[str, str], Any] = {}
    for sch in schemas:
        key = (sch.domain, sch.name)
        existing = latest.get(key)
        if existing is None or sch.since_version > existing.since_version:
            latest[key] = sch

    result: dict[tuple[str, str], list[str]] = {}
    for key, sch in latest.items():
        seen: set[str] = set()
        types: list[str] = []
        for tc in sch.type_constraints:
            for at in tc.allowed_type_strs:
                ts = _to_type_string(at)
                if ts not in seen:
                    seen.add(ts)
                    types.append(ts)
        result[key] = types
    return result


def _normalize_domain(domain: str) -> str:
    """Normalizes an operator domain: empty string -> ``"ai.onnx"``."""
    return domain or "ai.onnx"


def _type_proto_to_string(vt: Any) -> str | None:
    """Returns the ONNX type string for a :class:`TypeProto`, or ``None``.

    The returned strings match the format used by ``LightOpSchema`` type
    constraints, e.g. ``"tensor(float)"``, ``"seq(tensor(int32))"`` or
    ``"optional(tensor(double))"``. Handles ``tensor_type``,
    ``sequence_type`` and ``optional_type`` (the latter wrapping either a
    tensor or a sequence of tensors). Returns ``None`` when the element
    type is unknown (``0``) or unsupported.
    """
    if vt is None:
        return None
    tensor_type = getattr(vt, "tensor_type", None)
    if tensor_type is not None:
        try:
            elem_type = int(tensor_type.elem_type)
        except (AttributeError, TypeError, ValueError):
            elem_type = 0
        if elem_type != 0:
            return _TENSOR_PROTO_TO_TYPE_STRING.get(elem_type)
    seq_type = getattr(vt, "sequence_type", None)
    if seq_type is not None:
        try:
            seq_elem = int(seq_type.elem_type.tensor_type.elem_type)
        except (AttributeError, TypeError, ValueError):
            seq_elem = 0
        if seq_elem != 0 and seq_elem in _TENSOR_PROTO_TO_TYPE_STRING:
            return "seq(" + _TENSOR_PROTO_TO_TYPE_STRING[seq_elem] + ")"
    opt_type = getattr(vt, "optional_type", None)
    if opt_type is not None:
        try:
            inner = opt_type.elem_type
        except AttributeError:
            inner = None
        inner_str = _type_proto_to_string(inner) if inner is not None else None
        if inner_str is not None:
            return "optional(" + inner_str + ")"
    return None


def _build_name_to_type(graph: Any) -> dict[str, str]:
    """Builds ``{value_name: type_string}`` for every typed value in ``graph``.

    Combines graph inputs, outputs, ``value_info`` entries and initializers
    (whose ``data_type`` defines a ``tensor(...)`` type).
    """
    name_to_type: dict[str, str] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        ts = _type_proto_to_string(vi.type)
        if ts is not None:
            name_to_type[vi.name] = ts
    for init in graph.initializer:
        try:
            elem_type = int(init.data_type)
        except (AttributeError, TypeError, ValueError):
            elem_type = 0
        if elem_type != 0 and elem_type in _TENSOR_PROTO_TO_TYPE_STRING:
            name_to_type.setdefault(init.name, _TENSOR_PROTO_TO_TYPE_STRING[elem_type])
    return name_to_type


def _types_used_by_test_case(test_case: Any) -> list[tuple[str, str, set[str]]]:
    """Returns a list of ``(domain, op_type, used_type_strings)`` tuples,
    one per node in the test case's :class:`ModelProto`.

    Walks every node in the model (test cases may contain helper nodes such
    as ``SequenceConstruct`` before the operator actually being exercised).
    For each node, the used type strings are the types — pulled from graph
    inputs/outputs, ``value_info`` and initializers — of that node's named
    inputs and outputs. Returns an empty list when the model has no node.
    """
    model = test_case.model
    if model is None:
        return []
    graph = model.graph
    nodes = list(graph.node)
    if not nodes:
        return []
    name_to_type = _build_name_to_type(graph)
    entries: list[tuple[str, str, set[str]]] = []
    for node in nodes:
        op_type = node.op_type
        domain = _normalize_domain(node.domain)
        used: set[str] = set()
        for value_name in list(node.input) + list(node.output):
            if not value_name:
                continue
            ts = name_to_type.get(value_name)
            if ts is not None:
                used.add(ts)
        entries.append((domain, op_type, used))
    return entries


def compute_test_case_coverage(
    test_cases: Mapping[str, Any] | Iterable[Any] | None = None,
) -> CoverageReport:
    """Computes the backend test-case coverage report.

    The *baseline* is the set of light ONNX operator schemas (one per
    operator, latest opset version). For every operator, each of the tensor
    types it accepts contributes one signature to the baseline. The coverage
    is the number of those ``(operator, type)`` signatures for which at least
    one collected test case exists that uses the operator and that type.

    :param test_cases: Optional override for the test cases to score. Accepts
        the mapping returned by
        :func:`onnx_light.backend.test.case.base.collect_test_case` or any
        iterable of :class:`~onnx_light.backend.test.case.base.TestCase`
        objects. When ``None`` (the default) the test cases are collected via
        :func:`collect_test_case`.
    :returns: A :class:`CoverageReport` describing the coverage.
    """
    if test_cases is None:
        from .test.case.base import collect_test_case

        cases_iter: Iterable[Any] = collect_test_case().values()
    elif isinstance(test_cases, Mapping):
        cases_iter = test_cases.values()
    else:
        cases_iter = test_cases

    supported = _latest_schema_supported_types()

    # operator -> set of covered type strings
    covered_per_op: dict[tuple[str, str], set[str]] = {}
    for tc in cases_iter:
        for domain, op_type, used in _types_used_by_test_case(tc):
            key = (domain, op_type)
            if key not in supported:
                # Operator unknown to the light schema baseline (e.g. a private
                # op exported from a downstream Base subclass). Skip it.
                continue
            bucket = covered_per_op.setdefault(key, set())
            bucket.update(t for t in used if t in set(supported[key]))

    operator_coverages: list[OperatorCoverage] = []
    total_signatures = 0
    covered_signatures = 0
    uncovered_operators: list[tuple[str, str]] = []
    for key in sorted(supported.keys()):
        domain, name = key
        types = supported[key]
        covered_types_set = covered_per_op.get(key, set())
        covered_types = [t for t in types if t in covered_types_set]
        total_signatures += len(types)
        covered_signatures += len(covered_types)
        if types and not covered_types_set:
            uncovered_operators.append(key)
        operator_coverages.append(
            OperatorCoverage(
                domain=domain, name=name, supported_types=list(types), covered_types=covered_types
            )
        )

    return CoverageReport(
        total_signatures=total_signatures,
        covered_signatures=covered_signatures,
        operator_coverages=operator_coverages,
        uncovered_operators=uncovered_operators,
    )
