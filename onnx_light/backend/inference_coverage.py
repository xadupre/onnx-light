"""Shape-inference report for the ``inference``-tagged backend test cases.

The C++ ``lib_onnx_backend_test`` library ships a handful of small graphs
explicitly authored to exercise shape inference. They are tagged
``"inference"`` (see :attr:`onnx_light.backend.test.case.base.TestCase.tag`)
and they record the *expected* intermediate shapes in
``graph.value_info`` and the *expected* output shapes in
``graph.output[i].type``.

This module collects those cases, strips ``graph.value_info`` (the expected
intermediate shapes) from a copy of every model, runs the ``onnx_optim``
shape-inference pipeline on the stripped copy via
:func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`, and
produces a report comparing the *expected* shape of every value (the one
recorded by the test author) against the *computed* shape (the one produced
by shape inference).

The companion documentation page :ref:`l-design-inference-coverage` renders,
for every ``"inference"``-tagged case, a Mermaid diagram of the original
model (with the expected shapes) and a table contrasting expected and
computed shapes side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .. import onnx as onnxl
from ..onnx_optim.shape_inference import infer_shapes_model
from ..tools import to_mermaid
from .test.case import collect_test_case
from .test.case.base import TestCase


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ValueShape:
    """Shape / element type of a single tensor value (input, intermediate or
    output) in a backend test model.

    A shape entry is ``None`` when the corresponding ``ValueInfoProto`` has no
    type set (so the value is opaque to shape inference). Dimensions are
    encoded as either ``int`` (``dim_value``), ``str`` (``dim_param``) or
    ``None`` (unknown dimension).
    """

    elem_type: int | None
    """ONNX ``TensorProto.DataType`` enum value, or ``None`` when the value
    has no tensor type."""

    shape: list[int | str | None] | None
    """One entry per dimension or ``None`` when the value has no tensor
    shape set."""


@dataclass
class ValueComparison:
    """Per-value comparison of expected vs computed shape."""

    name: str
    """Name of the value (graph input, ``value_info`` or graph output)."""

    role: str
    """``"input"``, ``"value_info"`` or ``"output"``."""

    expected: ValueShape | None
    """Shape recorded in the original test model, or ``None`` when the value
    has no expected shape (for instance an intermediate without a
    ``value_info`` entry)."""

    computed: ValueShape | None
    """Shape produced by :func:`infer_shapes_model`, or ``None`` when shape
    inference produced no entry for that value."""

    @property
    def match(self) -> bool:
        """Returns whether ``expected`` and ``computed`` agree.

        A value with no expected shape (``expected is None``) is considered to
        match by convention so that intermediate values without a
        ``value_info`` entry do not flag the case as failing. Otherwise both
        the element type and the shape must be equal.
        """
        if self.expected is None:
            return True
        if self.computed is None:
            return False
        if self.expected.elem_type != self.computed.elem_type:
            return False
        if self.expected.shape is None and self.computed.shape is None:
            return True
        if self.expected.shape is None or self.computed.shape is None:
            return False
        return list(self.expected.shape) == list(self.computed.shape)


@dataclass
class InferenceCaseReport:
    """Shape-inference outcome for one ``"inference"``-tagged test case."""

    name: str
    """Name of the backend test case (e.g.
    ``"test_cc_shape_inference_add_concat_reshape"``)."""

    mermaid: str
    """Mermaid ``flowchart`` source rendering the *original* model (with
    its expected ``value_info`` annotations)."""

    error: str | None
    """Error message raised by :func:`infer_shapes_model`, or ``None`` when
    shape inference succeeded."""

    comparisons: list[ValueComparison] = field(default_factory=list)
    """One :class:`ValueComparison` per input, intermediate (``value_info``)
    and output value of the model. Empty when ``error`` is set."""

    @property
    def ok(self) -> bool:
        """``True`` when shape inference completed and every comparison
        matches."""
        if self.error is not None:
            return False
        return all(c.match for c in self.comparisons)


@dataclass
class InferenceCoverageReport:
    """Aggregates :class:`InferenceCaseReport` for every collected case."""

    cases: list[InferenceCaseReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.ok)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_value_shape(value_info) -> ValueShape | None:
    """Extracts a :class:`ValueShape` from a ``ValueInfoProto``.

    Returns ``None`` when the value has no type at all.
    """
    if not value_info.has_type():
        return None
    type_proto = value_info.type
    if not type_proto.has_tensor_type():
        # Non-tensor types (sequence, map, optional, sparse...) are not
        # part of the reports rendered on the doc page; we only record the
        # element type if we can.
        return ValueShape(elem_type=None, shape=None)
    tt = type_proto.tensor_type
    elem_type = int(tt.elem_type) if tt.elem_type else None
    if not tt.has_shape():
        return ValueShape(elem_type=elem_type, shape=None)
    dims: list[int | str | None] = []
    for dim in tt.shape.dim:
        if dim.has_dim_value():
            dims.append(int(dim.dim_value))
        elif dim.has_dim_param() and dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return ValueShape(elem_type=elem_type, shape=dims)


def _clone_model(model: onnxl.ModelProto) -> onnxl.ModelProto:
    """Returns a fresh :class:`onnxl.ModelProto` parsed from ``model``'s
    serialized form. The onnx-light proto classes do not support
    ``copy.deepcopy``."""
    clone = onnxl.ModelProto()
    clone.ParseFromString(model.SerializeToString())
    return clone


def _strip_value_info(model: onnxl.ModelProto) -> None:
    """Clears every ``graph.value_info`` entry in place.

    This mirrors what the C++ shape-inference tests do: shape inference must
    recover the intermediate shapes from scratch instead of taking them from
    the model's recorded ``value_info``.
    """
    del model.graph.value_info[:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _iter_inference_cases() -> Iterable[TestCase]:
    cases = collect_test_case()
    for name in sorted(cases):
        tc = cases[name]
        if tc.tag == "inference":
            yield tc


def compute_inference_coverage() -> InferenceCoverageReport:
    """Computes the shape-inference report for every ``"inference"`` case.

    For every case, the model is deep-cloned, its ``graph.value_info`` is
    cleared, and :func:`infer_shapes_model` (from ``onnx_optim``) is run on
    the clone. The report contrasts the *expected* shapes from the original
    model with the *computed* shapes from the inferred clone.
    """
    report = InferenceCoverageReport()

    for tc in _iter_inference_cases():
        original = tc.model
        if original is None:  # pragma: no cover - defensive
            continue
        try:
            mermaid = to_mermaid(original)
        except Exception as exc:  # pragma: no cover - defensive only
            mermaid = f"flowchart TB\n    err[\"to_mermaid failed: {exc}\"]"

        # Snapshot expected shapes from the original (untouched) model so
        # that we can compare them against the shapes produced by shape
        # inference on the stripped clone.
        expected_inputs = {vi.name: _extract_value_shape(vi) for vi in original.graph.input}
        expected_value_info = {
            vi.name: _extract_value_shape(vi) for vi in original.graph.value_info
        }
        expected_outputs = {vi.name: _extract_value_shape(vi) for vi in original.graph.output}

        clone = _clone_model(original)
        _strip_value_info(clone)
        error: str | None = None
        try:
            infer_shapes_model(clone)
        except Exception as exc:  # capture shape-inference failures
            error = str(exc)

        comparisons: list[ValueComparison] = []
        if error is None:
            computed_inputs = {vi.name: _extract_value_shape(vi) for vi in clone.graph.input}
            computed_value_info = {
                vi.name: _extract_value_shape(vi) for vi in clone.graph.value_info
            }
            computed_outputs = {vi.name: _extract_value_shape(vi) for vi in clone.graph.output}

            for name, expected in expected_inputs.items():
                comparisons.append(
                    ValueComparison(
                        name=name,
                        role="input",
                        expected=expected,
                        computed=computed_inputs.get(name),
                    )
                )
            for name, expected in expected_value_info.items():
                comparisons.append(
                    ValueComparison(
                        name=name,
                        role="value_info",
                        expected=expected,
                        computed=computed_value_info.get(name),
                    )
                )
            for name, expected in expected_outputs.items():
                comparisons.append(
                    ValueComparison(
                        name=name,
                        role="output",
                        expected=expected,
                        computed=computed_outputs.get(name),
                    )
                )

        report.cases.append(
            InferenceCaseReport(
                name=tc.name,
                mermaid=mermaid,
                error=error,
                comparisons=comparisons,
            )
        )

    return report


# ---------------------------------------------------------------------------
# Rendering helpers (used by the documentation page)
# ---------------------------------------------------------------------------


def _format_shape(shape: ValueShape | None) -> str:
    if shape is None:
        return "—"
    if shape.elem_type is None:
        elem = "?"
    else:
        try:
            elem = onnxl.TensorProto.DataType(shape.elem_type).name
        except Exception:  # pragma: no cover - defensive
            elem = str(shape.elem_type)
    if shape.shape is None:
        dims = "?"
    elif not shape.shape:
        dims = "[]"
    else:
        dims = (
            "["
            + ", ".join("?" if d is None else str(d) for d in shape.shape)
            + "]"
        )
    return f"{elem}{dims}"


def _format_match(comparison: ValueComparison) -> str:
    role = "green" if comparison.match else "red"
    text = "yes" if comparison.match else "no"
    return f":{role}:`{text}`"


def render_rst_summary(report: InferenceCoverageReport) -> str:
    """Renders a one-row summary as a reST ``list-table``."""
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 40 20 20 20",
        "",
        "    * - Scenario",
        "      - Passed",
        "      - Total",
        "      - Pass rate",
        "    * - onnx_optim shape inference",
        f"      - {report.passed}",
        f"      - {report.total}",
        f"      - {(100.0 * report.passed / report.total if report.total else 0.0):.1f}%",
    ]
    return "\n".join(lines) + "\n"


def render_rst_case(case: InferenceCaseReport) -> str:
    """Renders one test case as a reST section.

    The section contains:

    * a ``.. runmermaid::`` block rendering the original model;
    * either an error admonition (when shape inference raised) or a
      ``list-table`` contrasting expected and computed shapes for every
      input, intermediate and output value.
    """
    title = f"``{case.name}``"
    lines: list[str] = [title, "+" * len(title), ""]

    # Mermaid graph (raw form: the body of the directive is the Mermaid
    # source itself, which is exactly what ``to_mermaid`` returns).
    lines.append(".. runmermaid::")
    lines.append("")
    for ml in case.mermaid.splitlines():
        lines.append(f"    {ml}" if ml else "")
    lines.append("")

    if case.error is not None:
        lines.append(".. warning::")
        lines.append("")
        lines.append("    Shape inference raised:")
        lines.append("")
        lines.append("    .. code-block:: text")
        lines.append("")
        for el in case.error.splitlines() or [""]:
            lines.append(f"        {el}")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append(".. list-table::")
    lines.append("    :header-rows: 1")
    lines.append("    :widths: 25 15 25 25 10")
    lines.append("")
    lines.append("    * - Name")
    lines.append("      - Role")
    lines.append("      - Expected")
    lines.append("      - Computed")
    lines.append("      - Match")
    for cmp in case.comparisons:
        lines.append(f"    * - ``{cmp.name}``")
        lines.append(f"      - {cmp.role}")
        lines.append(f"      - {_format_shape(cmp.expected)}")
        lines.append(f"      - {_format_shape(cmp.computed)}")
        lines.append(f"      - {_format_match(cmp)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_rst_report(report: InferenceCoverageReport) -> str:
    """Renders every collected case back-to-back."""
    parts = [render_rst_case(c) for c in report.cases]
    return "\n".join(parts)


__all__ = [
    "InferenceCaseReport",
    "InferenceCoverageReport",
    "ValueComparison",
    "ValueShape",
    "compute_inference_coverage",
    "render_rst_case",
    "render_rst_report",
    "render_rst_summary",
]
