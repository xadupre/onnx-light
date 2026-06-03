"""Runtime coverage report for the backend test cases.

This module compares every backend test case collected by
:func:`onnx_light.backend.test.case.base.collect_test_case` against three
independent runtime / static-analysis scenarios:

* **onnxruntime CPU** — the model is executed with ``onnxruntime`` on the CPU
  execution provider and the maximum absolute discrepancy between the
  reference outputs and the ORT outputs is recorded;
* **static shape** — the model is passed to
  :func:`onnx_light.onnx_lib.shape_inference.infer_shapes` with the original
  (numeric) input shapes;
* **dynamic shapes** — every numeric input dimension is replaced with a
  symbolic ``dim_param`` (so identical numeric values share the same symbol)
  and shape inference is run on the resulting symbolic model.

The result is a :class:`RuntimeCoverageReport` containing one
:class:`TestCaseStatus` per backend test case, plus aggregate statistics per
domain and globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .. import onnx as onnxl
from ..onnx_lib.shape_inference import infer_shapes
from .test.case import collect_test_case
from .test.case.base import TestCase


@dataclass
class TestCaseStatus:
    """Outcome of the three runtime scenarios for one backend test case."""

    # Tell pytest this isn't a test class.
    __test__ = False

    name: str
    op_type: str
    domain: str
    onnxruntime_cpu: float | None
    """Maximum absolute discrepancy between reference and ORT outputs, or
    ``None`` when ORT could not load / run the model (registered op missing,
    unsupported domain, ...)."""

    onnxruntime_error: str | None
    """Error message returned by ORT when ``onnxruntime_cpu`` is ``None``."""

    static_shape: bool
    """``True`` if :func:`infer_shapes` succeeds on the original model."""

    static_shape_error: str | None

    dynamic_shapes: bool
    """``True`` if :func:`infer_shapes` succeeds on the symbolic-shape model."""

    dynamic_shapes_error: str | None

    @property
    def onnxruntime_ok(self) -> bool:
        """``True`` when ORT ran and outputs matched within tolerances."""
        if self.onnxruntime_cpu is None:
            return False
        # Compare against atol + rtol * scale, where scale conservatively
        # bounds typical reference output magnitudes. Backend test cases set
        # ``rtol`` for relative comparisons; the diff was computed as
        # ``max(|out - exp|)`` so we approximate the allowed slack as
        # ``atol + rtol``. This is enough to keep tests with ``atol == 0`` and
        # a non-zero ``rtol`` (e.g. trigonometric ops) reported as passing.
        atol = getattr(self, "_atol", 1e-3)
        rtol = getattr(self, "_rtol", 0.0)
        return self.onnxruntime_cpu <= atol + rtol


def _principal_op(tc: TestCase) -> tuple[str, str]:
    """Returns the ``(op_type, domain)`` representative of a test case.

    Most backend test cases are single-node models (possibly with a few
    ``Constant`` initializer nodes); the principal op is the first non
    ``Constant`` node, falling back to the first node when every node is a
    ``Constant``.
    """
    if tc.model is None:
        return ("", "")
    for node in tc.model.graph.node:
        if node.op_type != "Constant":
            return (node.op_type, node.domain)
    nodes = list(tc.model.graph.node)
    if nodes:
        return (nodes[0].op_type, nodes[0].domain)
    return ("", "")


def _run_onnxruntime(tc: TestCase) -> tuple[float | None, str | None]:
    """Runs ``tc.model`` with onnxruntime CPU and returns ``(max_diff, error)``.

    Returns ``(None, message)`` if ``onnxruntime`` cannot load or run the
    model. Returns ``(float('inf'), 'dtype/shape mismatch ...')`` if a
    structural mismatch (different element type, different shape, ...)
    prevents a meaningful numerical comparison.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return (None, f"onnxruntime not available ({exc})")

    if tc.model is None:
        return (None, "no model")

    try:
        sess = ort.InferenceSession(
            tc.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        return (None, type(exc).__name__ + ": " + str(exc).splitlines()[0])

    if not tc.data_sets:
        # No reference outputs — the test only validates that ORT can load it.
        return (0.0, None)

    max_diff = 0.0
    input_names = [i.name for i in sess.get_inputs()]
    for inputs, expected in tc.data_sets:
        try:
            outputs = sess.run(None, dict(zip(input_names, inputs)))
        except Exception as exc:  # noqa: BLE001
            return (None, type(exc).__name__ + ": " + str(exc).splitlines()[0])
        for out, exp in zip(outputs, expected):
            ea = np.asarray(exp)
            oa = np.asarray(out)
            if ea.dtype.kind in ("U", "S", "O"):
                if ea.shape != oa.shape or not (ea == oa).all():
                    return (float("inf"), "string output mismatch")
                continue
            if ea.shape != oa.shape:
                return (float("inf"), f"shape mismatch {ea.shape} vs {oa.shape}")
            if ea.size == 0:
                # Nothing to compare element-wise; matching shapes already
                # implies an equal output.
                continue
            # ``np.errstate(invalid="ignore")`` keeps NaN inputs from emitting
            # a ``RuntimeWarning`` to stderr; such a warning would otherwise be
            # captured by ``sphinx_runpython`` when this report is rendered in
            # the documentation and would corrupt the surrounding reST output.
            with np.errstate(invalid="ignore"):
                diff = float(
                    np.max(np.abs(ea.astype(np.float64) - oa.astype(np.float64)))
                )
            if diff > max_diff:
                max_diff = diff
    return (max_diff, None)


def _clone_model(model: onnxl.ModelProto) -> onnxl.ModelProto:
    """Returns a serialized round-trip copy of ``model``."""
    copy = onnxl.ModelProto()
    copy.ParseFromString(model.SerializeToString())
    return copy


def _run_static_shape(tc: TestCase) -> tuple[bool, str | None]:
    if tc.model is None:
        return (False, "no model")
    try:
        infer_shapes(_clone_model(tc.model))
    except Exception as exc:  # noqa: BLE001
        return (False, type(exc).__name__ + ": " + str(exc).splitlines()[0])
    return (True, None)


def _run_dynamic_shapes(tc: TestCase) -> tuple[bool, str | None]:
    if tc.model is None:
        return (False, "no model")
    model = _clone_model(tc.model)
    for vi in model.graph.input:
        if vi.type.tensor_type is None:
            continue
        if not vi.type.tensor_type.shape.dim:
            continue
        for dim in vi.type.tensor_type.shape.dim:
            if dim.has_dim_value:
                dim.dim_param = f"sym_v{dim.dim_value}"
    try:
        infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        return (False, type(exc).__name__ + ": " + str(exc).splitlines()[0])
    return (True, None)


@dataclass
class DomainSummary:
    """Aggregated pass counts for a single domain."""

    domain: str
    total: int = 0
    onnxruntime_ok: int = 0
    static_shape_ok: int = 0
    dynamic_shapes_ok: int = 0

    def percent(self, attr: str) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * getattr(self, attr) / self.total


@dataclass
class RuntimeCoverageReport:
    """Per-test-case runtime coverage report.

    The :attr:`statuses` list contains one :class:`TestCaseStatus` per
    collected test case. :attr:`summaries` groups the statuses by domain and
    counts how many test cases pass each scenario, and :attr:`overall` is the
    same aggregation across every test case.
    """

    statuses: list[TestCaseStatus] = field(default_factory=list)
    summaries: dict[str, DomainSummary] = field(default_factory=dict)
    overall: DomainSummary = field(default_factory=lambda: DomainSummary(domain="<all>"))


def compute_runtime_coverage(
    test_cases: Iterable[TestCase] | None = None,
) -> RuntimeCoverageReport:
    """Builds the runtime coverage report for every backend test case.

    :param test_cases: Optional iterable of :class:`TestCase` to evaluate. When
        ``None``, :func:`collect_test_case` is invoked to gather every
        available test case.
    :returns: A populated :class:`RuntimeCoverageReport`.
    """
    if test_cases is None:
        cases = list(collect_test_case().values())
    else:
        cases = list(test_cases)

    report = RuntimeCoverageReport()
    for tc in cases:
        if tc.model is None:
            continue
        op_type, domain = _principal_op(tc)
        ort_diff, ort_err = _run_onnxruntime(tc)
        static_ok, static_err = _run_static_shape(tc)
        dynamic_ok, dynamic_err = _run_dynamic_shapes(tc)
        status = TestCaseStatus(
            name=tc.name,
            op_type=op_type,
            domain=domain,
            onnxruntime_cpu=ort_diff,
            onnxruntime_error=ort_err,
            static_shape=static_ok,
            static_shape_error=static_err,
            dynamic_shapes=dynamic_ok,
            dynamic_shapes_error=dynamic_err,
        )
        # Attach the test-case tolerances so onnxruntime_ok can apply them.
        status._atol = tc.atol  # type: ignore[attr-defined]
        status._rtol = tc.rtol  # type: ignore[attr-defined]
        report.statuses.append(status)

        summary = report.summaries.setdefault(domain, DomainSummary(domain=domain))
        summary.total += 1
        report.overall.total += 1
        if status.onnxruntime_ok:
            summary.onnxruntime_ok += 1
            report.overall.onnxruntime_ok += 1
        if status.static_shape:
            summary.static_shape_ok += 1
            report.overall.static_shape_ok += 1
        if status.dynamic_shapes:
            summary.dynamic_shapes_ok += 1
            report.overall.dynamic_shapes_ok += 1

    # Sort statuses by (domain, op_type, name) so the rendered table is stable.
    report.statuses.sort(key=lambda s: (s.domain, s.op_type, s.name))
    return report


# ---------------------------------------------------------------------------
# Rendering helpers (used by the documentation page)
# ---------------------------------------------------------------------------


def _format_discrepancy(status: TestCaseStatus) -> str:
    """Returns the cell content for the ``discrepancies (onnxruntime CPU)`` column.

    The value is wrapped in a custom Sphinx role — ``:green:`` when the test
    passes, ``:red:`` otherwise — declared at the top of the documentation
    page and styled with the CSS classes defined in ``docs/_static/custom.css``.
    """
    if status.onnxruntime_cpu is None:
        text = "n/a"
    elif status.onnxruntime_cpu == float("inf"):
        text = "mismatch"
    else:
        text = f"{status.onnxruntime_cpu:.2e}"
    role = "green" if status.onnxruntime_ok else "red"
    return f":{role}:`{text}`"


def _format_bool(value: bool) -> str:
    role = "green" if value else "red"
    text = "yes" if value else "no"
    return f":{role}:`{text}`"


def render_rst_summary(report: RuntimeCoverageReport) -> str:
    """Renders the global pass-percentage summary as a reST ``list-table``."""
    o = report.overall
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 40 20 20 20",
        "",
        "    * - Scenario",
        "      - Passed",
        "      - Total",
        "      - Pass rate",
        "    * - onnxruntime (CPU)",
        f"      - {o.onnxruntime_ok}",
        f"      - {o.total}",
        f"      - {o.percent('onnxruntime_ok'):.1f}%",
        "    * - Static shape inference",
        f"      - {o.static_shape_ok}",
        f"      - {o.total}",
        f"      - {o.percent('static_shape_ok'):.1f}%",
        "    * - Dynamic shape inference",
        f"      - {o.dynamic_shapes_ok}",
        f"      - {o.total}",
        f"      - {o.percent('dynamic_shapes_ok'):.1f}%",
    ]
    return "\n".join(lines) + "\n"


def render_rst_domain_summary(report: RuntimeCoverageReport) -> str:
    """Renders one row per domain with its individual pass percentages."""
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 30 10 20 20 20",
        "",
        "    * - Domain",
        "      - Tests",
        "      - onnxruntime (CPU)",
        "      - Static shape",
        "      - Dynamic shapes",
    ]
    for domain in sorted(report.summaries):
        s = report.summaries[domain]
        label = domain or "ai.onnx (default)"
        lines.extend(
            [
                f"    * - ``{label}``",
                f"      - {s.total}",
                f"      - {s.onnxruntime_ok} ({s.percent('onnxruntime_ok'):.1f}%)",
                f"      - {s.static_shape_ok} ({s.percent('static_shape_ok'):.1f}%)",
                f"      - {s.dynamic_shapes_ok} ({s.percent('dynamic_shapes_ok'):.1f}%)",
            ]
        )
    return "\n".join(lines) + "\n"


def render_rst_table_for_domain(
    report: RuntimeCoverageReport, domain: str, css_class: str | None = None, indent: str = ""
) -> str:
    """Renders the per-test-case table for one ``domain`` as a reST ``list-table``.

    :param report: The runtime coverage report.
    :param domain: ONNX operator domain to filter on (e.g. ``""`` for the
        default ``ai.onnx`` domain).
    :param css_class: When provided, a ``:class:`` option is emitted on the
        ``list-table`` directive. Pass ``"sphinx-datatable"`` to opt into the
        interactive DataTables widget.
    :param indent: Optional whitespace prefix prepended to every output line —
        useful to inline the table inside a ``tab-item`` directive.
    """
    header_lines = [".. list-table::", "    :header-rows: 1", "    :widths: 20 30 20 15 15"]
    if css_class:
        header_lines.append(f"    :class: {css_class}")
    header_lines.extend(
        [
            "",
            "    * - test op",
            "      - test name",
            "      - discrepancies (onnxruntime CPU)",
            "      - static shape",
            "      - dynamic_shapes",
        ]
    )
    body_lines: list[str] = []
    for s in report.statuses:
        if s.domain != domain:
            continue
        body_lines.extend(
            [
                f"    * - {s.op_type}",
                f"      - {s.name}",
                f"      - {_format_discrepancy(s)}",
                f"      - {_format_bool(s.static_shape)}",
                f"      - {_format_bool(s.dynamic_shapes)}",
            ]
        )
    out = "\n".join(header_lines + body_lines) + "\n"
    if indent:
        out = "\n".join(indent + line if line else line for line in out.splitlines()) + "\n"
    return out


def render_rst_domain_sections(
    report: RuntimeCoverageReport, css_class: str | None = "sphinx-datatable"
) -> str:
    """Returns one coverage table per domain without tab-based navigation."""
    lines: list[str] = []
    for domain in sorted(report.summaries):
        label = domain or "ai.onnx (default)"
        lines.append(f".. rubric:: {label}")
        lines.append("")
        table = render_rst_table_for_domain(report, domain=domain, css_class=css_class)
        lines.extend(table.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def render_rst_domain_tabs(
    report: RuntimeCoverageReport, css_class: str | None = "sphinx-datatable"
) -> str:
    """Returns :func:`render_rst_domain_sections` output for backward compatibility."""
    return render_rst_domain_sections(report, css_class=css_class)
