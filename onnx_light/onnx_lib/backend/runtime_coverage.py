"""Runtime coverage report for the backend test cases.

This module compares every backend test case collected by
:func:`onnx_light.backend.test.case.base.collect_test_case` against three
independent runtime / static-analysis scenarios:

* **onnxruntime CPU** — the model is executed with ``onnxruntime`` on the CPU
  execution provider and the maximum absolute discrepancy between the
  reference outputs and the ORT outputs is recorded;
* **static shape** — the model's recorded ``graph.value_info`` and output
  shapes are stripped, :func:`onnx_light.onnx_lib.shape_inference.infer_shapes`
  is run on the stripped clone, and the inferred output shapes are checked
  against the originally-recorded ones;
* **dynamic shapes** — every numeric input dimension is replaced with a
  symbolic ``dim_param`` (so identical numeric values share the same symbol),
  ``value_info`` and output shapes are stripped, and shape inference is run
  on the resulting symbolic model with the same output-shape verification.

The result is a :class:`RuntimeCoverageReport` containing one
:class:`TestCaseStatus` per backend test case, plus aggregate statistics per
domain and globally.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import cache
from typing import Iterable

from ...onnx_py._onnxpybackend import backend_test as _backend_test_cc  # type: ignore

import numpy as np

from ... import onnx as onnxl
from ...onnx_proto import _helper as onnxl_helper
from ..defs import onnx_ir_version, onnx_opset_version
from ...onnx_lib.shape_inference import infer_shapes
from .test.case import collect_test_case
from .test.case.base import TestCase, _unload_test_case


@dataclass
class TestCaseStatus:
    """Outcome of the three runtime scenarios for one backend test case."""

    # Tell pytest this isn't a test class.
    __test__ = False

    name: str
    op_type: str
    domain: str
    tag: str
    """Free-form tag attached to the C++ test case (e.g. ``"nan_inf"``,
    ``"local_function"``, ``"inference"``). When non-empty, the test case is
    grouped in the coverage report under this tag rather than under its
    principal op's :attr:`domain`."""

    onnxruntime_cpu: float | None
    """Maximum absolute discrepancy between reference and ORT outputs, or
    ``None`` when ORT could not load / run the model, or when the test case
    carries no reference outputs to compare against (registered op missing,
    unsupported domain, no expected values, ...)."""

    onnxruntime_ok: bool
    """``True`` when ORT ran and every output matched the recorded reference
    within the test-case ``atol``/``rtol`` tolerances. ``False`` when ORT could
    not run, when an output differed, or when the case has no reference outputs
    to compare against."""

    onnxruntime_error: str | None
    """Error message returned by ORT when ``onnxruntime_cpu`` is ``None``."""

    static_shape: bool
    """``True`` if :func:`infer_shapes` succeeds on the original model."""

    static_shape_error: str | None

    dynamic_shapes: bool
    """``True`` if :func:`infer_shapes` succeeds on the symbolic-shape model."""

    dynamic_shapes_error: str | None

    @property
    def group(self) -> str:
        """Group key used to bucket the test case in the coverage report.

        Returns :attr:`tag` when it is set (so cases tagged with e.g.
        ``"nan_inf"`` get their own section, separate from the regular ops of
        the principal domain), otherwise the principal op :attr:`domain`.
        """
        return self.tag or self.domain


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


# Test cases that are known to abort (SIGABRT / exit) ``onnxruntime`` rather
# than raise a catchable Python exception. They are skipped here so the
# coverage report can still be computed; the cases remain fully exercised by
# the reference backend and by the C++ shape-inference tests.
#   * ``test_cc_shape_inference_shape_identity_unsqueeze`` — exercises the
#     in-memory INT64 initializer path in
#     ``Graph::SaveShapeValuesFromDataPropagation``. ORT versions predating
#     microsoft/onnxruntime#28778 abort while loading the model.
_ORT_SKIP_CASES = frozenset({"test_cc_shape_inference_shape_identity_unsqueeze"})


@cache
def _ort_fail_error() -> type[Exception] | None:
    """Returns the ONNX Runtime model-load failure exception type."""
    try:
        state = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")
    except ImportError:
        return None
    fail_error = getattr(state, "Fail", None)
    if isinstance(fail_error, type) and issubclass(fail_error, Exception):
        return fail_error
    return None


@cache
def ort_max_ir_version() -> int:
    """Returns the highest ONNX IR version accepted by ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        return 0
    fail_error = _ort_fail_error()
    if fail_error is None:
        return 0

    value_info = onnxl_helper.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [1])
    output_info = onnxl_helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])
    graph = onnxl_helper.make_graph(
        [onnxl_helper.make_node("Identity", ["x"], ["y"])],
        "onnxruntime_ir_version_probe",
        [value_info],
        [output_info],
    )
    opset_imports = [onnxl_helper.make_opsetid("", 13)]

    for ir_version in range(onnx_ir_version(), 0, -1):
        model = onnxl_helper.make_model(graph, ir_version=ir_version, opset_imports=opset_imports)
        try:
            ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        except fail_error:
            continue
        return ir_version

    raise RuntimeError("onnxruntime does not accept any supported ONNX IR version")


@cache
def ort_max_opset_version() -> int:
    """Returns the highest default-domain opset accepted by ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        return 0
    fail_error = _ort_fail_error()
    if fail_error is None:
        return 0

    value_info = onnxl_helper.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [1])
    output_info = onnxl_helper.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [1])
    graph = onnxl_helper.make_graph(
        [onnxl_helper.make_node("Identity", ["x"], ["y"])],
        "onnxruntime_opset_version_probe",
        [value_info],
        [output_info],
    )
    ir_version = ort_max_ir_version()

    for opset_version in range(onnx_opset_version(), 0, -1):
        model = onnxl_helper.make_model(
            graph,
            ir_version=ir_version,
            opset_imports=[onnxl_helper.make_opsetid("", opset_version)],
        )
        try:
            ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        except fail_error:
            continue
        return opset_version

    raise RuntimeError("onnxruntime does not accept any supported default-domain opset")


def _run_onnxruntime(tc: TestCase) -> tuple[float | None, bool, str | None]:
    """Runs ``tc.model`` with onnxruntime CPU and returns ``(max_diff, ok, error)``.

    ``max_diff`` is the maximum absolute discrepancy between the reference and
    ORT outputs (for display), ``ok`` is ``True`` only when every output matched
    within the test-case ``atol``/``rtol`` tolerances, and ``error`` carries a
    short message when the comparison could not be performed.

    Returns ``(None, False, message)`` if ``onnxruntime`` cannot load or run the
    model, or if the test case carries no reference outputs to compare against.
    Returns ``(float('inf'), False, 'dtype/shape mismatch ...')`` if a
    structural mismatch (different element type, different shape, ...) prevents
    a meaningful numerical comparison.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return (None, False, f"onnxruntime not available ({exc})")

    if tc.model is None:
        return (None, False, "no model")

    if tc.name in _ORT_SKIP_CASES:
        return (
            None,
            False,
            "skipped: known to abort onnxruntime (see microsoft/onnxruntime#28778)",
        )

    max_opset_version = ort_max_opset_version()
    for opset in tc.model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version > max_opset_version:
            return (
                None,
                False,
                (
                    f"skipped: model opset version {opset.version} exceeds "
                    f"onnxruntime maximum {max_opset_version}"
                ),
            )

    # Freeze the model IR version to the highest value onnxruntime accepts.
    # Backend test models track the latest ONNX IR version, which routinely runs
    # ahead of the version the bundled onnxruntime supports. Without capping it,
    # the coverage dashboard would report a large number of spurious "IR version
    # exceeds onnxruntime maximum" skips whenever ONNX bumps its IR version.
    max_ir_version = ort_max_ir_version()
    run_model = tc.model
    if max_ir_version and tc.model.ir_version > max_ir_version:
        run_model = _clone_model(tc.model)
        run_model.ir_version = max_ir_version

    try:
        sess = ort.InferenceSession(
            run_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        return (None, False, type(exc).__name__ + ": " + str(exc).splitlines()[0])

    if not tc.data_sets:
        # No reference outputs to compare against: the test only validates that
        # ORT can load the model. Report ``n/a`` rather than a fake ``0.0``
        # discrepancy, so a missing comparison is never counted as a pass.
        return (None, False, "no reference outputs")

    max_diff = 0.0
    ok = True
    input_names = [i.name for i in sess.get_inputs()]
    for inputs, expected in tc.data_sets:
        try:
            outputs = sess.run(None, dict(zip(input_names, inputs)))
        except Exception as exc:  # noqa: BLE001
            return (None, False, type(exc).__name__ + ": " + str(exc).splitlines()[0])
        for out, exp in zip(outputs, expected):
            ea = np.asarray(exp)
            oa = np.asarray(out)
            if ea.dtype.kind in ("U", "S", "O"):
                if ea.shape != oa.shape or not (ea == oa).all():
                    return (float("inf"), False, "string output mismatch")
                continue
            if ea.shape != oa.shape:
                return (float("inf"), False, f"shape mismatch {ea.shape} vs {oa.shape}")
            if ea.size == 0:
                # Nothing to compare element-wise; matching shapes already
                # implies an equal output.
                continue
            eaf = ea.astype(np.float64)
            oaf = oa.astype(np.float64)
            # ``np.errstate(invalid="ignore")`` keeps NaN inputs from emitting
            # a ``RuntimeWarning`` to stderr; such a warning would otherwise be
            # captured by ``sphinx_runpython`` when this report is rendered in
            # the documentation and would corrupt the surrounding reST output.
            with np.errstate(invalid="ignore"):
                diff = float(np.max(np.abs(eaf - oaf)))
                # Pass/fail must honor the test-case tolerances the way
                # ``assert_allclose`` does: the allowed slack is ``atol +
                # rtol * |expected|`` per element, not a single ``atol + rtol``
                # threshold on the maximum absolute error. ``equal_nan`` keeps
                # NaN-producing cases (e.g. ``nan_inf`` tagged tests) passing
                # when the reference is NaN in the same positions.
                close = np.allclose(oaf, eaf, rtol=tc.rtol, atol=tc.atol, equal_nan=True)
            if not close:
                ok = False
                if not np.isfinite(diff):
                    # A non-finite maximum means the reference and ORT outputs
                    # disagree on NaN/inf placement. There is no finite
                    # magnitude to display, so surface it as ``mismatch``
                    # instead of a misleading ``0`` discrepancy.
                    max_diff = float("inf")
                    continue
            if np.isfinite(diff) and diff > max_diff:
                max_diff = diff
    return (max_diff, ok, None)


def _clone_model(model: onnxl.ModelProto) -> onnxl.ModelProto:
    """Returns a serialized round-trip copy of ``model``."""
    copy = onnxl.ModelProto()
    copy.ParseFromString(model.SerializeToString())
    return copy


def _output_shapes(model: onnxl.ModelProto) -> dict[str, list[int | str | None] | None]:
    """Returns ``{name: shape}`` for every graph output.

    A shape is represented as a list mixing ``int`` (concrete dim), ``str``
    (symbolic ``dim_param``) and ``None`` (dim with neither value nor
    parameter). Outputs with no shape recorded map to ``None``.
    """
    result: dict[str, list[int | str | None] | None] = {}
    for vi in model.graph.output:
        shape: list[int | str | None] | None = None
        if vi.has_type():
            ttype = vi.type.tensor_type
            if ttype is not None and ttype.has_shape():
                shape = []
                for dim in ttype.shape.dim:
                    if dim.has_dim_value:
                        shape.append(dim.dim_value)
                    elif dim.has_dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append(None)
        result[vi.name] = shape
    return result


def _shapes_compatible(
    expected: list[int | str | None] | None, computed: list[int | str | None] | None
) -> bool:
    """Returns ``True`` if ``computed`` is consistent with ``expected``.

    Both shapes must have the same rank. For every dimension, an unknown
    component (``None`` or symbolic ``str``) on either side is considered
    compatible with anything; concrete integer dimensions must match
    exactly. ``expected is None`` means the original model did not record an
    output shape, in which case any computed shape (including ``None``) is
    accepted.
    """
    if expected is None:
        return True
    if computed is None:
        return False
    if len(expected) != len(computed):
        return False
    for e, c in zip(expected, computed):
        if isinstance(e, int) and isinstance(c, int) and e != c:
            return False
    return True


def _strip_intermediate_and_output_shapes(model: onnxl.ModelProto) -> None:
    """Strips intermediate (``value_info``) and output shapes in place.

    Backend test cases typically embed the expected shapes for every
    intermediate and output value. Leaving those entries in the model would
    make :func:`infer_shapes` succeed trivially without actually computing
    anything: the ONNX shape-inference contract preserves shapes already
    present in the graph. Removing ``value_info`` entirely and clearing the
    shape of each ``graph.output`` (while keeping the output names and
    element types) forces shape inference to recompute every intermediate
    and output shape from the graph inputs and operator semantics.
    """
    del model.graph.value_info[:]
    for vi in model.graph.output:
        if not vi.has_type():
            continue
        ttype = vi.type.tensor_type
        if ttype is None or not ttype.has_shape():
            continue
        # Force shape inference to recompute the shape by removing the one
        # currently stored on the output's tensor type.
        ttype.ClearField("shape")


def _run_static_shape(tc: TestCase) -> tuple[bool, str | None]:
    if tc.model is None:
        return (False, "no model")
    expected = _output_shapes(tc.model)
    model = _clone_model(tc.model)
    _strip_intermediate_and_output_shapes(model)
    try:
        inferred = infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        return (False, type(exc).__name__ + ": " + str(exc).splitlines()[0])
    computed = _output_shapes(inferred)
    for name, exp in expected.items():
        comp = computed.get(name)
        if not _shapes_compatible(exp, comp):
            return (False, f"output {name!r} shape {comp} does not match expected {exp}")
    return (True, None)


def _run_dynamic_shapes(tc: TestCase) -> tuple[bool, str | None]:
    if tc.model is None:
        return (False, "no model")
    expected = _output_shapes(tc.model)
    model = _clone_model(tc.model)
    _strip_intermediate_and_output_shapes(model)
    for vi in model.graph.input:
        if vi.type.tensor_type is None:
            continue
        if not vi.type.tensor_type.shape.dim:
            continue
        for dim in vi.type.tensor_type.shape.dim:
            if dim.has_dim_value:
                dim.dim_param = f"sym_v{dim.dim_value}"
    try:
        inferred = infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        return (False, type(exc).__name__ + ": " + str(exc).splitlines()[0])
    computed = _output_shapes(inferred)
    for name, exp in expected.items():
        comp = computed.get(name)
        # In the dynamic-shapes scenario the expected concrete dims have been
        # converted to symbols; only ranks and any remaining concrete dim are
        # checked, which ``_shapes_compatible`` already handles.
        if not _shapes_compatible(exp, comp):
            return (False, f"output {name!r} shape {comp} does not match expected {exp}")
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
    test_cases: Iterable[TestCase] | None = None, unload: bool = True
) -> RuntimeCoverageReport:
    """Builds the runtime coverage report for every backend test case.

    :param test_cases: Optional iterable of :class:`TestCase` to evaluate. When
        ``None``, :func:`collect_test_case` is invoked to gather every
        available test case.
    :param unload: Releases each native-backed test case after processing it.
        Defaults to ``True``.
    :returns: A populated :class:`RuntimeCoverageReport`.
    """
    if test_cases is None:
        cases = list(collect_test_case().values())
    else:
        cases = list(test_cases)

    report = RuntimeCoverageReport()
    for tc in cases:
        try:
            if tc.model is None:
                continue
            op_type, domain = _principal_op(tc)
            ort_diff, ort_ok, ort_err = _run_onnxruntime(tc)
            static_ok, static_err = _run_static_shape(tc)
            dynamic_ok, dynamic_err = _run_dynamic_shapes(tc)
            status = TestCaseStatus(
                name=tc.name,
                op_type=op_type,
                domain=domain,
                tag=(_backend_test_cc.test_case_tag_name(tc.tag) if hasattr(tc, "tag") else ""),
                onnxruntime_cpu=ort_diff,
                onnxruntime_ok=ort_ok,
                onnxruntime_error=ort_err,
                static_shape=static_ok,
                static_shape_error=static_err,
                dynamic_shapes=dynamic_ok,
                dynamic_shapes_error=dynamic_err,
            )
            report.statuses.append(status)

            group = status.group
            summary = report.summaries.setdefault(group, DomainSummary(domain=group))
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
        finally:
            _unload_test_case(tc, unload)

    # Sort statuses by (group, op_type, name) so the rendered table is stable.
    report.statuses.sort(key=lambda s: (s.group, s.op_type, s.name))
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


def _group_label(group: str) -> str:
    """Renders a human-friendly label for a coverage group.

    Group keys are either an ONNX op domain (``""`` for the default
    ``ai.onnx`` domain, ``"ai.onnx.ml"``, ``"ai.onnx.preview.training"``, ...)
    or a free-form test-case tag (``"nan_inf"``, ``"local_function"``,
    ``"inference"``, ...) used to bucket cases that exercise cross-cutting
    runtime / shape-inference scenarios rather than one specific op.
    """
    return group or "ai.onnx (default)"


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
    for group in sorted(report.summaries):
        s = report.summaries[group]
        label = _group_label(group)
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
    """Renders the per-test-case table for one ``domain`` (or tag group) as a
    reST ``list-table``.

    :param report: The runtime coverage report.
    :param domain: Group key to filter on. This is either an ONNX operator
        domain (``""`` for the default ``ai.onnx`` domain) or a test-case tag
        (e.g. ``"nan_inf"``) under which tagged cases are grouped instead of
        their principal op's domain.
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
        if s.group != domain:
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
    for group in sorted(report.summaries):
        label = _group_label(group)
        lines.append(f".. rubric:: {label}")
        lines.append("")
        table = render_rst_table_for_domain(report, domain=group, css_class=css_class)
        lines.extend(table.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def render_rst_domain_tabs(
    report: RuntimeCoverageReport, css_class: str | None = "sphinx-datatable"
) -> str:
    """Returns :func:`render_rst_domain_sections` output for backward compatibility."""
    return render_rst_domain_sections(report, css_class=css_class)
