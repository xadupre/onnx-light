"""Comparison of ONNX :class:`OpSchema` and ``onnx_light`` ``LightOpSchema``.

This module produces a structured comparison between the operator schemas
exposed by the reference :epkg:`onnx` package (``onnx.defs.OpSchema``) and the
lightweight schemas exposed by ``onnx_light`` (``LightOpSchema`` from the C++
``onnx_op`` extension).

For every operator known to either side it reports:

* whether the operator is described by an ``OpSchema`` (``onnx``);
* whether the operator is described by a ``LightOpSchema`` (``onnx_light``);
* whether a shape-inference function is registered on the ``onnx`` side
  (``OpSchema.has_type_and_shape_inference_function``);
* whether a shape-inference function is registered on the ``onnx_light`` side
  (in the ``onnx_optim`` C++ library, see
  :cpp:func:`ComputeShapeNode`);
* how many backend test cases exercise the operator in each package (a test
  case is attributed to the ``op_type`` of the first node of its model — the
  convention used by the ONNX node test suite).

The :func:`compute_schema_comparison` function returns a
:class:`SchemaComparison` describing the rows. The :func:`render_rst_table`
helper turns the comparison into a Sphinx ``list-table`` ready to be embedded
in a documentation page (see ``docs/design/schema_comparison.rst``).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


# Operators whose output shapes can be inferred by ``onnx_optim`` (see
# ``onnx_light/onnx_optim/shapes/shape_inference.cc``). The list is small on
# purpose: ``onnx_optim`` only implements a handful of operators today. Keep
# this set in sync with the dispatch table in ``shape_inference.cc``.
ONNX_OPTIM_SHAPE_INFERENCE_OPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ai.onnx", "Abs"),
        ("ai.onnx", "Add"),
        ("ai.onnx", "And"),
    }
)


@dataclass
class SchemaComparisonRow:
    """One row of :class:`SchemaComparison`, describing a single operator.

    :param domain: Operator domain (``ai.onnx`` for the default ONNX domain).
    :param name: Operator name.
    :param in_onnx: ``True`` when the operator has an ``OpSchema`` in
        :mod:`onnx.defs`.
    :param in_onnx_light: ``True`` when the operator has a ``LightOpSchema``
        in the ``onnx_op`` C++ extension.
    :param onnx_shape_inference: ``True`` when ``onnx`` registers a type and
        shape inference function for the operator.
    :param onnx_light_shape_inference: ``True`` when ``onnx_optim`` registers
        a shape-inference dispatch entry for the operator.
    :param onnx_backend_tests: Number of node backend tests in :mod:`onnx`
        whose first node is the operator.
    :param onnx_light_backend_tests: Number of node backend tests collected by
        :func:`onnx_light.backend.test.case.base.collect_test_case` whose
        first node is the operator.
    """

    domain: str
    name: str
    in_onnx: bool = False
    in_onnx_light: bool = False
    onnx_shape_inference: bool = False
    onnx_light_shape_inference: bool = False
    onnx_backend_tests: int = 0
    onnx_light_backend_tests: int = 0


@dataclass
class SchemaComparison:
    """Aggregated comparison between ``onnx`` and ``onnx_light`` operators.

    :param rows: One :class:`SchemaComparisonRow` per ``(domain, op_name)``
        known to either side, sorted by ``(domain, name)``.
    :param onnx_available: ``True`` when the reference :mod:`onnx` package is
        importable and was used to populate the ``onnx``-side columns.
    """

    rows: list[SchemaComparisonRow] = field(default_factory=list)
    onnx_available: bool = True

    @property
    def total_onnx(self) -> int:
        """Number of operators with an ``OpSchema``."""
        return sum(1 for r in self.rows if r.in_onnx)

    @property
    def total_onnx_light(self) -> int:
        """Number of operators with a ``LightOpSchema``."""
        return sum(1 for r in self.rows if r.in_onnx_light)

    @property
    def total_onnx_shape_inference(self) -> int:
        """Operators with shape inference on the ``onnx`` side."""
        return sum(1 for r in self.rows if r.onnx_shape_inference)

    @property
    def total_onnx_light_shape_inference(self) -> int:
        """Operators with shape inference on the ``onnx_light`` (``onnx_optim``) side."""
        return sum(1 for r in self.rows if r.onnx_light_shape_inference)

    @property
    def total_onnx_backend_tests(self) -> int:
        """Total number of ``onnx`` node backend tests counted."""
        return sum(r.onnx_backend_tests for r in self.rows)

    @property
    def total_onnx_light_backend_tests(self) -> int:
        """Total number of ``onnx_light`` node backend tests counted."""
        return sum(r.onnx_light_backend_tests for r in self.rows)


def _count_onnx_light_backend_tests() -> Counter[tuple[str, str]]:
    """Counts ``onnx_light`` node backend tests by ``(domain, first_op_type)``."""
    from .backend.test.case.base import collect_test_case

    counts: Counter[tuple[str, str]] = Counter()
    for tc in collect_test_case().values():
        model = tc.model
        if model is None:
            continue
        nodes = list(model.graph.node)
        if not nodes:
            continue
        n = nodes[0]
        counts[(n.domain or "ai.onnx", n.op_type)] += 1
    return counts


def _light_schemas_latest() -> dict[tuple[str, str], Any]:
    """Returns ``{(domain, name): schema}`` keeping only the latest version."""
    from .onnx_py._onnxpy import onnx_op as _op  # type: ignore[attr-defined]

    latest: dict[tuple[str, str], Any] = {}
    for s in _op.GetAllOnnxOpSchemasWithHistory():
        key = (s.domain, s.name)
        if key not in latest or s.since_version > latest[key].since_version:
            latest[key] = s
    return latest


def _count_onnx_backend_tests() -> Counter[tuple[str, str]]:
    """Counts upstream ``onnx`` node backend tests by ``(domain, first_op_type)``.

    Returns an empty counter when the upstream ``onnx`` package (or its
    backend test data) is not available.
    """
    try:
        import onnx
        from onnx.backend.test.loader import load_model_tests
    except ImportError:
        return Counter()

    try:
        tests = load_model_tests(kind="node")
    except Exception:  # pragma: no cover - defensive
        return Counter()

    counts: Counter[tuple[str, str]] = Counter()
    for t in tests:
        model_path = os.path.join(t.model_dir, "model.onnx") if t.model_dir else None
        if not model_path or not os.path.exists(model_path):
            continue
        try:
            m = onnx.load(model_path, load_external_data=False)
        except Exception:  # pragma: no cover - defensive
            continue
        if not m.graph.node:
            continue
        n = m.graph.node[0]
        counts[(n.domain or "ai.onnx", n.op_type)] += 1
    return counts


def _onnx_schemas_with_shape_inference() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Returns ``(all_op_keys, shape_inference_op_keys)`` from :mod:`onnx.defs`.

    Both sets are empty when the upstream ``onnx`` package is not importable.
    """
    try:
        from onnx import defs
    except ImportError:
        return set(), set()

    all_ops: set[tuple[str, str]] = set()
    with_shape: set[tuple[str, str]] = set()
    for s in defs.get_all_schemas():
        key = (s.domain or "ai.onnx", s.name)
        all_ops.add(key)
        if s.has_type_and_shape_inference_function:
            with_shape.add(key)
    return all_ops, with_shape


def compute_schema_comparison() -> SchemaComparison:
    """Computes the full :class:`SchemaComparison`.

    The function silently degrades when the upstream :mod:`onnx` package is
    not importable: the ``onnx``-side columns are then left empty and
    :attr:`SchemaComparison.onnx_available` is set to ``False``.
    """
    light_schemas = _light_schemas_latest()
    light_tests = _count_onnx_light_backend_tests()
    onnx_all, onnx_with_shape = _onnx_schemas_with_shape_inference()
    onnx_tests = _count_onnx_backend_tests()
    onnx_available = bool(onnx_all)

    all_keys: set[tuple[str, str]] = set(light_schemas) | onnx_all | set(light_tests) | set(
        onnx_tests
    )

    rows: list[SchemaComparisonRow] = []
    for key in sorted(all_keys):
        domain, name = key
        rows.append(
            SchemaComparisonRow(
                domain=domain,
                name=name,
                in_onnx=key in onnx_all,
                in_onnx_light=key in light_schemas,
                onnx_shape_inference=key in onnx_with_shape,
                onnx_light_shape_inference=key in ONNX_OPTIM_SHAPE_INFERENCE_OPS,
                onnx_backend_tests=int(onnx_tests.get(key, 0)),
                onnx_light_backend_tests=int(light_tests.get(key, 0)),
            )
        )
    return SchemaComparison(rows=rows, onnx_available=onnx_available)


def _yn(value: bool) -> str:
    """Renders a boolean as ``yes``/``no`` for table cells."""
    return "yes" if value else "no"


def render_rst_table(comparison: SchemaComparison, only_in_either: bool = True) -> str:
    """Renders *comparison* as a reST ``list-table`` directive.

    :param comparison: The comparison to render.
    :param only_in_either: When ``True`` (the default), operators that are
        absent from both ``onnx`` and ``onnx_light`` (this can happen for
        custom-domain test fixtures) are filtered out.
    :returns: A multi-line string containing the directive, ready to be
        emitted in a ``runpython`` block.
    """
    header = (
        ".. list-table::\n"
        "    :header-rows: 1\n"
        "    :widths: 12 18 8 8 14 14 13 13\n"
        "\n"
        "    * - Domain\n"
        "      - Operator\n"
        "      - ``onnx``\n"
        "      - ``onnx_light``\n"
        "      - ``onnx`` shape inference\n"
        "      - ``onnx_light`` shape inference (``onnx_optim``)\n"
        "      - ``onnx`` backend tests\n"
        "      - ``onnx_light`` backend tests\n"
    )
    body_parts: list[str] = []
    for r in comparison.rows:
        if only_in_either and not r.in_onnx and not r.in_onnx_light:
            continue
        body_parts.append(
            "    * - "
            + r.domain
            + "\n      - "
            + r.name
            + "\n      - "
            + _yn(r.in_onnx)
            + "\n      - "
            + _yn(r.in_onnx_light)
            + "\n      - "
            + _yn(r.onnx_shape_inference)
            + "\n      - "
            + _yn(r.onnx_light_shape_inference)
            + "\n      - "
            + str(r.onnx_backend_tests)
            + "\n      - "
            + str(r.onnx_light_backend_tests)
            + "\n"
        )
    return header + "".join(body_parts)


def _html_escape(text: str) -> str:
    """Minimal HTML escape for table cells."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html_table(
    comparison: SchemaComparison,
    only_in_either: bool = True,
    table_id: str = "onnx-light-schema-comparison",
) -> str:
    """Renders *comparison* as a sortable + searchable HTML table.

    The output is wrapped in a Sphinx ``.. raw:: html`` directive so it can
    be embedded directly in a ``runpython`` block. The table carries the
    ``onnx-light-sortable`` CSS class and is paired with a search input
    referencing ``table_id``; client-side behaviour is provided by
    ``docs/_static/sortable_table.js`` (sortable headers, full-text row
    filter).

    :param comparison: The comparison to render.
    :param only_in_either: When ``True`` (the default), operators that are
        absent from both ``onnx`` and ``onnx_light`` are filtered out (same
        behaviour as :func:`render_rst_table`).
    :param table_id: ``id`` attribute used by the search input to find the
        table; must be a valid HTML id and unique on the rendered page.
    :returns: A multi-line string starting with ``.. raw:: html``.
    """
    headers = (
        "Domain",
        "Operator",
        "onnx",
        "onnx_light",
        "onnx shape inference",
        "onnx_light shape inference (onnx_optim)",
        "onnx backend tests",
        "onnx_light backend tests",
    )
    lines: list[str] = [
        ".. raw:: html",
        "",
        '    <div class="onnx-light-table-toolbar">',
        (
            '      <input type="search" class="onnx-light-table-filter" '
            f'data-table-target="{_html_escape(table_id)}" '
            'placeholder="Filter operators (e.g. Conv, ai.onnx, yes) ..." '
            'aria-label="Filter operators">'
        ),
        "    </div>",
        f'    <table id="{_html_escape(table_id)}" class="onnx-light-sortable docutils">',
        "      <thead>",
        "        <tr>",
    ]
    for h in headers:
        lines.append(f"          <th>{_html_escape(h)}</th>")
    lines.extend(["        </tr>", "      </thead>", "      <tbody>"])
    for r in comparison.rows:
        if only_in_either and not r.in_onnx and not r.in_onnx_light:
            continue
        cells = (
            r.domain,
            r.name,
            _yn(r.in_onnx),
            _yn(r.in_onnx_light),
            _yn(r.onnx_shape_inference),
            _yn(r.onnx_light_shape_inference),
            str(r.onnx_backend_tests),
            str(r.onnx_light_backend_tests),
        )
        lines.append("        <tr>")
        for c in cells:
            lines.append(f"          <td>{_html_escape(c)}</td>")
        lines.append("        </tr>")
    lines.extend(["      </tbody>", "    </table>", ""])
    return "\n".join(lines) + "\n"


def render_rst_summary(comparison: SchemaComparison) -> str:
    """Renders a short reST summary (totals) for *comparison*."""
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 50 25 25",
        "",
        "    * - Metric",
        "      - ``onnx``",
        "      - ``onnx_light``",
        "    * - Operators with a schema",
        f"      - {comparison.total_onnx}",
        f"      - {comparison.total_onnx_light}",
        "    * - Operators with shape inference",
        f"      - {comparison.total_onnx_shape_inference}",
        f"      - {comparison.total_onnx_light_shape_inference}",
        "    * - Node backend tests (counted)",
        f"      - {comparison.total_onnx_backend_tests}",
        f"      - {comparison.total_onnx_light_backend_tests}",
    ]
    return "\n".join(lines) + "\n"


def iter_rows(comparison: SchemaComparison) -> Iterable[SchemaComparisonRow]:
    """Iterates over ``comparison.rows`` (convenience helper)."""
    yield from comparison.rows
