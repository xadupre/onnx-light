"""Standalone helpers built on top of :mod:`onnx_light`.

This sub-package collects small utilities that are not part of the
upstream ``onnx`` API but are useful when working with ONNX models
through :mod:`onnx_light`.

Currently provided helpers:

* :func:`onnx_light.tools.to_dot` -- render a model or a graph as a
  `Graphviz <https://graphviz.org/>`_ DOT string.
* :func:`onnx_light.tools.to_mermaid` -- render a model or a graph as a
  `Mermaid <https://mermaid.js.org/>`_ flowchart.
* :func:`onnx_light.tools.to_svg` -- render a model or a graph as a
  standalone `SVG <https://www.w3.org/Graphics/SVG/>`_ image.
* :func:`onnx_light.tools.compute_value_and_node_tags` -- infer semantic
  value/node tags.
* :func:`onnx_light.tools.write_value_and_node_tags_to_metadata` -- tag
  values and nodes as ``shape``, ``axes``, ``weight`` or ``ambiguous`` in metadata.
* :func:`onnx_light.tools.pretty_onnx` -- render any ONNX proto
  (model, graph, function, node, attribute, value info, tensor) as a
  compact human-readable string.
* :func:`onnx_light.tools.translate` -- translate a model or graph into
  Python code that rebuilds it, either as a compact
  :mod:`onnx_light.onnx.helper` expression (``api="onnx-compact"``) or as
  a :class:`~onnx_light.onnx_core.graph_builder.GraphBuilder` script
  (``api="builder"``).
* :func:`onnx_light.tools.build_kernel_inventory` -- enumerate every
  registered native kernel path and classify it as ``serial``,
  ``parallel_fixed_policy``, ``tunable`` or ``calibratable``.
* :func:`onnx_light.tools.run_kernel_baseline_report` -- run the
  deterministic benchmark corpus and combine it with the kernel inventory
  into one machine-readable cross-machine baseline report.
"""

from __future__ import annotations

from .dot import to_dot, to_dot_graph
from .kernel_baseline import (
    get_cpu_descriptor,
    run_benchmark_corpus,
    run_kernel_baseline_report,
)
from .kernel_inventory import build_kernel_inventory
from .mermaid import to_mermaid, to_mermaid_graph
from .pretty_print import pretty_onnx
from ._proto_utils import (
    compute_value_and_node_tags,
    infer_value_and_node_tags,
    write_value_and_node_tags_to_metadata,
)
from .svg import to_svg, to_svg_graph
from .translate import translate, translate_header

__all__ = [
    "build_kernel_inventory",
    "compute_value_and_node_tags",
    "get_cpu_descriptor",
    "infer_value_and_node_tags",
    "pretty_onnx",
    "run_benchmark_corpus",
    "run_kernel_baseline_report",
    "to_dot",
    "to_dot_graph",
    "to_mermaid",
    "to_mermaid_graph",
    "to_svg",
    "to_svg_graph",
    "translate",
    "translate_header",
    "write_value_and_node_tags_to_metadata",
]
