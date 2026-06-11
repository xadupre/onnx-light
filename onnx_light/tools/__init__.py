"""Standalone helpers built on top of :mod:`onnx_light`.

This sub-package collects small utilities that are not part of the
upstream ``onnx`` API but are useful when working with ONNX models
through :mod:`onnx_light`.

Currently provided helpers:

* :func:`onnx_light.tools.to_mermaid` -- render a model or a graph as a
  `Mermaid <https://mermaid.js.org/>`_ flowchart.
* :func:`onnx_light.tools.pretty_print` -- render a model, graph,
  function or node as human-readable text.
"""

from __future__ import annotations

from .mermaid import to_mermaid, to_mermaid_graph
from .pretty_print import pretty_print, pretty_print_graph, pretty_print_node

__all__ = [
    "pretty_print",
    "pretty_print_graph",
    "pretty_print_node",
    "to_mermaid",
    "to_mermaid_graph",
]
