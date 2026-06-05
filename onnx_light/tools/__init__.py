"""Standalone helpers built on top of :mod:`onnx_light`.

This sub-package collects small utilities that are not part of the
upstream ``onnx`` API but are useful when working with ONNX models
through :mod:`onnx_light`.

Currently provided helpers:

* :func:`onnx_light.tools.to_mermaid` -- render a model or a graph as a
  `Mermaid <https://mermaid.js.org/>`_ flowchart.
"""

from __future__ import annotations

from .mermaid import to_mermaid, to_mermaid_graph

__all__ = ["to_mermaid", "to_mermaid_graph"]
