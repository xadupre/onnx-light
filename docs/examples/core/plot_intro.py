"""
.. _l-plot-onnx-light-intro:

Introduction to onnx-light
===========================

This example gives a brief overview of *onnx-light*, a library that
works with ONNX models without requiring protobuf.

It demonstrates the use of the ``.. runpython::`` directive (executed at
documentation build time) and a ``.. mermaid::`` diagram.

.. mermaid::

    graph LR
        A[Input tensor] --> B[ONNX Graph]
        B --> C[Output tensor]
        B --> D[onnx-light<br/>no protobuf]
        D --> C
"""

# %%
# Package version
# ---------------
#
# The snippet below is executed with ``.. runpython::`` so the output
# appears verbatim in the generated HTML page.

import onnx_light

print(f"onnx-light version: {onnx_light.__version__}")

# %%
# What is onnx-light?
# --------------------
#
# *onnx-light* manipulates ONNX models without the overhead of the
# protobuf serialization layer.  The package exposes the same
# high-level abstractions (nodes, graphs, initializers) through
# plain Python data-classes, making it easier to embed in
# memory-constrained or dependency-free environments.
