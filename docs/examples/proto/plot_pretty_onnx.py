"""
.. _l-example-plot-pretty-onnx:

pretty_onnx: shape info, shape tags, inplace and release annotations
======================================================================

:func:`~onnx_light.tools.pretty_onnx` renders any ONNX proto as a
compact, human-readable text listing.  Beyond the basic node-by-node
view it supports four optional annotation layers:

* **shape info** — run shape inference before rendering so every input,
  intermediate and output tensor shows its inferred dtype and shape.
* **shape tags** — prefix nodes (and highlight tensors) whose role is
  identified as ``shape``, ``axes`` or ``weight`` by the semantic tag
  inference.
* **inplace info** — annotate nodes whose output can safely reuse an
  input's buffer (``onnx_light.inplace_reuse`` metadata).
* **release info** — annotate nodes after which a tensor is no longer
  needed and its memory can be freed (``onnx_light.release_after``
  metadata).

The example below builds a small graph, enriches it with all four kinds
of metadata and then shows the output of :func:`~onnx_light.tools.pretty_onnx`
at each level of verbosity.

Graph structure
++++++++++++++++

.. code-block:: none

   X : float[N, 4]
   S  = Shape(X)       → int64[2]   # S is a "shape" tensor
   A  = Abs(X)         → float[N, 4]
   B  = Relu(A)        → float[N, 4]  # B can reuse A's buffer (inplace)
   Z  = Reshape(B, S)  → float[4, N] or float[N, 4] after inference
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/pretty_onnx.png"

from __future__ import annotations

import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.shape_inference import (
    ShapesContext,
    apply_inferred_shapes_to_model,
    compute_shape_model,
    write_inplace_reuse_to_metadata,
    write_value_and_node_tags_to_metadata,
)
from onnx_light.tools import pretty_onnx

# Built-in operator schemas must be registered before shape inference.
defs.register_onnx_operator_set_schema()


#####################################
# Build the model
# +++++++++++++++
#
# The graph is intentionally simple so that all four annotation layers are
# visible on just a handful of nodes.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Shape", ["X"], ["S"]),
            oh.make_node("Abs", ["X"], ["A"]),
            oh.make_node("Relu", ["A"], ["B"]),
            oh.make_node("Reshape", ["B", "S"], ["Z"]),
        ],
        "pretty_onnx_demo",
        inputs=[oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])],
        outputs=[oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)

# %%
# Plain rendering — no annotations.

print("=== plain (no annotations) ===")
print(pretty_onnx(model))


#####################################
# Shape info
# ++++++++++
#
# Pass ``shape_inference=True`` to run :mod:`onnx_light.onnx_core.shape_inference`
# shape inference before rendering.  After inference every value in
# ``model.graph.value_info`` and ``model.graph.output`` carries its inferred
# dtype and shape, which :func:`~onnx_light.tools.pretty_onnx` then shows
# next to every input/output/intermediate name.
#
# .. note::
#   ``shape_inference=True`` creates a *copy* of the model with the
#   inferred shapes filled in; the original ``model`` object is not mutated.

print("\n=== shape_inference=True ===")
print(pretty_onnx(model, shape_inference=True))


#####################################
# Shape tags
# ++++++++++
#
# Shape-tag inference labels every tensor and node with a semantic role:
#
# * ``shape`` — a tensor whose value represents a shape or size (e.g. the
#   output of a ``Shape`` node).
# * ``axes`` — a tensor whose value represents a set of axis indices.
# * ``weight`` — a trainable parameter (initializer).
#
# :func:`~onnx_light.onnx_core.shape_inference.write_value_and_node_tags_to_metadata`
# writes these labels into ``metadata_props``; then
# ``include_node_tags=True`` renders them as ``[shape]``, ``[axes]`` or
# ``[weight]`` prefixes on the relevant node lines.

write_value_and_node_tags_to_metadata(model.graph)

print("\n=== include_node_tags=True (after write_value_and_node_tags_to_metadata) ===")
print(pretty_onnx(model, include_node_tags=True))


#####################################
# Inplace info
# ++++++++++++
#
# Buffer-reuse analysis identifies pairs of node inputs and outputs that
# are *aliasable*: the output can overwrite the input's allocation when the
# shapes match and the input is not needed by any later node.
#
# :func:`~onnx_light.onnx_core.shape_inference.write_inplace_reuse_to_metadata`
# records each reuse opportunity under the ``onnx_light.inplace_reuse``
# metadata key as a ``out_idx:in_idx:kind`` triplet.  With
# ``include_inplace=True`` these are rendered as
# ``inplace: out0=in0(equal)`` (or ``(greater)`` when the output buffer is
# strictly larger than the input).
#
# Shape inference must be run first so the analysis has concrete shapes to
# compare.

ctx = ShapesContext()
compute_shape_model(ctx, model)
apply_inferred_shapes_to_model(ctx, model)
write_inplace_reuse_to_metadata(ctx, model.graph)

print("\n=== include_inplace=True (after write_inplace_reuse_to_metadata) ===")
print(pretty_onnx(model, include_inplace=True))


#####################################
# Release info
# ++++++++++++
#
# The same ``write_inplace_reuse_to_metadata`` call also records, for each
# node, the set of tensors whose last use ends at that node.  A runtime can
# free those buffers immediately after the node executes.  The names are
# stored under the ``onnx_light.release_after`` metadata key as a
# ``;``-separated list.  With ``include_release=True`` they are rendered
# as ``release: A`` (or ``release: A, B`` when multiple tensors are freed).

print("\n=== include_release=True ===")
print(pretty_onnx(model, include_release=True))


#####################################
# All annotations combined
# +++++++++++++++++++++++++
#
# The four flags are independent and compose freely.  The listing below
# enables all of them at once so you can see the full picture in a single
# pass.

print("\n=== all annotations combined ===")
print(pretty_onnx(model, include_node_tags=True, include_inplace=True, include_release=True))
