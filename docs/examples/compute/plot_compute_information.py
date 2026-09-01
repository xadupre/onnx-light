"""
.. _l-example-plot-compute-information:

Computing shape, tag, constant, release and in-place information
================================================================

*onnx-light* derives several pieces of information about a graph before it
is executed:

* **shape** — the element type and shape of every value
  (:mod:`onnx_light.onnx_core.shape_inference`).
* **shape_tag** — a semantic tag (``shape``, ``axes``, ``weight`` or
  ``ambiguous``) attached to every value and node, guessing what role a
  tensor plays (e.g. a tensor holding a shape versus a numerical weight).
* **constant** — whether a value's content is entirely known before
  inference starts (initializers, ``Constant`` outputs, and outputs of
  deterministic nodes whose inputs are all constant).
* **release** — the last node after which a value is no longer needed and
  its buffer can be released.
* **inplace** — which node outputs can reuse one of their input buffers
  instead of allocating a new one.

This example builds a small model that exercises all five analyses, runs
them through :class:`~onnx_light.onnx_core.shape_inference.ComputeContext`
and the free functions in :mod:`onnx_light.onnx_core.shape_inference`, and
shows how to retrieve the results either as in-memory objects or as
``metadata_props`` entries written directly on the model.

The model computes::

    two        = Constant(2)                     # constant scalar
    const_prod = Mul(W, two)                      # constant (W is an initializer)
    added      = Add(X, const_prod)               # not constant (depends on X)
    relu_out   = Relu(added)                       # may reuse "added" in place
    x_shape    = Shape(X)                          # tagged "shape"
    Z          = Reshape(relu_out, x_shape)        # uses the "shape" tensor
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/compute_information.png"

from __future__ import annotations

import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.shape_inference import (
    INPLACE_REUSE_METADATA_KEY,
    RELEASE_AFTER_METADATA_KEY,
    RELEASE_AFTER_SHAPE_TAG_METADATA_KEY,
    ComputeContext,
    ShapesContext,
    apply_inferred_shapes_to_model,
    compute_inplace_reuse,
    compute_shape_model,
    write_constant_info_to_metadata,
    write_inplace_reuse_to_metadata,
)
from onnx_light.tools import pretty_onnx

# Not exposed as a Python constant yet; mirrors the C++
# ``onnx_compute::kNotUsedAfterMetadataKey`` used by
# :func:`write_inplace_reuse_to_metadata`.
NOT_USED_AFTER_METADATA_KEY = "onnx_light.not_used_after"


def _metadata_value(node, key: str) -> str:
    """Returns the ``metadata_props`` value for ``key`` on ``node``, or ``""``."""
    return next((entry.value for entry in node.metadata_props if entry.key == key), "")


# Make sure the built-in operator schemas are registered before running
# shape inference (the C++ dispatch table looks them up).
defs.register_onnx_operator_set_schema()

#####################################
# Build the model
# +++++++++++++++

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node(
                "Constant",
                [],
                ["two"],
                value=oh.make_tensor("two", onnxl.TensorProto.FLOAT, [], [2.0]),
            ),
            oh.make_node("Mul", ["W", "two"], ["const_prod"]),
            oh.make_node("Add", ["X", "const_prod"], ["added"]),
            oh.make_node("Relu", ["added"], ["relu_out"]),
            oh.make_node("Shape", ["X"], ["x_shape"]),
            oh.make_node("Reshape", ["relu_out", "x_shape"], ["Z"]),
        ],
        "compute_information_demo",
        inputs=[oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3, 4])],
        outputs=[oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, None)],
        initializer=[oh.make_tensor("W", onnxl.TensorProto.FLOAT, [4], [1.0, 2.0, 3.0, 4.0])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)

print(pretty_onnx(model))

#####################################
# 1. shape
# ++++++++
#
# :func:`compute_shape_model` seeds a :class:`ShapesContext` from the model's
# opset, initializers and inputs, then runs per-operator shape inference on
# every node. :func:`apply_inferred_shapes_to_model` writes the result back
# into ``model.graph.value_info`` and ``model.graph.output``.

shapes_ctx = ShapesContext()
compute_shape_model(shapes_ctx, model)
apply_inferred_shapes_to_model(shapes_ctx, model)

print("Inferred shapes:")
for name in ("two", "const_prod", "added", "relu_out", "x_shape", "Z"):
    descriptor = shapes_ctx.get(name)
    print(f"  {name:<12} dtype={descriptor.dtype} shape={descriptor.shape.dims()}")

#####################################
# 2. shape_tag
# ++++++++++++
#
# :meth:`ComputeContext.compute_value_and_node_tags` classifies every value
# and node. ``x_shape`` is tagged ``"shape"`` because it is produced by a
# ``Shape`` node and consumed as a shape argument by ``Reshape``.

compute_ctx = ComputeContext()
value_tags, node_tags = compute_ctx.compute_value_and_node_tags(model.graph)

print("\nValue tags:")
for name, tag in sorted(value_tags.items()):
    print(f"  {name:<12} {tag}")

print("\nNode tags:")
for node, tag in zip(model.graph.node, node_tags):
    print(f"  {node.op_type:<10} outputs={list(node.output)!s:<16} tag={tag}")

#####################################
# 3. constant
# +++++++++++
#
# ``two`` and ``const_prod`` only depend on the ``Constant`` node and the
# initializer ``W``, so both are constant. ``added``, ``relu_out``,
# ``x_shape`` and ``Z`` depend on the graph input ``X`` and are not.
# :func:`write_constant_info_to_metadata` records the result directly on
# the model, under the ``onnx_light.constant`` metadata key.

write_constant_info_to_metadata(model)


def _is_constant(value_infos) -> dict[str, bool]:
    """Returns ``{name: is_constant}`` read from ``onnx_light.constant`` metadata."""
    result = {}
    for value_info in value_infos:
        result[value_info.name] = any(
            entry.key == "onnx_light.constant" for entry in value_info.metadata_props
        )
    return result


constant_values = _is_constant(model.graph.value_info)
constant_values.update(_is_constant(model.graph.initializer))
constant_values.update(_is_constant(model.graph.output))

print("\nConstant values:")
for name in ("W", "two", "const_prod", "added", "relu_out", "x_shape", "Z"):
    print(f"  {name:<12} {constant_values.get(name, False)}")

#####################################
# 4. release / 5. inplace
# ++++++++++++++++++++++++
#
# :func:`compute_inplace_reuse` returns, for every node, the list of
# ``InPlaceReuse`` opportunities (which output can reuse which input
# buffer). :func:`write_inplace_reuse_to_metadata` additionally records
# *release* information on ``metadata_props``:
#
# * ``onnx_light.inplace_reuse`` — the in-place opportunities, as
#   ``output_index:input_index:kind`` triplets.
# * ``onnx_light.release_after`` — the values that are no longer needed
#   once the node has run.
# * ``onnx_light.release_after_shape_tag`` — the subset of those released
#   values that carry the ``"shape"`` tag (from ``value_tags`` above).
# * ``onnx_light.not_used_after`` — declared graph inputs or initializers
#   that reach their last use at this node.
#
# Here ``relu_out = Relu(added)`` can overwrite the buffer of ``added`` in
# place (same element type and shape), and ``added`` is released right
# after ``Relu`` runs since nothing else reads it.

reuse = compute_inplace_reuse(shapes_ctx, model.graph)
write_inplace_reuse_to_metadata(shapes_ctx, model.graph, value_tags)

print("\nIn-place reuse and release information per node:")
for node, node_reuse in zip(model.graph.node, reuse):
    reuse_desc = ", ".join(
        f"out{r.output_index}=in{r.input_index}({r.kind.name})" for r in node_reuse
    )
    inplace_metadata = _metadata_value(node, INPLACE_REUSE_METADATA_KEY)
    release_after = _metadata_value(node, RELEASE_AFTER_METADATA_KEY)
    release_after_shape_tag = _metadata_value(node, RELEASE_AFTER_SHAPE_TAG_METADATA_KEY)
    not_used_after = _metadata_value(node, NOT_USED_AFTER_METADATA_KEY)
    print(
        f"  {node.op_type:<10} outputs={list(node.output)!s:<16} "
        f"inplace=[{reuse_desc}] (metadata={inplace_metadata!r}) "
        f"release_after={release_after!r} "
        f"release_after_shape_tag={release_after_shape_tag!r} "
        f"not_used_after={not_used_after!r}"
    )
