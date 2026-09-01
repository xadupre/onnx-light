"""
.. _l-example-plot-shape-inference-custom-op:

Shape inference with a custom operator
======================================

Most ONNX shape inference relies on the built-in dispatch table that
knows the schemas of standard operators. Real-world models often include
custom operators from a private domain. For those, the built-in
inference has no way to compute output shapes and stops with an error.

*onnx-light* lets users register a Python callback to handle the shape
inference of a custom operator. The callback receives the current
:class:`ShapesContext` and the node, and is responsible for setting an
:class:`SymTensor` for every output. Once registered, the callback is
invoked automatically by
:func:`~onnx_light.onnx_core.shape_inference.compute_shape_node`
(and by the graph/model variants) whenever a node with the matching
``(domain, op_type)`` is encountered.

The example below defines a tiny ``ScaledLinear`` operator in the
``com.example`` domain. Its semantics are the ones of a fused
``MatMul + Add + scalar multiplication``:

.. code-block:: none

    Y = scale * (X @ W + B)

where:

* ``X`` has shape ``[batch, in_features]``
* ``W`` has shape ``[in_features, out_features]``
* ``B`` has shape ``[out_features]``
* ``scale`` is a float attribute
* ``Y`` has shape ``[batch, out_features]`` and the same dtype as ``X``
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/shape_inference_custom_op.png"

from __future__ import annotations
import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
from onnx_light.onnx_py._onnxpycore import shape_inference as si

# Make sure the built-in operator schemas are registered before running
# shape inference (the C++ dispatch table looks them up for standard ops).
defs.register_onnx_operator_set_schema()


#####################################
# Build a model that uses the custom operator
# +++++++++++++++++++++++++++++++++++++++++++
#
# The graph applies a standard `Relu` followed by the custom
# ``ScaledLinear`` operator from the ``com.example`` domain. The
# Relu output keeps the input shape ``[N, 4]`` and feeds into
# ``ScaledLinear`` together with a weight matrix ``W`` of shape
# ``[4, 3]`` and a bias ``B`` of shape ``[3]``. The expected output
# shape for ``Y`` is therefore ``[N, 3]``.

CUSTOM_DOMAIN = "com.example"
CUSTOM_OP = "ScaledLinear"

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Relu", ["X"], ["X_relu"]),
            oh.make_node(CUSTOM_OP, ["X_relu", "W", "B"], ["Y"], domain=CUSTOM_DOMAIN, scale=2.0),
        ],
        "custom_op_shape_inference_demo",
        inputs=[
            oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4]),
            oh.make_tensor_value_info("W", onnxl.TensorProto.FLOAT, [4, 3]),
            oh.make_tensor_value_info("B", onnxl.TensorProto.FLOAT, [3]),
        ],
        outputs=[oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(CUSTOM_DOMAIN, 1)],
    ir_version=8,
)


def seed_context(model) -> si.ShapesContext:
    """Returns a fresh ShapesContext seeded with the model inputs and opsets."""
    ctx = si.ShapesContext()
    for opset in model.opset_import:
        ctx.set_opset_version(opset.domain, opset.version)
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
        ctx.set(inp.name, si.SymTensor(tt.elem_type, dims))
    return ctx


#####################################
# Without a callback: inference fails
# +++++++++++++++++++++++++++++++++++
#
# When no callback is registered for ``(com.example, ScaledLinear)``,
# :func:`compute_shape_node` cannot find a shape function for the
# operator and raises :class:`ValueError`. The example catches the
# error to keep the gallery script running.

ctx = seed_context(model)
relu_node, custom_node = list(model.graph.node)

si.compute_shape_node(ctx, relu_node)
print("After Relu, X_relu shape:", list(ctx.get("X_relu").shape))

print(
    "\nCustom op already registered before any call:",
    ctx.has_custom_shape_inference_function(CUSTOM_DOMAIN, CUSTOM_OP),
)

try:
    si.compute_shape_node(ctx, custom_node)
except ValueError as e:
    print(f"\nAs expected, inference failed without a callback:\n  {e}")


#####################################
# Define and register the custom shape function
# +++++++++++++++++++++++++++++++++++++++++++++
#
# The callback receives ``(ctx, node)`` where ``ctx`` is the current
# :class:`ShapesContext` and ``node`` is the :class:`NodeProto` being
# processed. It must read the input descriptors from the context and
# call :meth:`ShapesContext.set` for every output of the node.
#
# Best-practice checks performed below:
#
# * the operator receives 3 inputs and produces 1 output;
# * ``X`` is 2-D and ``W`` is 2-D;
# * the contracting dimensions of ``X`` and ``W`` agree when both are
#   concrete integers;
# * ``B`` is 1-D and matches the second dimension of ``W``;
# * the output dtype is the dtype of ``X``.


def infer_scaled_linear(ctx: si.ShapesContext, node) -> None:
    """Infers the output shape of the ``com.example::ScaledLinear`` operator."""
    if len(node.input) != 3:
        raise ValueError(f"{CUSTOM_OP} expects 3 inputs (X, W, B), got {len(node.input)}.")
    if len(node.output) != 1:
        raise ValueError(f"{CUSTOM_OP} expects 1 output (Y), got {len(node.output)}.")

    x = ctx.get(str(node.input[0]))
    w = ctx.get(str(node.input[1]))
    b = ctx.get(str(node.input[2]))

    x_shape = list(x.shape)
    w_shape = list(w.shape)
    b_shape = list(b.shape)

    if len(x_shape) != 2:
        raise ValueError(f"{CUSTOM_OP}: X must be 2-D, got shape {x_shape}.")
    if len(w_shape) != 2:
        raise ValueError(f"{CUSTOM_OP}: W must be 2-D, got shape {w_shape}.")
    if len(b_shape) != 1:
        raise ValueError(f"{CUSTOM_OP}: B must be 1-D, got shape {b_shape}.")

    # Check the contracting dimension only when both sides are concrete ints.
    if isinstance(x_shape[1], int) and isinstance(w_shape[0], int):
        if x_shape[1] != w_shape[0]:
            raise ValueError(
                f"{CUSTOM_OP}: contracting dims disagree: "
                f"X[1]={x_shape[1]} vs W[0]={w_shape[0]}."
            )
    if isinstance(b_shape[0], int) and isinstance(w_shape[1], int):
        if b_shape[0] != w_shape[1]:
            raise ValueError(f"{CUSTOM_OP}: B[0]={b_shape[0]} must match W[1]={w_shape[1]}.")

    out_shape = [x_shape[0], w_shape[1]]
    ctx.set(str(node.output[0]), si.SymTensor(x.dtype, out_shape))


ctx.set_custom_shape_inference_function(CUSTOM_DOMAIN, CUSTOM_OP, infer_scaled_linear)
print("\nRegistered callbacks:", list(ctx.custom_shape_inference_keys()))


#####################################
# With the callback: inference succeeds
# +++++++++++++++++++++++++++++++++++++
#
# Re-running :func:`compute_shape_node` on the same node now dispatches
# to ``infer_scaled_linear`` and writes the inferred descriptor for
# ``Y`` into the context.

si.compute_shape_node(ctx, custom_node)
y = ctx.get("Y")
print("\nAfter ScaledLinear, Y shape:", list(y.shape), "dtype:", y.dtype)
assert list(y.shape) == ["N", 3]
assert y.dtype == onnxl.TensorProto.FLOAT


#####################################
# End-to-end inference on the whole model
# +++++++++++++++++++++++++++++++++++++++
#
# The same callback works with :func:`compute_shape_model`, which
# walks the whole graph and infers every intermediate tensor in one
# call. The callback is registered on the context *before*
# :func:`compute_shape_model` is invoked.

ctx2 = si.ShapesContext()
ctx2.set_custom_shape_inference_function(CUSTOM_DOMAIN, CUSTOM_OP, infer_scaled_linear)
si.compute_shape_model(ctx2, model)

print("\nWhole-model inference results:")
for name in ["X", "X_relu", "Y"]:
    t = ctx2.get(name)
    print(f"  {name:<8} shape={list(t.shape)} dtype={t.dtype}")

assert list(ctx2.get("Y").shape) == ["N", 3]


#####################################
# Takeaways
# +++++++++
#
# * Custom operators are supported by
#   :func:`compute_shape_node`/:func:`compute_shape_model` via the
#   :meth:`ShapesContext.set_custom_shape_inference_function` registration.
# * The callback signature is ``fn(ctx, node) -> None``; it must call
#   ``ctx.set(name, SymTensor(dtype, shape))`` for each output of the node.
# * Validate input ranks and dimension compatibility inside the callback to
#   catch malformed graphs early and to keep symbolic dims working when
#   concrete sizes are not yet known.
