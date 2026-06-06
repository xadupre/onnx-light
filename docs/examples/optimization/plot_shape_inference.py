"""
.. _l-example-plot-shape-inference:

Shape inference with ``onnx_light.onnx_optim``
==============================================

This example shows the two flavours of shape inference exposed by
:mod:`onnx_light.onnx_optim`:

* a single call at the **model** level via
  :func:`onnx_light.onnx_optim.shape_inference.infer_shapes_model`, the
  Python counterpart of the C++ helper ``InferShapesModel``,
* a **node-by-node** walk relying on a manually managed
  :class:`ShapesContext` and on
  ``onnx_light.onnx_py._onnxpy.shape_inference.compute_shape_node``.

Both routes produce the same per-tensor element types and shapes;
the second one is mostly useful when one needs to inspect intermediate
results during shape inference (for example to debug a custom op or to
report the shapes of activations that are not surfaced as
``value_info``).
"""

from __future__ import annotations

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx_optim.shape_inference import infer_shapes_model
from onnx_light.onnx_py._onnxpy import shape_inference as si

# Make sure the built-in operator schemas are registered before running
# shape inference (the C++ dispatch table looks them up).
onnxl.defs.register_onnx_operator_set_schema()


#####################################
# Build a small ``ModelProto``
# ++++++++++++++++++++++++++++
#
# The graph computes ``Y = Relu(X @ W + B)`` with a symbolic batch
# dimension ``N``. ``X`` and ``W`` are graph inputs, ``B`` is an
# initializer.

import numpy as np  # noqa: E402

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("MatMul", ["X", "W"], ["XW"]),
            oh.make_node("Add", ["XW", "B"], ["Z"]),
            oh.make_node("Relu", ["Z"], ["Y"]),
        ],
        "shape_inference_demo",
        inputs=[oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, ["N", 4])],
        outputs=[oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, None)],
        initializer=[
            oh.make_tensor(
                "W", onnxl.TensorProto.FLOAT, [4, 3], np.zeros((4, 3), dtype=np.float32).flatten()
            ),
            oh.make_tensor("B", onnxl.TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32)),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)


#####################################
# Shape inference at the model level
# ++++++++++++++++++++++++++++++++++
#
# A single call to :func:`infer_shapes_model` mutates ``model`` in
# place. The inferred element types and shapes are written back into
# ``model.graph.output`` and ``model.graph.value_info``.

infer_shapes_model(model)

print("Outputs after infer_shapes_model:")
for o in model.graph.output:
    shape = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: dtype={o.type.tensor_type.elem_type}, shape={shape}")

print()
print("value_info after infer_shapes_model:")
for v in model.graph.value_info:
    shape = [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    print(f"  {v.name}: dtype={v.type.tensor_type.elem_type}, shape={shape}")


#####################################
# Shape inference node by node
# ++++++++++++++++++++++++++++
#
# The lower-level API exposes a :class:`ShapesContext` that stores a
# ``name -> OptimTensor`` map. The user seeds the context with the
# graph inputs and initializers and then dispatches each
# :class:`NodeProto` to :func:`compute_shape_node`. This mirrors what
# :func:`infer_shapes_model` does internally, but it gives the user
# the chance to inspect ``ctx`` after every node.

ctx = si.ShapesContext()

# Opset versions are looked up by the per-op shape functions.
for opset in model.opset_import:
    ctx.set_opset_version(opset.domain, opset.version)

# Seed the context with the graph inputs.
for inp in model.graph.input:
    tt = inp.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
    ctx.set(inp.name, si.OptimTensor(tt.elem_type, dims))

# Seed the context with the initializers (constants).
for init in model.graph.initializer:
    ctx.set(init.name, si.OptimTensor(init.data_type, list(init.dims)))

# Walk the graph in topological order.
for node in model.graph.node:
    si.check_inputs_available(ctx, node)
    si.compute_shape_node(ctx, node)
    for out_name in node.output:
        if not out_name:
            continue
        t = ctx.get(str(out_name))
        print(f"after {node.op_type:<7s} -> {out_name}: dtype={t.dtype}, shape={list(t.shape)}")


#####################################
# The two routes agree
# ++++++++++++++++++++
#
# As a sanity check, the per-node walk yields the same descriptor for
# the graph output ``Y`` as the model-level call.

y_model = next(o for o in model.graph.output if o.name == "Y")
y_model_shape = [d.dim_value or d.dim_param for d in y_model.type.tensor_type.shape.dim]
y_node = ctx.get("Y")
print()
print(f"Y (model-level): dtype={y_model.type.tensor_type.elem_type}, shape={y_model_shape}")
print(f"Y (node-by-node): dtype={y_node.dtype}, shape={list(y_node.shape)}")
assert y_model.type.tensor_type.elem_type == y_node.dtype
assert y_model_shape == list(y_node.shape)
