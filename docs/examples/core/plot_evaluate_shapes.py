"""
.. _l-example-plot-evaluate-shapes:

Evaluating inferred shapes with concrete input dimensions
=========================================================

ONNX models commonly use *symbolic* names — sometimes called *tokens* —
for the dimensions of their inputs (``"batch"``, ``"seq_length"``,
``"d_model"``, ...).  After running shape inference, every intermediate
tensor's shape is expressed in terms of those input tokens, either as a
plain dim_param (``"batch"``) or as a symbolic arithmetic expression
(``"2*d_model"``, ``"batch*seq_length"``, ...).

This example shows how to:

1. Build a small model whose inputs carry symbolic dimensions.
2. Run model-level shape inference with
   :func:`~onnx_light.onnx_core.shape_inference.infer_shapes_model` to
   populate ``model.graph.value_info`` and ``model.graph.output`` with
   symbolic shapes.
3. Evaluate every inferred dimension to a concrete integer for several
   assignments of the input tokens, using
   :func:`~onnx_light.onnx_core.expressions.evaluate_expression`.

The same approach is useful in profiling and capacity-planning workflows
where one wants to know the concrete tensor sizes a model produces for a
given batch size, sequence length, hidden dimension, etc., without
actually running the model.
"""

from __future__ import annotations

import onnx_light.onnx as onnxl
import onnx_light.onnx.defs as defs
import onnx_light.onnx.helper as oh
from onnx_light.onnx_core.expressions import evaluate_expression
from onnx_light.onnx_core.shape_inference import infer_shapes_model

# Built-in operator schemas must be registered before shape inference
# (the C++ dispatch table looks them up).
defs.register_onnx_operator_set_schema()


#####################################
# Build a model with symbolic input dimensions
# ++++++++++++++++++++++++++++++++++++++++++++
#
# The graph mirrors a tiny transformer-style block:
#
# .. code-block:: none
#
#    added      = Add(X, Y)                       # [batch, seq, d_model]
#    concat_out = Concat(added, X, axis=2)        # [batch, seq, 2*d_model]
#    Z          = Reshape(concat_out, [0, 0, -1]) # [batch, seq, 2*d_model]
#
# Every input dimension is a symbolic *token*: ``"batch"``, ``"seq"`` and
# ``"d_model"``.  None of them carries a concrete integer value at the
# time the model is built.

INPUT_TOKENS = ["batch", "seq", "d_model"]

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["X", "Y"], ["added"]),
            oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
            oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
        ],
        "evaluate_shapes_demo",
        inputs=[
            oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, INPUT_TOKENS),
            oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, INPUT_TOKENS),
        ],
        outputs=[oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, None)],
        initializer=[oh.make_tensor("reshape_shape", onnxl.TensorProto.INT64, [3], [0, 0, -1])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)


#####################################
# Run shape inference
# +++++++++++++++++++
#
# :func:`infer_shapes_model` mutates ``model`` in place and writes the
# inferred symbolic shapes into ``model.graph.value_info`` (intermediate
# tensors) and ``model.graph.output`` (graph outputs).  Because the
# Reshape target ``[0, 0, -1]`` is propagated from the initializer, the
# last dimension of ``concat_out`` and ``Z`` is resolved symbolically to
# ``"2*d_model"``.

infer_shapes_model(model)


def dims_of(value_info):
    """Returns the list of dim values/params for a ``ValueInfoProto``."""
    return [
        d.dim_value if d.dim_value else d.dim_param for d in value_info.type.tensor_type.shape.dim
    ]


# Collect symbolic shapes for every named tensor in the graph: inputs,
# intermediate value_info entries, and graph outputs.
symbolic_shapes: dict[str, list] = {}
for vi in model.graph.input:
    symbolic_shapes[vi.name] = dims_of(vi)
for vi in model.graph.value_info:
    symbolic_shapes[vi.name] = dims_of(vi)
for vi in model.graph.output:
    symbolic_shapes[vi.name] = dims_of(vi)

print("Symbolic shapes after inference:")
for name, shape in symbolic_shapes.items():
    print(f"  {name:<12} -> {shape}")


#####################################
# Evaluate every shape for concrete input values
# ++++++++++++++++++++++++++++++++++++++++++++++
#
# Given an assignment of the input tokens to concrete integers, every
# dimension of every tensor can be evaluated:
#
# * Integer dims are returned unchanged.
# * String dims that match an input token are looked up directly.
# * String dims that are arithmetic expressions (e.g. ``"2*d_model"``)
#   are reduced with
#   :func:`~onnx_light.onnx_core.expressions.evaluate_expression`,
#   which accepts the mapping from token names to integer values.


def evaluate_shape(shape: list, assignment: "dict[str, int]") -> list[int]:
    """Returns *shape* with every symbolic dim replaced by its integer value."""
    concrete: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            concrete.append(dim)
        elif dim in assignment:
            concrete.append(assignment[dim])
        else:
            # Symbolic expression such as ``"2*d_model"``.
            concrete.append(evaluate_expression(str(dim), assignment))
    return concrete


# Three different input-token assignments. The first one matches a tiny
# debug configuration; the next two scale the batch and sequence length.
ASSIGNMENTS = [
    {"batch": 1, "seq": 8, "d_model": 16},
    {"batch": 4, "seq": 128, "d_model": 64},
    {"batch": 32, "seq": 512, "d_model": 768},
]

print("\nConcrete shapes for each input-token assignment:")
header_assignments = [
    "{" + ", ".join(f"{k}={v}" for k, v in a.items()) + "}" for a in ASSIGNMENTS
]
print(f"  {'tensor':<12} " + " ".join(f"{h:>28}" for h in header_assignments))
for name, shape in symbolic_shapes.items():
    cells = [str(evaluate_shape(shape, a)) for a in ASSIGNMENTS]
    print(f"  {name:<12} " + " ".join(f"{c:>28}" for c in cells))


#####################################
# Per-tensor element count
# ++++++++++++++++++++++++
#
# A common follow-up question is *"how many elements does each tensor
# hold for this configuration?"*.  Once the shape has been evaluated to
# concrete integers the answer is just the product of the dimensions.

from math import prod  # noqa: E402

print("\nElement counts per tensor and per assignment:")
print(f"  {'tensor':<12} " + " ".join(f"{h:>28}" for h in header_assignments))
for name, shape in symbolic_shapes.items():
    cells = [str(prod(evaluate_shape(shape, a))) for a in ASSIGNMENTS]
    print(f"  {name:<12} " + " ".join(f"{c:>28}" for c in cells))
