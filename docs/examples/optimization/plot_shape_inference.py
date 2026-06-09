"""
.. _l-example-plot-shape-inference:

Optimized Shape inference
=========================

This example compares three approaches to shape inference on a graph that
mirrors a common transformer pattern:

.. code-block:: none

   added      = Add(X, Y)              # [batch, seq, d_model]
   concat_out = Concat(added, X, axis=2)  # [batch, seq, 2·d_model]
   Z          = Reshape(concat_out, [0, 0, -1])  # [batch, seq, 2·d_model]

The ``Reshape`` target shape ``[0, 0, -1]`` means "keep dim 0, keep dim 1,
infer dim 2 from the element count".  Its values are stored in an
initializer tensor named ``reshape_shape``.

The three approaches compared here are:

* **model-level** via :func:`~onnx_light.onnx_optim.shape_inference.infer_shapes_model` —
  internally reads initializer *values*, so ``Z`` is fully resolved.
* **node-by-node (naïve)** — seeds the :class:`ShapesContext` with
  initializer *shapes* only; the ``[0, 0, -1]`` values are not
  propagated, so the Reshape output carries symbolic placeholders.
* **node-by-node (with value propagation)** — additionally calls
  :meth:`~onnx_light.onnx_py._onnxpy.shape_inference.OptimTensor.set_value_as_shape`
  for each initializer, enabling full resolution of ``Z``.

The final plot shows the last inferred dimension for each tensor under
every approach, making the divergence for ``Z`` immediately visible.

Understanding value_info
+++++++++++++++++++++++++

ONNX models store inferred shapes in ``model.graph.value_info``, which is a
list of ``ValueInfoProto`` messages. Each entry associates a tensor name with
its type and shape. When you call
:func:`~onnx_light.onnx_optim.shape_inference.infer_shapes_model`, the
function mutates the model in place and populates ``value_info`` with the
inferred shapes for all intermediate tensors. Graph inputs and outputs store
their shapes directly in ``model.graph.input`` and ``model.graph.output``.

The example below also shows how to use :func:`make_test_class` from
:mod:`onnx_light.backend.test.case` to systematically validate shape
inference across many test cases—similar to how
:ref:`l-example-plot-backend-test-case` demonstrates backend testing.
"""

from __future__ import annotations

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx_optim.shape_inference import infer_shapes_model
from onnx_light.onnx_py._onnxpy import shape_inference as si

# Make sure the built-in operator schemas are registered before running
# shape inference (the C++ dispatch table looks them up).
onnxl.defs.register_onnx_operator_set_schema()


#####################################
# Build the model
# +++++++++++++++
#
# The graph computes ``Z = Reshape(Concat(Add(X, Y), X, axis=2), [0, 0, -1])``.
# Both ``X`` and ``Y`` are 3-D float inputs with concrete dimensions
# ``[2, 5, 8]``.  The Reshape target shape is stored as an INT64
# initializer ``reshape_shape = [0, 0, -1]``.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["X", "Y"], ["added"]),
            oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
            oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
        ],
        "shape_inference_demo",
        inputs=[
            oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [2, 5, 8]),
            oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [2, 5, 8]),
        ],
        outputs=[oh.make_tensor_value_info("Z", onnxl.TensorProto.FLOAT, None)],
        initializer=[oh.make_tensor("reshape_shape", onnxl.TensorProto.INT64, [3], [0, 0, -1])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=8,
)

# Ordered list of intermediate / output tensors to track.
TRACKED = ["added", "concat_out", "Z"]


# ---------------------------------------------------------------------------
# Helper: extract the last inferred dimension as an integer, or ``None`` when
# the dimension is symbolic (i.e. not a concrete integer).
# ---------------------------------------------------------------------------
def last_dim_int(shape):
    """Return the last element of *shape* if it is a concrete integer."""
    if not shape:
        return None
    last = shape[-1]
    return last if isinstance(last, int) else None


#####################################
# Approach 1 — model-level inference
# ++++++++++++++++++++++++++++++++++++
#
# :func:`infer_shapes_model` mutates ``model`` in place, writing the inferred
# types and shapes back into ``model.graph.output`` and
# ``model.graph.value_info``.  Internally it reads the *values* of every
# initializer, which lets it fully resolve the ``[0, 0, -1]`` target and
# derive ``Z = [2, 5, 16]``.

infer_shapes_model(model)

model_shapes = {}
# Collect from value_info (intermediate tensors).
for vi in model.graph.value_info:
    model_shapes[vi.name] = [
        d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim
    ]
# Collect from the graph output.
for o in model.graph.output:
    model_shapes[o.name] = [
        d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim
    ]

print("Model-level shapes:")
for name in TRACKED:
    print(f"  {name}: {model_shapes.get(name, '(not inferred)')}")


#####################################
# Approach 2 — naïve node-by-node (no value propagation)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# The context is seeded with each initializer's *shape* (an INT64 1-D
# tensor of length 3) but not its *values*.  The Reshape shape function
# therefore cannot read ``[0, 0, -1]`` and falls back to symbolic
# placeholder dimensions (``Reshape_dim0``, ``Reshape_dim1``, ``Reshape_dim2``).


def run_node_by_node(model, propagate_values: bool) -> dict:
    """Walk the graph and return ``{name: shape}`` for every tracked tensor."""
    ctx = si.ShapesContext()
    for opset in model.opset_import:
        ctx.set_opset_version(opset.domain, opset.version)

    # Seed graph inputs.
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.dim_value else d.dim_param for d in tt.shape.dim]
        ctx.set(inp.name, si.OptimTensor(tt.elem_type, dims))

    # Seed initializers, optionally propagating their values.
    for init in model.graph.initializer:
        t = si.OptimTensor(init.data_type, list(init.dims))
        if propagate_values:
            values = [int(v) for v in onh.to_array(init).flat]
            t.set_value_as_shape(values)
        ctx.set(init.name, t)

    results = {}
    for node in model.graph.node:
        si.check_inputs_available(ctx, node)
        si.compute_shape_node(ctx, node)
        for out_name in node.output:
            if not out_name:
                continue
            results[str(out_name)] = list(ctx.get(str(out_name)).shape)

    return results


naive_shapes = run_node_by_node(model, propagate_values=False)
print("\nNaïve node-by-node shapes:")
for name in TRACKED:
    print(f"  {name}: {naive_shapes.get(name, '(not inferred)')}")


#####################################
# Approach 3 — enhanced node-by-node (with value propagation)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Calling :meth:`OptimTensor.set_value_as_shape` on ``reshape_shape``
# mirrors what :func:`infer_shapes_model` does internally. The Reshape
# shape function can now read ``[0, 0, -1]`` and derive
# ``Z = [2, 5, 16]``.

enhanced_shapes = run_node_by_node(model, propagate_values=True)
print("\nEnhanced node-by-node shapes:")
for name in TRACKED:
    print(f"  {name}: {enhanced_shapes.get(name, '(not inferred)')}")


#####################################
# Comparison table
# ++++++++++++++++
#
# Model-level and enhanced node-by-node agree: ``Z = [2, 5, 16]``.
# The naïve approach cannot resolve the ``Reshape`` output because the
# shape values ``[0, 0, -1]`` are not available in the context.

print("\nComparison (last dimension of each tracked tensor):")
print(f"  {'tensor':<15} {'model-level':>15} {'naive':>15} {'enhanced':>15}")
for name in TRACKED:
    m = last_dim_int(model_shapes.get(name) or [])
    n = last_dim_int(naive_shapes.get(name) or [])
    e = last_dim_int(enhanced_shapes.get(name) or [])
    print(f"  {name:<15} {str(m):>15} {str(n):>15} {str(e):>15}")


#####################################
# Plot
# ++++
#
# The grouped bar chart below shows the inferred last dimension for every
# tracked tensor under each approach.  ``None`` (grey bar) marks a
# dimension that could not be resolved to a concrete integer.
#
# The gap for ``Z`` in the naïve column (grey) versus the concrete ``16``
# in the model-level and enhanced columns makes the divergence immediately
# visible.

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

approaches = ["model-level", "naive", "enhanced"]
colors = ["steelblue", "tomato", "darkorange"]

# Collect data: last-dim per tensor per approach (use 0 as placeholder for None).
data = {
    "model-level": [last_dim_int(model_shapes.get(n) or []) for n in TRACKED],
    "naive": [last_dim_int(naive_shapes.get(n) or []) for n in TRACKED],
    "enhanced": [last_dim_int(enhanced_shapes.get(n) or []) for n in TRACKED],
}

x = np.arange(len(TRACKED))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 4))

for i, (approach, color) in enumerate(zip(approaches, colors)):
    vals = data[approach]
    bar_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + i * width, bar_vals, width, label=approach, color=color)
    # Mark unresolved bars with a "?" annotation.
    for bar, v in zip(bars, vals):
        if v is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                "?",
                ha="center",
                va="bottom",
                fontsize=12,
                color="grey",
            )

ax.set_xticks(x + width)
ax.set_xticklabels(TRACKED)
ax.set_ylabel("Last inferred dimension")
ax.set_title(
    "Shape inference: last dimension per tensor\n"
    "(grey '?' = dimension not resolved to a concrete integer)"
)
ax.legend(handles=[mpatches.Patch(color=c, label=a) for a, c in zip(approaches, colors)])
ax.grid(axis="y", linestyle="--", alpha=0.6)
fig.tight_layout()
fig.savefig("plot_shape_inference.png")


#####################################
# Systematic testing with make_test_class
# ++++++++++++++++++++++++++++++++++++++++
#
# For comprehensive validation, you can use
# :func:`~onnx_light.backend.test.case.make_test_class` to create a test
# class that runs shape inference on every backend test case. This mirrors
# the pattern shown in :ref:`l-example-plot-backend-test-case`.
#
# The function :func:`make_test_class` accepts a callable that receives a
# model (and optionally input data sets). Here we define a simple validator
# that runs :func:`infer_shapes_model` and checks that the inferred shapes
# for intermediate tensors match the expected ``value_info`` stored in each
# test case.

# Import is used in the commented example code below to demonstrate the pattern.
from onnx_light.backend.test.case import make_test_class  # noqa: E402, F401


def validate_shape_inference(model_with_expected_shapes: onnxl.ModelProto):
    """
    Validates that infer_shapes_model reproduces the expected value_info.

    Test cases in the backend test suite often include pre-computed
    ``value_info`` entries that represent the "ground truth" for shape
    inference. This function:

    1. Copies the model.
    2. Clears its ``value_info`` to simulate a model with no intermediate shapes.
    3. Calls :func:`infer_shapes_model` to re-infer shapes.
    4. Compares the inferred shapes against the original ``value_info``.

    If the inferred shapes do not match the expected shapes, an assertion
    will fail, helping you catch regressions in the shape inference logic.
    """
    # Keep a reference to the expected value_info.
    expected_value_info = {vi.name: vi for vi in model_with_expected_shapes.graph.value_info}

    # Make a working copy and clear its value_info.
    work = onnxl.ModelProto()
    work.CopyFrom(model_with_expected_shapes)
    work.graph.value_info.clear()

    # Run shape inference.
    infer_shapes_model(work)

    # Compare inferred shapes to the expected shapes.
    inferred_value_info = {vi.name: vi for vi in work.graph.value_info}
    for name, expected in expected_value_info.items():
        if name not in inferred_value_info:
            raise AssertionError(f"Shape inference did not produce value_info for {name!r}")
        inferred = inferred_value_info[name]
        if not expected.type.tensor_type or not inferred.type.tensor_type:
            continue  # Skip non-tensor types.
        exp_shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in expected.type.tensor_type.shape.dim
        )
        inf_shape = tuple(
            d.dim_param if d.dim_param else d.dim_value
            for d in inferred.type.tensor_type.shape.dim
        )
        # For this simple check, we only verify that ranks match.
        # A production test might enforce exact symbolic/numeric dimension matches.
        if len(exp_shape) != len(inf_shape):
            raise AssertionError(
                f"Shape mismatch for {name!r}: expected rank {len(exp_shape)}, "
                f"got rank {len(inf_shape)}"
            )


# Uncomment the lines below to create a test class and run it with pytest or unittest.
# This example demonstrates the pattern; in practice, you would run this in a
# separate test file.
#
# TestShapeInferenceBackend = make_test_class(
#     validate_shape_inference,
#     include_regex=["test_abs", "test_add"],  # Filter to a small subset for demo.
# )
#
# if __name__ == "__main__":
#     import unittest
#     unittest.main(verbosity=2)
#
# print(
#     "\nTo systematically test shape inference across all backend cases, "
#     "use make_test_class as shown above."
# )
