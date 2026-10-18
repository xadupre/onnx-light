"""
.. _l-example-gradient-linear-regression:

Gradient and training loop for linear regression
=================================================

This example shows how to use :mod:`onnx_light.onnx_core.gradient` to:

1. Build linear-regression forward and MSE loss graphs as ONNX nodes.
2. Compute its gradient with respect to the trainable parameters (weights
   and bias) using :func:`~onnx_light.onnx_core.gradient.gradient_of_nodes`.
3. Execute forward, loss, backward and gradient-descent updates with the native
   C++ runtime. Python only orchestrates the sessions; neither PyTorch nor
   ONNX Runtime is required.

The model is ``y = X @ W + b`` where ``X`` is the feature matrix (shape
``[N, F]``), ``W`` is the weight vector (shape ``[F, 1]``), and ``b`` is a
scalar bias.  The loss is the mean-squared error (MSE)
``L = mean((y_pred - y_true) ** 2)``.

Analytical gradients
--------------------

For the MSE loss ``L`` and prediction ``y_pred = X @ W + b``:

* ``dL/dW = (2/N) * X.T @ (y_pred - y_true)``
* ``dL/db = (2/N) * sum(y_pred - y_true)``

The gradient FunctionProto produced by
:func:`~onnx_light.onnx_core.gradient.gradient_of_nodes` propagates the supplied
unit scalar loss gradient back to the parameters. Its ``grad_b`` output contains one
contribution per prediction; summing these contributions accounts for the
broadcast of the scalar bias. This reduction and both parameter updates are
also ONNX nodes executed by native kernels.
"""

from __future__ import annotations

import numpy as np

######################################
# Build the forward graph
# -----------------------
#
# The forward pass ``y = X @ W + b`` is expressed as two ONNX nodes: a
# ``MatMul`` followed by an ``Add``. ``Sub``, ``Mul`` and ``ReduceMean`` compute
# the scalar MSE.

import onnx_light.onnx.helper as oh  # noqa: E402
import onnx_light.onnx.numpy_helper as onh  # noqa: E402
from onnx_light.onnx import TensorProto  # noqa: E402
from onnx_light.onnx_py._onnxpykernels.runtime import (  # noqa: E402
    RuntimeContext,
    RuntimeSession,
    tensor_from_proto,
)

forward_nodes = [
    oh.make_node("MatMul", ["X", "W"], ["mm"]),
    oh.make_node("Add", ["mm", "b"], ["y_pred"]),
    oh.make_node("Sub", ["y_pred", "y_true"], ["residual"]),
    oh.make_node("Mul", ["residual", "residual"], ["squared"]),
    oh.make_node("ReduceMean", ["squared"], ["loss"], keepdims=0),
]

print("Forward nodes:")
for n in forward_nodes:
    print(f"  {n.op_type}({list(n.input)}) -> {list(n.output)}")

######################################
# Compute the gradient FunctionProto
# -----------------------------------
#
# :func:`~onnx_light.onnx_core.gradient.gradient_of_nodes` performs reverse-mode
# automatic differentiation over the forward nodes and returns a
# :class:`~onnx_light.onnx_proto._helper.FunctionProto` that encodes the
# backward computation.
#
# * ``xs`` — parameters to differentiate with respect to (``W`` and ``b``).
# * ``y`` — the scalar (or tensor) output whose gradient is propagated back.
# * ``zs`` — additional backward inputs, including saved forward intermediates.
#
# The generated backward nodes reference ``residual`` and ``squared``.
# Exposing them as forward graph outputs keeps them alive even with
# ``release_intermediates=True``. Both sessions share a context, so backward
# consumes these saved outputs from the current iteration. Keeping tensors in
# one context also keeps their owning memory arenas alive throughout training.

from onnx_light.onnx_core.gradient import gradient_of_nodes  # noqa: E402

grad_fn = gradient_of_nodes(
    nodes=forward_nodes,
    inputs=["X", "W", "b", "y_true"],
    initializers=[],
    xs=["W", "b"],
    y="loss",
    zs=["X", "y_true", "residual", "squared"],
)

print("\nGradient FunctionProto:")
print(f"  inputs  = {list(grad_fn.input)}")
print(f"  outputs = {list(grad_fn.output)}")
print(f"  nodes   = {[str(n.op_type) for n in grad_fn.node]}")

######################################
# Training loop with the generated gradient
# ------------------------------------------
#
# The first session computes predictions, loss and saved intermediates. The
# second executes the generated backward nodes, reduces the scalar-bias
# gradient, then applies ordinary gradient descent. Parameters stay native
# tensors between iterations; NumPy is only used to read losses and plot.

shapes = {
    "X": ["N", 4],
    "W": [4, 1],
    "b": [],
    "y_true": ["N", 1],
    "y_pred": ["N", 1],
    "residual": ["N", 1],
    "squared": ["N", 1],
    "loss": [],
    "dy": [],
    "learning_rate": [],
    "grad_W": [4, 1],
    "grad_b": ["N", 1],
    "grad_b_scalar": [],
    "W_next": [4, 1],
    "b_next": [],
}


def make_session(nodes, name, inputs, outputs):
    """Builds a native session with explicit input and output shapes."""
    graph = oh.make_graph(
        nodes,
        name,
        [oh.make_tensor_value_info(n, TensorProto.FLOAT, shapes[n]) for n in inputs],
        [oh.make_tensor_value_info(n, TensorProto.FLOAT, shapes[n]) for n in outputs],
    )
    return RuntimeSession(
        oh.make_model(graph, opset_imports=[oh.make_opsetid("", 21)]), check_shapes=True
    )


forward_session = make_session(
    forward_nodes,
    "forward_loss",
    ["X", "W", "b", "y_true"],
    ["y_pred", "loss", "residual", "squared"],
)
backward_session = make_session(
    [
        *grad_fn.node,
        oh.make_node("ReduceSum", ["grad_b"], ["grad_b_scalar"], keepdims=0),
        oh.make_node("Mul", ["learning_rate", "grad_W"], ["step_W"]),
        oh.make_node("Sub", ["W", "step_W"], ["W_next"]),
        oh.make_node("Mul", ["learning_rate", "grad_b_scalar"], ["step_b"]),
        oh.make_node("Sub", ["b", "step_b"], ["b_next"]),
    ],
    "backward_update",
    [*grad_fn.input, "learning_rate"],
    ["grad_W", "grad_b", "grad_b_scalar", "W_next", "b_next"],
)

# Reproducible random seed
rng = np.random.default_rng(42)

# --- Synthetic dataset ---
N, F = 200, 4  # 200 samples, 4 features
W_true = rng.standard_normal((F, 1)).astype(np.float32)
b_true = np.float32(1.5)
X_data = rng.standard_normal((N, F)).astype(np.float32)
y_data = X_data @ W_true + b_true + 0.05 * rng.standard_normal((N, 1)).astype(np.float32)

# --- Initialize parameters ---
W = rng.standard_normal((F, 1)).astype(np.float32) * 0.1
b = np.float32(0.0)

learning_rate = 0.05
n_epochs = 60
loss_history: list[float] = []

context = RuntimeContext(release_intermediates=True)
for name, value in {"X": X_data, "y_true": y_data, "W": W, "b": np.asarray(b)}.items():
    context.set(name, tensor_from_proto(onh.from_array(value, name=name)))
for name, value in {"dy": 1.0, "learning_rate": learning_rate}.items():
    context.set(
        name, tensor_from_proto(onh.from_array(np.array(value, dtype=np.float32), name=name))
    )

for _epoch in range(n_epochs):
    forward_session.run(context)
    loss_history.append(float(np.from_dlpack(context.get("loss"))))
    backward_session.run(context)
    context.put("W", context.get("W_next"))
    context.put("b", context.get("b_next"))

# Copies plotting snapshots before another run can reuse native storage.
W = np.from_dlpack(context.get("W")).copy()
b = np.from_dlpack(context.get("b")).copy()

print(f"\nTraining summary (lr={learning_rate}, epochs={n_epochs}):")
print(f"  Initial loss : {loss_history[0]:.4f}")
print(f"  Final   loss : {loss_history[-1]:.6f}")
print(f"  Learned W    : {W.ravel()}")
print(f"  True    W    : {W_true.ravel()}")
print(f"  Learned b    : {b:.4f}  (true: {b_true:.4f})")

######################################
# Plot — loss curve
# -----------------

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, n_epochs + 1), loss_history, linewidth=2, color="steelblue")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Linear regression — gradient-descent training")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("plot_gradient_linear_regression.png")

######################################
# Plot — predicted vs true values
# --------------------------------

forward_session.run(context)
y_final = np.from_dlpack(context.get("y_pred")).copy()
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.scatter(y_data, y_final, s=10, alpha=0.5, color="steelblue", label="predictions")
lims = [float(y_data.min()), float(y_data.max())]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="perfect fit")
ax2.set_xlabel("True y")
ax2.set_ylabel("Predicted y")
ax2.set_title("Predicted vs true values after training")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig("plot_gradient_linear_regression_scatter.png")
