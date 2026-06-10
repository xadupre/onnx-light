"""
.. _l-example-plot-runtime-events:

Run a model with the C++ runtime and inspect intermediate results
==================================================================

:mod:`onnx_light` ships a C++ kernel dispatcher exposed in Python
through :mod:`onnx_light.onnx_py._onnxpykernels` (re-exported as the
``runtime`` submodule). Beyond computing the final outputs the
runtime maintains a :class:`RuntimeContext` whose ``events()``
method returns an append-only log of every tensor map mutation:
graph initializers seeded by :func:`run_graph`, inputs injected by
the caller, intermediate values produced by each node kernel and
outputs propagated back to the caller.

This example:

* builds a small graph with an initializer and two operators
  (``Mul`` then ``Add``),
* runs it through :func:`run_model` while collecting the event log,
* prints the events, illustrating how to peek at intermediate
  results without re-instrumenting the graph,
* renders a compact table of the recorded events for visual
  inspection.
"""

from __future__ import annotations

import numpy as np

from onnx_light.onnx_lib import numpy_helper, parser
from onnx_light.onnx_py._onnxpykernels import runtime as _runtime

#####################################
# Build a small ONNX model
# ++++++++++++++++++++++++
#
# The graph multiplies its input by an initializer ``two`` and adds
# the original input back. The two intermediate values we expect to
# see in the event log are ``z = x * two`` and ``y = z + x``.

model = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 18]>'
    "agraph (float[3] x) => (float[3] y) <float two = {2.0}>"
    "{"
    "  z = Mul(x, two)"
    "  y = Add(z, x)"
    "}"
)
print(model)

#####################################
# Prepare the runtime context
# +++++++++++++++++++++++++++
#
# ``RuntimeContext`` owns the name-keyed tensor map shared across
# every node. The construction-time :class:`KernelContext` carries
# the opset version used to instantiate each per-operator kernel.
# Inputs are inserted with :meth:`RuntimeContext.set` after being
# converted from :class:`numpy.ndarray` to a runtime ``Tensor`` via
# :func:`tensor_from_proto`.

ctx = _runtime.RuntimeContext(_runtime.KernelContext(_runtime.default_opset(18)))

x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
ctx.set("x", _runtime.tensor_from_proto(numpy_helper.from_array(x, name="x")))

#####################################
# Run the model
# +++++++++++++
#
# :func:`run_model` registers every model-local ``FunctionProto`` in
# the runtime's function registry and then delegates to
# :func:`run_graph`, which seeds the context with every
# ``TensorProto`` in ``graph.initializer`` before executing the node
# sequence.

_runtime.run_model(model, ctx)

(y,) = [
    np.frombuffer(ctx.get(name).raw_data(), dtype=np.float32).reshape(
        tuple(int(d) for d in ctx.get(name).shape)
    )
    for name in ["y"]
]
print(f"y = {y}")

#####################################
# Inspect the event log
# +++++++++++++++++++++
#
# :meth:`RuntimeContext.events` returns a list of
# :class:`TensorEvent` entries. Each event carries the
# ``action`` (``"add"`` / ``"replace"`` / ``"remove"``), the
# ``kind`` of value (``"input"``, ``"initializer"``,
# ``"intermediate"`` or ``"output"``), the tensor ``name``,
# ``data_type``, ``shape``, the number of element values captured
# (``value_count``) and the first few element values themselves.
#
# Because the event buffer is fixed-size (capped at 8 entries),
# tensors with more than 8 elements are summarised: ``data_type``
# is set to ``-1`` and ``shape`` is left empty to signal the
# truncated payload.

events = ctx.events()
print(f"Recorded {len(events)} event(s):")
for ev in events:
    d = ev.as_dict()
    print(
        f"  [{d['action']:<7s} {d['kind']:<12s}] {d['name']:<6s} "
        f"dtype={d['data_type']:<3d} shape={d['shape']} "
        f"values={d.get('values') or d.get('string_values')}"
    )

#####################################
# Filter intermediate results
# +++++++++++++++++++++++++++
#
# Filtering the event log by ``kind`` is the easiest way to recover
# only the values produced by node kernels — i.e. the intermediate
# tensors of the graph.

print("Intermediate tensors produced by node kernels:")
for ev in events:
    d = ev.as_dict()
    if d["kind"] == "intermediate":
        print(f"  {d['name']:<6s} = {d.get('values')}")

#####################################
# Render the event log as a table
# +++++++++++++++++++++++++++++++
#
# A simple matplotlib figure is used both as the sphinx-gallery
# thumbnail and as a compact visual recap of the captured events.

import matplotlib.pyplot as plt  # noqa: E402

rows = []
for ev in events:
    d = ev.as_dict()
    values = d.get("values") or d.get("string_values") or []
    rows.append([d["action"], d["kind"], d["name"], str(d["shape"]), str(values)])

fig, ax = plt.subplots(figsize=(7, 1.6 + 0.3 * len(rows)))
ax.set_axis_off()
table = ax.table(
    cellText=rows,
    colLabels=["action", "kind", "name", "shape", "values"],
    loc="center",
    cellLoc="left",
    colLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.3)
ax.set_title("RuntimeContext event log")
fig.tight_layout()
