"""
.. _l-example-plot-runtime-events:

Run a model with the runtime and inspect intermediate results
=============================================================

:mod:`onnx_light` ships a C++ kernel dispatcher exposed in Python
through :mod:`onnx_light.kernels`. Its ``runtime`` submodule owns a
:class:`RuntimeContext` whose :meth:`events` method returns an
append-only log of every tensor map mutation: graph initializers
seeded by :func:`run_graph`, inputs injected by the caller,
intermediate values produced by each node kernel and outputs
propagated back to the caller.

This example:

* builds a small graph with an initializer and two operators
  (``Mul`` then ``Add``),
* runs :func:`~onnx_light.onnx_optim.shape_inference.infer_shapes_model`
  on it so the expected shape of each intermediate tensor is
  recorded in ``graph.value_info``,
* drives the runtime through :func:`run_model` while collecting
  the event log,
* prints the events, illustrating how to peek at intermediate
  results without re-instrumenting the graph,
* cross-checks the runtime-observed shape of every intermediate
  tensor against the statically inferred shape,
* renders a compact table of the recorded events for visual
  inspection.
"""

from __future__ import annotations

import numpy as np

from onnx_light.kernels import runtime
from onnx_light.onnx_lib import numpy_helper, parser
from onnx_light.onnx_optim.shape_inference import infer_shapes_model

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
# Infer the shapes of every intermediate tensor
# +++++++++++++++++++++++++++++++++++++++++++++
#
# :func:`infer_shapes_model` walks the graph in topological order,
# applies the shape-inference rule registered for each operator and
# writes the inferred element type and shape of every intermediate
# tensor to ``graph.value_info``. We collect those inferred shapes
# in a dictionary so they can be compared against the shapes
# observed at runtime.

infer_shapes_model(model)


def _shape_of(type_proto):
    return tuple(
        d.dim_param if d.dim_param else int(d.dim_value) for d in type_proto.tensor_type.shape.dim
    )


inferred_shapes = {vi.name: _shape_of(vi.type) for vi in model.graph.value_info}
for inp in model.graph.input:
    inferred_shapes[inp.name] = _shape_of(inp.type)
for out in model.graph.output:
    inferred_shapes[out.name] = _shape_of(out.type)

print("Statically inferred shapes:")
for name, shape in inferred_shapes.items():
    print(f"  {name:<6s} -> {shape}")

#####################################
# Prepare the runtime context
# +++++++++++++++++++++++++++
#
# :class:`RuntimeContext` owns the name-keyed tensor map shared
# across every node. The construction-time :class:`KernelContext`
# carries the opset version used to instantiate each per-operator
# kernel. Inputs are inserted with :meth:`RuntimeContext.set` after
# being converted from :class:`numpy.ndarray` to a runtime
# ``Tensor`` via :func:`tensor_from_proto`.

ctx = runtime.RuntimeContext(runtime.KernelContext(runtime.default_opset(18)))

x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
ctx.set("x", runtime.tensor_from_proto(numpy_helper.from_array(x, name="x")))

#####################################
# Run the model
# +++++++++++++
#
# :func:`run_model` registers every model-local ``FunctionProto`` in
# the runtime's function registry and then delegates to
# :func:`run_graph`, which seeds the context with every
# ``TensorProto`` in ``graph.initializer`` before executing the node
# sequence.

runtime.run_model(model, ctx)

y_tensor = ctx.get("y")
y = np.frombuffer(y_tensor.raw_data(), dtype=np.float32).reshape(
    tuple(int(d) for d in y_tensor.shape)
)
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
# Each event stores at most a fixed number of element values
# (currently 8) from the recorded tensor. Tensors with more
# elements are summarised: ``data_type`` is set to ``-1`` and
# ``shape`` is left empty to signal the truncated payload. The
# total number of events in the log itself is unbounded.

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
# Cross-check intermediate shapes
# +++++++++++++++++++++++++++++++
#
# Filtering the event log by ``kind`` is the easiest way to recover
# only the values produced by node kernels — i.e. the intermediate
# tensors of the graph. For each one we compare the runtime-observed
# shape against the shape :func:`infer_shapes_model` had pre-computed.

print("Intermediate tensors produced by node kernels:")
for ev in events:
    d = ev.as_dict()
    if d["kind"] != "intermediate":
        continue
    runtime_shape = tuple(d["shape"])
    inferred = inferred_shapes.get(d["name"])
    match = "OK" if inferred == runtime_shape else "MISMATCH"
    print(
        f"  {d['name']:<6s} runtime={runtime_shape} inferred={inferred} [{match}]"
        f"  values={d.get('values')}"
    )

#####################################
# Render the event log as a table
# +++++++++++++++++++++++++++++++
#
# A simple matplotlib figure is used both as the sphinx-gallery
# thumbnail and as a compact visual recap of the captured events.
# The last column shows the statically inferred shape so it can be
# read alongside the runtime-observed shape.

import matplotlib.pyplot as plt  # noqa: E402

rows = []
for ev in events:
    d = ev.as_dict()
    values = d.get("values") or d.get("string_values") or []
    rows.append(
        [
            d["action"],
            d["kind"],
            d["name"],
            str(tuple(d["shape"])),
            str(inferred_shapes.get(d["name"], "")),
            str(values),
        ]
    )

fig, ax = plt.subplots(figsize=(8, 1.6 + 0.3 * len(rows)))
ax.set_axis_off()
table = ax.table(
    cellText=rows,
    colLabels=["action", "kind", "name", "runtime shape", "inferred shape", "values"],
    loc="center",
    cellLoc="left",
    colLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.3)
ax.set_title("RuntimeContext event log")
fig.tight_layout()
