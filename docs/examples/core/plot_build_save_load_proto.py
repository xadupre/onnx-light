"""
.. _l-example-plot-build-save-load-proto:

Build, save and load a tiny ONNX model with onnx_light
=======================================================

This example is the Python counterpart of the standalone C++ example
``examples/build_save_load_onnx_proto`` (see
:ref:`l-cpp-build-save-load-onnx-proto-example`).

It shows the minimal API surface required to *produce* and *consume*
an ``.onnx`` file using only the proto layer of *onnx-light*:

* build a tiny :class:`onnx_light.onnx.ModelProto` from scratch
  (single ``Add`` node, ``FLOAT[3]`` input and ``FLOAT[3]``
  initializer ``B = [1, 2, 3]``),
* serialise it to disk with :func:`onnx_light.onnx.save`,
* parse it back with :func:`onnx_light.onnx.load`,
* check the round-trip on the initializer values.

No operator schemas, shape inference, checker or backend kernels are
exercised: this is the Python mirror of linking only against
``onnx_light::lib_onnx_proto`` from C++.
"""

import os

import numpy as np
import onnx_light.onnx as onnxl
from onnx_light.onnx import helper as oh
from onnx_light.onnx import numpy_helper as onh


# %%
# Build the model from scratch
# ----------------------------
#
# The graph computes ``Y = X + B`` with a single ``Add`` node, one input
# ``X`` of shape ``[3]`` and one initializer ``B`` of shape ``[3]``.

b_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

inputs = [oh.make_tensor_value_info("X", onnxl.TensorProto.FLOAT, [3])]
outputs = [oh.make_tensor_value_info("Y", onnxl.TensorProto.FLOAT, [3])]
initializers = [onh.from_array(b_values, name="B")]
nodes = [oh.make_node("Add", ["X", "B"], ["Y"], name="add_node")]

graph = oh.make_graph(nodes, "add_graph", inputs, outputs, initializer=initializers)
onnx_model = oh.make_model(
    graph,
    opset_imports=[oh.make_opsetid("", 14)],
    ir_version=7,
    producer_name="plot_build_save_load_proto",
    producer_version="1.0.0",
    doc_string="Single-node Add model built from scratch.",
)
onnx_model.domain = "ai.onnx-light.example"
onnx_model.model_version = 1

print(f"Built model with {len(onnx_model.graph.node)} node(s) "
      f"and {len(onnx_model.graph.initializer)} initializer(s).")

# %%
# Save the model with onnx_light
# -------------------------------
#
# :func:`onnx_light.onnx.save` serialises the model through the C++
# ``lib_onnx_proto`` layer, no operator metadata is required.

out_dir = "temp_plot_build_save_load_proto"
os.makedirs(out_dir, exist_ok=True)
model_path = os.path.join(out_dir, "add_model.onnx")

onnxl.save(onnx_model, model_path)
print(f"Saved: {model_path} ({os.path.getsize(model_path)} bytes)")

# %%
# Load the model back
# --------------------
#
# :func:`onnx_light.onnx.load` memory-maps the ``.onnx`` file and parses
# it through the same proto layer.

loaded = onnxl.load(model_path)
print(f"Loaded ir_version={loaded.ir_version}, producer={loaded.producer_name!r}")
print(f"Graph name: {loaded.graph.name}, nodes={len(loaded.graph.node)}, "
      f"initializers={len(loaded.graph.initializer)}")

# %%
# Verify the round-trip
# ----------------------
#
# Decode the raw bytes of the initializer ``B`` and compare them to the
# original numpy values.

assert len(loaded.graph.initializer) == 1
b_init = loaded.graph.initializer[0]
assert b_init.name == "B"
restored = np.frombuffer(bytes(b_init.raw_data), dtype=np.float32)
print(f"Initializer B (restored): {restored.tolist()}")
np.testing.assert_array_equal(restored, b_values)
print("Round-trip OK")

# %%
# See also
# --------
#
# * :ref:`l-cpp-build-save-load-onnx-proto-example` – the C++ version of
#   this example, linking only against ``onnx_light::lib_onnx_proto``.
# * :ref:`l-example-plot-load-save-external` – save and load with
#   external weight files.
