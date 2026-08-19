"""
.. _l-example-plot-run-tensor-sequence-dict:

Run the reference evaluator with tensor, sequence and dictionary inputs/outputs
================================================================================

:class:`~onnx_light.onnx.reference.ReferenceEvaluator` supports three kinds
of ONNX values at the graph boundary:

* **Tensor** – the standard case.  Feed a :class:`numpy.ndarray`; retrieve a
  :class:`numpy.ndarray`.
* **Sequence** – an ordered collection of tensors (``seq(T)``).  Feed a
  Python ``list`` (or ``tuple``) of :class:`numpy.ndarray` objects, one per
  element; retrieve a ``list`` of :class:`numpy.ndarray` objects.
* **Dictionary** – a key-value map (``map(K, V)``).  Feed a Python ``dict``
  under the graph-input name.

This example builds small ONNX models and demonstrates both *how to supply
the inputs* and *how to read the outputs*.
"""

from __future__ import annotations

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
from onnx_light.onnx.reference import ReferenceEvaluator
import onnx_light.onnx.numpy_helper as onh
from onnx_light.tools import pretty_onnx

# ---------------------------------------------------------------------------
# 1. Tensor input and tensor output
# ++++++++++++++++++++++++++++++++++
#
# The simplest case: the graph takes a single float tensor ``x`` and returns
# ``y = Abs(x)``.  Tensor inputs are fed as plain :class:`numpy.ndarray`
# values; tensor outputs are returned as :class:`numpy.ndarray` values at
# the corresponding index of the result list.

tensor_graph = oh.make_graph(
    [oh.make_node("Abs", ["x"], ["y"])],
    "abs_graph",
    [oh.make_tensor_value_info("x", onnxl.TensorProto.FLOAT, [4])],
    [oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [4])],
)
tensor_model = oh.make_model(tensor_graph, opset_imports=[oh.make_opsetid("", 18)])
print("=== Tensor model ===")
print(pretty_onnx(tensor_model))

tensor_sess = ReferenceEvaluator(tensor_model)
print("input_names :", tensor_sess.input_names)
print("output_names:", tensor_sess.output_names)

x = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
results = tensor_sess.run(None, {"x": x})

# ``results`` is a list with one entry per declared output.
# A tensor output is a :class:`numpy.ndarray`.
tensor_output = results[0]
print("\nInput x          :", x)
print("Output y = Abs(x):", tensor_output)
assert isinstance(tensor_output, np.ndarray)

# ---------------------------------------------------------------------------
# 2. Sequence input and sequence output
# ++++++++++++++++++++++++++++++++++++++
#
# A ``seq(T)``-typed graph input is fed as a Python ``list`` (or ``tuple``)
# of :class:`numpy.ndarray` objects, one per sequence element.  The output
# is returned as a ``list`` of :class:`numpy.ndarray` objects.
#
# Here we use ``SequenceMap`` with an ``Identity`` body to pass each element
# unchanged through the graph, so the output sequence mirrors the input.

body = oh.make_graph(
    [oh.make_node("Identity", ["elem"], ["out"])],
    "identity_body",
    [oh.make_tensor_value_info("elem", onnxl.TensorProto.FLOAT, None)],
    [oh.make_tensor_value_info("out", onnxl.TensorProto.FLOAT, None)],
)
seq_node = oh.make_node("SequenceMap", ["seq_in"], ["seq_out"], body=body)
seq_graph = oh.make_graph(
    [seq_node],
    "seq_graph",
    [oh.make_tensor_sequence_value_info("seq_in", onnxl.TensorProto.FLOAT, None)],
    [oh.make_tensor_sequence_value_info("seq_out", onnxl.TensorProto.FLOAT, None)],
)
seq_model = oh.make_model(seq_graph, opset_imports=[oh.make_opsetid("", 18)])
print("\n=== Sequence model ===")
print(pretty_onnx(seq_model))

seq_sess = ReferenceEvaluator(seq_model)
print("input_names :", seq_sess.input_names)
print("output_names:", seq_sess.output_names)

# Feed a sequence as a list of numpy arrays (one per element).
seq_input = [
    np.array([1.0, 2.0], dtype=np.float32),
    np.array([3.0, 4.0, 5.0], dtype=np.float32),
    np.array([6.0], dtype=np.float32),
]
seq_results = seq_sess.run(None, {"seq_in": seq_input})

# A sequence output is a Python ``list`` of :class:`numpy.ndarray` objects.
seq_output = seq_results[0]
assert isinstance(seq_output, list)
print("\nSequence input  (", len(seq_input), "elements):")
for i, arr in enumerate(seq_input):
    print(f"  element[{i}]:", arr)
print("Sequence output (", len(seq_output), "elements):")
for i, arr in enumerate(seq_output):
    print(f"  element[{i}]:", arr)

# ---------------------------------------------------------------------------
# 3. Dictionary (map) input and tensor output
# ++++++++++++++++++++++++++++++++++++++++++++
#
# A ``map(K, V)``-typed graph input is fed as a Python ``dict`` under the
# graph-input name.  Internally the runtime represents the map as two
# sequences (keys and values; ``unordered_map`` in C++), but the Python API
# accepts a plain ``dict`` and performs the split automatically.
#
# Here we use ``ai.onnx.ml::DictVectorizer`` to convert a
# ``map(int64, float)`` input into a dense 1-D output tensor.  The
# ``int64_vocabulary`` attribute defines the vocabulary order, so
# ``{10: 1.5, 30: 2.5}`` is mapped to ``[1.5, 0.0, 2.5]``.

dict_graph = oh.make_graph(
    nodes=[
        oh.make_node(
            "DictVectorizer", ["d"], ["y"], domain="ai.onnx.ml", int64_vocabulary=[10, 20, 30]
        )
    ],
    name="dict_graph",
    inputs=[
        oh.make_value_info(
            "d",
            oh.make_map_type_proto(
                onnxl.TensorProto.INT64, oh.make_tensor_type_proto(onnxl.TensorProto.FLOAT, None)
            ),
        )
    ],
    outputs=[oh.make_tensor_value_info("y", onnxl.TensorProto.FLOAT, [3])],
)
dict_model = oh.make_model(
    dict_graph, opset_imports=[oh.make_opsetid("", 13), oh.make_opsetid("ai.onnx.ml", 1)]
)
print("\n=== Dictionary model ===")
print(pretty_onnx(dict_model))

dict_sess = ReferenceEvaluator(dict_model)
print("input_names :", dict_sess.input_names)  # ['d']
print("output_names:", dict_sess.output_names)

# Feed the map as a Python dict under the original input name.
dict_results = dict_sess.run(None, {"d": {10: 1.5, 30: 2.5}})
print("\nInput  d = { 10: 1.5, 30: 2.5 }")
print("Output y :", dict_results[0])
np.testing.assert_array_equal(dict_results[0], np.array([1.5, 0.0, 2.5], dtype=np.float32))

#####################################
# 4. Map of tensors – numpy_helper round-trip
# ++++++++++++++++++++++++++++++++++++++++++++
#
# An ONNX map whose *values* are tensors – e.g.
# ``map(int64, tensor(float))`` – is represented in Python as a plain
# ``dict`` mapping integer (or string) keys to :class:`numpy.ndarray`
# objects.
#
# Use :func:`~onnx_light.onnx.numpy_helper.from_dict` to serialize such a
# dict to a :class:`~onnx_light.onnx.MapProto` and
# :func:`~onnx_light.onnx.numpy_helper.to_dict` to recover it.  The key
# type and the tensor element type are inferred automatically from the
# Python objects.

tensor_map = {
    np.int64(0): np.array([1.0, 2.0, 3.0], dtype=np.float32),
    np.int64(1): np.array([4.0, 5.0, 6.0], dtype=np.float32),
    np.int64(2): np.array([7.0, 8.0, 9.0], dtype=np.float32),
}

map_proto = onh.from_dict(tensor_map, name="feature_map")
print("\n=== Map of tensors – MapProto ===")
print(
    "key_type         :", int(map_proto.key_type), "(INT64 =", int(onnxl.TensorProto.INT64), ")"
)
print("values.elem_type :", int(map_proto.values.elem_type))
print("number of entries:", len(map_proto.keys))

# Recover the Python dict from the MapProto.
recovered = onh.to_dict(map_proto)
print("\nRecovered dict:")
for k, v in recovered.items():
    print(f"  key={k}  value={v}")

# Verify round-trip fidelity.
for k in tensor_map:
    np.testing.assert_array_equal(recovered[k], tensor_map[k])

#####################################
# Gallery thumbnail
# +++++++++++++++++
#
# Render a simple text figure used as the sphinx-gallery thumbnail for this
# example.

import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(4, 3))
ax.text(0.5, 0.5, "tensor\nsequence\ndict", ha="center", va="center", fontsize=22)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("plot_run_with_tensor_sequence_dict.png")
