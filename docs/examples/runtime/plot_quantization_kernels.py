"""
.. _l-example-quantization-kernels:

Calibrates quantization with graph kernels
=========================================

This example runs ``ai.rt::Quantize`` and ``ai.rt::Dequantize`` in a native
runtime session. It compares automatic INT4 block calibration with explicit
scales, inspects the encoded graph output, and reuses it as an initializer.
These operators are onnx-light extensions, not ONNX ``QuantizeLinear`` and
``DequantizeLinear``. See :ref:`l-quantized-values` for supported formats and
the additional parameters required by learned or transformed layouts.
"""

import matplotlib.pyplot
import numpy

from onnx_light import onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx_core.quantization import (
    QuantizationFormat,
    make_quantization_plan,
    make_quantization_type,
)
from onnx_light.onnx_py._onnxpykernels.runtime import (
    RuntimeContext,
    RuntimeSession,
    tensor_from_proto,
)

# %%
# Build a graph with an encoded edge
# ---------------------------------
#
# Each block covers four consecutive elements. The type describes the
# storage layout, not the numerical scales: Quantize computes those from
# each input block. The encoded value keeps the logical FLOAT input dtype;
# Dequantize requests a DOUBLE output independently.
#
# Declaring ``Q`` as a graph output retains it for inspection after the run.
# Encoded outputs use ``get_value`` rather than the tensor-only ``get``.
# Importing the native runtime above registers its built-in kernels.

values = numpy.array([-8, -3.3, 0.2, 7, -16, -1.2, 8.5, 14], dtype=numpy.float32)
plan = make_quantization_plan(QuantizationFormat.INT4, values.size, block_size=4)
destination = onnx.TypeProto()
destination.struct_type.CopyFrom(make_quantization_type(plan))
encode = oh.make_node("Quantize", ["X"], ["Q"], domain="ai.rt", type=destination)
decode = oh.make_node("Dequantize", ["Q"], ["Y"], domain="ai.rt", dtype=onnx.TensorProto.DOUBLE)
graph = oh.make_graph(
    [encode, decode],
    "automatic_quantization",
    [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [values.size])],
    [
        oh.make_value_info("Q", destination),
        oh.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, [values.size]),
    ],
)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 21), oh.make_opsetid("ai.rt", 1)])

# %%
# Calibrate scales from the input
# -------------------------------
#
# Signed INT4 codes span [-8, 7]. The two blocks fit that range with scales
# 1 and 2 respectively. Reconstruction rounds to those grids; requesting a
# DOUBLE output does not recover precision lost during quantization.

context = RuntimeContext()
context.set("X", tensor_from_proto(onh.from_array(values, name="X")))
session = RuntimeSession(model)
session.run(context)
automatic = numpy.from_dlpack(context.get("Y"))
encoded = context.get_value("Q")
assert isinstance(encoded, onnx.EncodedValueProto)
assert automatic.dtype == numpy.float64
numpy.testing.assert_array_equal(automatic, [-8, -3, 0, 7, -16, -2, 8, 14])
print("Input:     ", values)
print("Automatic: ", automatic)
print("Encoded output:", type(encoded).__name__)

# %%
# Override the scales explicitly
# ------------------------------
#
# The optional second input supplies one scale per block (or a scalar for
# all blocks). Here scales 2 and 4 deliberately coarsen the reconstruction.
# The parameter dtype can differ from the input dtype. Other optional inputs
# supply zero points, offsets, learned tables, transforms and outlier indices;
# choosing a profile does not train those parameters.

explicit_model = onnx.ModelProto()
explicit_model.CopyFrom(model)
explicit_model.graph.node[0].input.append("scales")
explicit_model.graph.input.append(
    oh.make_tensor_value_info("scales", onnx.TensorProto.DOUBLE, [2])
)
scales = numpy.array([2, 4], dtype=numpy.float64)
explicit_context = RuntimeContext()
explicit_context.set("X", context.get("X"))
explicit_context.set("scales", tensor_from_proto(onh.from_array(scales, name="scales")))
explicit_session = RuntimeSession(explicit_model)
explicit_session.run(explicit_context)
explicit = numpy.from_dlpack(explicit_context.get("Y"))
numpy.testing.assert_array_equal(explicit, [-8, -4, 0, 8, -16, 0, 8, 16])
print("Explicit:  ", explicit)

# %%
# Serialize and reuse the encoded value
# ------------------------------------
#
# The encoded message contains its inline storage type and calibrated
# parameters. A Dequantize-only graph can therefore reconstruct it without
# the original input or plan. It belongs in ``encoded_initializer``, not in
# the ordinary tensor ``initializer`` collection.

restored = onnx.EncodedValueProto()
restored.ParseFromString(encoded.SerializeToString())
initializer_graph = oh.make_graph(
    [decode],
    "decode_encoded_initializer",
    [],
    [oh.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, [values.size])],
)
initializer_graph.encoded_initializer.append(restored)
initializer_model = oh.make_model(initializer_graph, opset_imports=[oh.make_opsetid("ai.rt", 1)])
initializer_context = RuntimeContext()
initializer_session = RuntimeSession(initializer_model)
initializer_session.run(initializer_context)
reconstructed = numpy.from_dlpack(initializer_context.get("Y"))
numpy.testing.assert_array_equal(reconstructed, automatic)
print("Serialized:", reconstructed)

# %%
# Compare reconstruction errors
# -----------------------------
#
# These intentionally coarse scales make the difference visible. Neither
# simple range calibration nor this small example guarantees optimal
# accuracy for a real model.

figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(10, 4))
axes[0].plot(values, "o-", label="Input")
axes[0].plot(automatic, "x--", label="Automatic scales")
axes[0].plot(explicit, "+:", label="Explicit scales")
axes[0].set_title("INT4 reconstruction")
axes[0].set_xlabel("Element")
axes[0].legend()
indices = numpy.arange(values.size)
axes[1].bar(indices - 0.2, numpy.abs(automatic - values), width=0.4, label="Automatic")
axes[1].bar(indices + 0.2, numpy.abs(explicit - values), width=0.4, label="Explicit")
axes[1].set_title("Absolute reconstruction error")
axes[1].set_xlabel("Element")
axes[1].legend()
figure.tight_layout()
matplotlib.pyplot.show()
