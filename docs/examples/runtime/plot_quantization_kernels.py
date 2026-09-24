"""
.. _l-example-quantization-kernels:

Calibrates quantization with graph kernels
=========================================

This example runs ``ai.rt::Quantize`` and ``ai.rt::Dequantize`` in a native
runtime session. It first demonstrates linear INT8 quantization with a scale
and zero point, then compares automatic NF4 block calibration with explicit
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
# Linear quantization and dequantization with INT8
# -----------------------------------------------
#
# This graph uses **Quantize and Dequantize**, not QuantizeLinear and
# DequantizeLinear, to implement the familiar affine equations:
#
# * ``q = clip(round(X / scale) + zero_point, -128, 127)``
# * ``Y = (q - zero_point) * scale``
#
# One block covers the whole tensor, so a scalar scale and zero point apply
# to every element. They are supplied as floating tensor initializers; the
# zero point must still have an integer value. Unlike QuantizeLinear, the
# intermediate ``Q`` is an EncodedValueProto containing both the INT8 codes
# and their reconstruction parameters, not a bare INT8 tensor.
#
# Dequantize only needs that encoded value and its requested output dtype.
# The endpoints below deliberately demonstrate saturation to the code range.

linear_values = numpy.array([-40, -1, -0.6, 0, 0.6, 1, 40], dtype=numpy.float32)
linear_scale = numpy.array(0.25, dtype=numpy.float32)
linear_zero_point = numpy.array(-3, dtype=numpy.float32)
linear_plan = make_quantization_plan(
    QuantizationFormat.INT8, linear_values.size, block_size=linear_values.size
)
linear_type = onnx.TypeProto()
linear_type.struct_type.CopyFrom(make_quantization_type(linear_plan))
linear_graph = oh.make_graph(
    [
        oh.make_node(
            "Quantize", ["X", "scale", "zero_point"], ["Q"], domain="ai.rt", type=linear_type
        ),
        oh.make_node("Dequantize", ["Q"], ["Y"], domain="ai.rt", dtype=onnx.TensorProto.FLOAT),
    ],
    "linear_quantization",
    [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [linear_values.size])],
    [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [linear_values.size])],
    initializer=[
        onh.from_array(linear_scale, name="scale"),
        onh.from_array(linear_zero_point, name="zero_point"),
    ],
)
linear_model = oh.make_model(
    linear_graph, opset_imports=[oh.make_opsetid("", 21), oh.make_opsetid("ai.rt", 1)]
)
linear_context = RuntimeContext()
linear_context.set("X", tensor_from_proto(onh.from_array(linear_values, name="X")))
linear_session = RuntimeSession(linear_model)
linear_session.run(linear_context)
linear_output = numpy.from_dlpack(linear_context.get("Y"))
reference_codes = numpy.clip(
    numpy.rint(linear_values / linear_scale) + linear_zero_point, -128, 127
).astype(numpy.int8)
reference_output = (reference_codes.astype(numpy.float32) - linear_zero_point) * linear_scale
numpy.testing.assert_array_equal(reference_codes, [-128, -7, -5, -3, -1, 1, 127])
numpy.testing.assert_array_equal(linear_output, reference_output)
numpy.testing.assert_array_equal(linear_output, [-31.25, -1, -0.5, 0, 0.5, 1, 32.5])
assert linear_output.dtype == numpy.float32
print("Linear input:          ", linear_values)
print("Reference INT8 codes:  ", reference_codes)
print("Quantize -> Dequantize:", linear_output)

# %%
# Nonlinear quantization with an NF4 codebook
# -----------------------------------------
#
# NF4 stores four-bit indices into a fixed table of 16 nonuniform values,
# rather than uniformly spaced integer codes. Quantize selects the nearest
# table entry after scaling; Dequantize reconstructs ``scale * codebook[index]``.
# There is no affine zero-point shift.
#
# Each block covers four consecutive elements and has its own scale.
# The type describes the storage layout, not the numerical scales: Quantize
# computes those from each input block. The encoded value keeps the logical
# FLOAT input dtype; Dequantize requests a DOUBLE output independently.
#
# Declaring ``Q`` as a graph output retains it for inspection after the run.
# Encoded outputs use ``get_value`` rather than the tensor-only ``get``.
# Importing the native runtime above registers its built-in kernels.

values = numpy.array([-8, -3.3, 0.2, 7, -16, -1.2, 8.5, 14], dtype=numpy.float32)
plan = make_quantization_plan(QuantizationFormat.NF4, values.size, block_size=4)
codebook = numpy.array(plan.run(0).block(0).codebook, dtype=numpy.float64)
assert codebook.size == 16
assert not numpy.allclose(numpy.diff(codebook), numpy.diff(codebook)[0])
print("Nonuniform NF4 codebook:", codebook)
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
# The NF4 table spans [-1, 1], so calibration uses each block's maximum
# absolute value: 8 and 16 here. Unlike linear quantization, reconstruction
# selects among nonuniform levels. The NumPy reference below searches those
# levels independently of the native kernels.


def reconstruct_nf4(source, block_scales):
    """Returns the nearest scaled NF4 table entry for each source element."""
    blocks = source.astype(numpy.float64).reshape(-1, 4)
    levels = block_scales[:, None] * codebook[None, :]
    distances = numpy.abs(blocks[:, :, None] - levels[:, None, :])
    indices = distances.argmin(axis=2)
    return numpy.take_along_axis(levels, indices, axis=1).reshape(source.shape)


context = RuntimeContext()
context.set("X", tensor_from_proto(onh.from_array(values, name="X")))
session = RuntimeSession(model)
session.run(context)
automatic = numpy.from_dlpack(context.get("Y"))
encoded = context.get_value("Q")
assert isinstance(encoded, onnx.EncodedValueProto)
assert automatic.dtype == numpy.float64
automatic_scales = numpy.max(numpy.abs(values.reshape(-1, 4)), axis=1).astype(numpy.float64)
numpy.testing.assert_array_equal(automatic_scales, [8, 16])
numpy.testing.assert_allclose(
    automatic, reconstruct_nf4(values, automatic_scales), rtol=1e-12, atol=1e-12
)
print("Input:     ", values)
print("Automatic: ", automatic)
print("Encoded output:", type(encoded).__name__)

# %%
# Override the scales explicitly
# ------------------------------
#
# The optional second input supplies one scale per block (or a scalar for
# all blocks). Scales 10 and 20 widen the reconstruction ranges and move
# the nonuniform levels. They need not improve accuracy.
# The parameter dtype can differ from the input dtype. NF4 zero points and
# offsets remain zero. Other profiles may require learned tables or transforms;
# choosing a profile does not train those parameters.

explicit_model = onnx.ModelProto()
explicit_model.CopyFrom(model)
explicit_model.graph.node[0].input.append("scales")
explicit_model.graph.input.append(
    oh.make_tensor_value_info("scales", onnx.TensorProto.DOUBLE, [2])
)
scales = numpy.array([10, 20], dtype=numpy.float64)
explicit_context = RuntimeContext()
explicit_context.set("X", context.get("X"))
explicit_context.set("scales", tensor_from_proto(onh.from_array(scales, name="scales")))
explicit_session = RuntimeSession(explicit_model)
explicit_session.run(explicit_context)
explicit = numpy.from_dlpack(explicit_context.get("Y"))
numpy.testing.assert_allclose(explicit, reconstruct_nf4(values, scales), rtol=1e-12, atol=1e-12)
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
# The two scale choices yield different NF4 reconstruction levels. Neither
# simple range calibration nor this small example guarantees optimal accuracy
# for a real model.

figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(10, 4))
axes[0].plot(values, "o-", label="Input")
axes[0].plot(automatic, "x--", label="Automatic scales")
axes[0].plot(explicit, "+:", label="Explicit scales")
axes[0].set_title("NF4 codebook reconstruction")
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
