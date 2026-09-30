"""
.. _l-example-quantize-paged-cache:

Quantizes and dequantizes selected pages of a KV cache
=====================================================

This example executes ``ai.rt::QuantizePagedCache`` twice in one graph:
first to quantize one page, then to restore its dense floating-point storage.
Other pages remain unchanged. The cache is a ``PagedCacheProto``, not one
uniformly quantized tensor: K and V can use different formats on each page.

The example uses supported affine INT4/UINT4 storage, not the generic
``Q4_K``/``Q5_K`` profiles. See :doc:`/howto/persistent_feedback` for the
paged-cache and persistence contracts.
"""

import matplotlib.pyplot
import numpy

from onnx_light import onnx
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx.helper as oh
from onnx_light.onnx_proto import verify
from onnx_light.onnx_py._onnxpykernels.runtime import (
    RuntimeContext,
    RuntimeSession,
    tensor_from_proto,
)

# %%
# Create a cache with two pages
# -----------------------------
#
# Each payload has shape [batch=1, heads=1, capacity=2, width=4].
# The first page contains two tokens. The second contains only one:
# its unused row contains NaNs to demonstrate that conversion reads only
# the valid prefix, preserves capacity, and initializes unused output rows.

cache = onnx.PagedCacheProto()
first = cache.blocks.add()
first.start = 0
first.length = 2
first_keys = numpy.linspace(-0.5, 0.5, 8, dtype=numpy.float32).reshape(1, 1, 2, 4)
first.key.CopyFrom(onh.from_array(first_keys))
first.value.CopyFrom(onh.from_array(-first_keys))

second = cache.blocks.add()
second.start = 2
second.length = 1
keys = numpy.array(
    [-1.3, -0.4, 0.3, 1.2, numpy.nan, numpy.nan, numpy.nan, numpy.nan], dtype=numpy.float32
).reshape(1, 1, 2, 4)
values = numpy.array(
    [-0.8, -0.3, 0.2, 0.7, numpy.nan, numpy.nan, numpy.nan, numpy.nan], dtype=numpy.float32
).reshape(1, 1, 2, 4)
second.key.CopyFrom(onh.from_array(keys))
second.value.CopyFrom(onh.from_array(values))

# %%
# Build the quantization/dequantization graph
# ------------------------------------------
#
# The *dtype* of each zero-point input selects the destination format:
# INT4 for K, UINT4 for V, then FLOAT16 and BFLOAT16 for dense restoration.
# Integer zero-point values participate in the affine transform. Floating
# zero-points are only type markers; their values and scales are ignored
# numerically, although scales must still be positive finite FLOAT scalars.
#
# An empty struct declaration lets the runtime validate the cache layout.
# These nodes are onnx-light extensions, not standard ONNX operators.

cache_type = onnx.TypeProto(struct_type=onnx.StructTypeProto())
model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node(
                "QuantizePagedCache",
                ["past", "indices", "key_scale", "key_zero", "value_scale", "value_zero"],
                ["quantized"],
                domain="ai.rt",
            ),
            oh.make_node(
                "QuantizePagedCache",
                ["quantized", "indices", "one", "float16_marker", "one", "bfloat16_marker"],
                ["restored"],
                domain="ai.rt",
            ),
        ],
        "selected_page_roundtrip",
        [
            oh.make_value_info("past", cache_type),
            oh.make_tensor_value_info("indices", onnx.TensorProto.INT64, [1]),
        ],
        [oh.make_value_info("quantized", cache_type), oh.make_value_info("restored", cache_type)],
        initializer=[
            oh.make_tensor("key_scale", onnx.TensorProto.FLOAT, [], [0.25]),
            oh.make_tensor("key_zero", onnx.TensorProto.INT4, [], [0]),
            oh.make_tensor("value_scale", onnx.TensorProto.FLOAT, [], [0.125]),
            oh.make_tensor("value_zero", onnx.TensorProto.UINT4, [], [8]),
            oh.make_tensor("one", onnx.TensorProto.FLOAT, [], [1]),
            oh.make_tensor("float16_marker", onnx.TensorProto.FLOAT16, [], [0]),
            oh.make_tensor("bfloat16_marker", onnx.TensorProto.BFLOAT16, [], [0]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 23), oh.make_opsetid("ai.rt", 1)],
    ir_version=10,
)
verify.verify_model(model)

# %%
# Run with page index 1
# ---------------------
#
# Importing the native runtime registers QuantizePagedCache. Structured
# values use put_value/get_value; tensor inputs use set/get.
# Both cache outputs are declared so they remain available for inspection.

context = RuntimeContext()
context.put_value("past", cache)
context.set("indices", tensor_from_proto(onh.from_array(numpy.array([1], numpy.int64))))
session = RuntimeSession(model)
session.run(context)
quantized = context.get_value("quantized")
restored = context.get_value("restored")
assert isinstance(quantized, onnx.PagedCacheProto)
assert isinstance(restored, onnx.PagedCacheProto)

for output in (quantized, restored):
    assert len(output.blocks) == 2
    numpy.testing.assert_array_equal(onh.to_array(output.blocks[0].key), first_keys)
    numpy.testing.assert_array_equal(onh.to_array(output.blocks[0].value), -first_keys)
    assert output.blocks[1].start == 2
    assert output.blocks[1].length == 1

encoded_page = quantized.blocks[1]
assert encoded_page.encoded_key.affine.storage_type == onnx.TensorProto.INT4
assert encoded_page.encoded_value.affine.storage_type == onnx.TensorProto.UINT4
assert len(encoded_page.encoded_key.raw_data) == 4
assert len(encoded_page.encoded_value.raw_data) == 4
print("Page 0: unchanged FLOAT K/V.")
print("Page 1: INT4 K / UINT4 V, 8 payload bytes instead of 64 (excluding metadata).")

# %%
# Check reconstruction and partial-page capacity
# ----------------------------------------------
#
# Dequantization restores the quantized approximation, not the original
# values. The chosen inputs do not saturate; the maximum error is half the
# corresponding scale. Only the selected page becomes FLOAT16/BFLOAT16.

restored_keys = onh.to_array(restored.blocks[1].key)
restored_values = onh.to_array(restored.blocks[1].value)
assert restored_keys.shape == restored_values.shape == (1, 1, 2, 4)
assert restored.blocks[1].key.data_type == onnx.TensorProto.FLOAT16
assert restored.blocks[1].value.data_type == onnx.TensorProto.BFLOAT16
expected_keys = numpy.clip(numpy.rint(keys[:, :, :1] / 0.25), -8, 7) * 0.25
expected_values = (numpy.clip(numpy.rint(values[:, :, :1] / 0.125) + 8, 0, 15) - 8) * 0.125
numpy.testing.assert_array_equal(restored_keys[:, :, :1], expected_keys)
numpy.testing.assert_array_equal(restored_values[:, :, :1].astype(numpy.float32), expected_values)
numpy.testing.assert_array_equal(restored_keys[:, :, 1:], 0)
numpy.testing.assert_array_equal(restored_values[:, :, 1:].astype(numpy.float32), 0)
assert numpy.max(numpy.abs(restored_keys[:, :, :1] - keys[:, :, :1])) <= 0.125
assert (
    numpy.max(numpy.abs(restored_values[:, :, :1].astype(numpy.float32) - values[:, :, :1]))
    <= 0.0625
)
print("Restored page: FLOAT16 K / BFLOAT16 V, capacity 2, valid length 1.")
print("K:", restored_keys[:, :, :1].ravel())
print("V:", restored_values[:, :, :1].astype(numpy.float32).ravel())

# %%
# Serialize and reuse the mixed-format cache
# ------------------------------------------
#
# PagedCacheProto preserves the independent payload formats. The restored
# message can be fed into another session; here the same graph requantizes
# and dequantizes it again with identical parameters.

reloaded = onnx.PagedCacheProto()
reloaded.ParseFromString(quantized.SerializeToString())
context.put_value("past", reloaded)
session.run(context)
again = context.get_value("restored")
numpy.testing.assert_array_equal(onh.to_array(again.blocks[1].key), restored_keys)
numpy.testing.assert_array_equal(
    onh.to_array(again.blocks[1].value).astype(numpy.float32),
    restored_values.astype(numpy.float32),
)

# %%
# Compare the valid token before and after conversion
# ---------------------------------------------------

figure, axes = matplotlib.pyplot.subplots(1, 2, figsize=(9, 3))
for axis, name, source, result in (
    (axes[0], "K: INT4 -> FLOAT16", keys, restored_keys),
    (axes[1], "V: UINT4 -> BFLOAT16", values, restored_values),
):
    axis.plot(source[:, :, :1].ravel(), "o-", label="Original")
    axis.plot(result[:, :, :1].astype(numpy.float32).ravel(), "x--", label="Restored")
    axis.set_title(name)
    axis.set_xlabel("Head component")
    axis.legend()
figure.tight_layout()
