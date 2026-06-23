"""
.. _l-example-plot-raw-data-callback:

Track tensor weights while parsing with a raw_data callback
===========================================================

This example shows how to use :attr:`onnx_light.onnx.ParseOptions.raw_data_callback`
to hook into model parsing.  The callback is invoked for every
:class:`onnx_light.onnx.TensorProto` once its ``raw_data`` has been resolved
(including external-data tensors, after their bytes have been loaded).

For each tensor the callback may return a *deleter* — a zero-argument callable
that runs when the tensor's ``raw_data`` is released (when the model and every
copy sharing the same buffer go out of scope).  This lets callers take custom
ownership of tensor data and register matching cleanup, regardless of whether
the bytes live on disk (``no_copy`` view of an mmap or external file) or in CPU
memory.  Returning ``None`` leaves the tensor's ownership unchanged, which makes
the callback equally usable as a read-only inspection hook.
"""

import numpy as np

import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
import onnx_light.onnx as onnxl

# %%
# Build a tiny model with a couple of weight tensors
# --------------------------------------------------
#
# Both initializers are stored as ``raw_data`` (this is what
# :func:`onnx_light.onnx.numpy_helper.from_array` produces), so the callback
# fires once per tensor.

w0 = np.random.randn(8, 8).astype(np.float32)
w1 = np.random.randn(8).astype(np.float32)

initializers = [onh.from_array(w0, name="W0"), onh.from_array(w1, name="W1")]
graph = oh.make_graph([], "demo_graph", [], [], initializer=initializers)
onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)

serialized = onnx_model.SerializeToString()
print(f"Number of initializers: {len(onnx_model.graph.initializer)}")

# %%
# Define the parsing callback
# ---------------------------
#
# ``raw_data_callback`` receives the freshly parsed tensor.  Here we record the
# name and byte size of every tensor and return a deleter that notes when the
# underlying buffer is released.

parsed_tensors = []
released_tensors = []


def on_raw_data(tensor: onnxl.TensorProto):
    """Records the tensor and returns a deleter run when its raw_data is freed."""
    parsed_tensors.append((tensor.name, len(tensor.raw_data)))

    def deleter():
        released_tensors.append(tensor.name)

    return deleter


# %%
# Parse the model with the callback enabled
# -----------------------------------------
#
# The callback is set on :class:`onnx_light.onnx.ParseOptions` and passed to
# ``ParseFromString``.  Because nested protos share the same options, it fires
# for every tensor in the whole model.

options = onnxl.ParseOptions()
options.raw_data_callback = on_raw_data

parsed_model = onnxl.ModelProto()
parsed_model.ParseFromString(serialized, options)

for name, size in parsed_tensors:
    print(f"parsed tensor {name!r}: {size} bytes of raw_data")

# %%
# The tensor data is fully usable: attaching a deleter does not move the bytes.

np.testing.assert_array_equal(onh.to_array(parsed_model.graph.initializer[0]), w0)
np.testing.assert_array_equal(onh.to_array(parsed_model.graph.initializer[1]), w1)
print("Tensor values match the originals.")

# %%
# Releasing the model triggers the registered deleters
# ----------------------------------------------------
#
# When the parsed model (and every tensor sharing the same buffer) is released,
# each deleter runs exactly once.

del parsed_model

import gc

gc.collect()

print(f"released tensors: {released_tensors}")

# %%
# Read-only inspection
# --------------------
#
# Returning ``None`` from the callback leaves ownership untouched, so it can be
# used purely to inspect tensors as they are parsed.

names = []

inspect_options = onnxl.ParseOptions()


def record_name(tensor: onnxl.TensorProto):
    """Records the tensor name and returns None to keep ownership unchanged."""
    names.append(tensor.name)
    return None


inspect_options.raw_data_callback = record_name

inspected = onnxl.ModelProto()
inspected.ParseFromString(serialized, inspect_options)
print(f"inspected tensor names: {names}")

# %%
# Reporting progress with ``RawDataCallback``
# -------------------------------------------
#
# :class:`onnx_light.onnx.RawDataCallback` is a ready-made callback object for
# the common case of *only* reporting progress: it forwards every parsed tensor
# to an ``on_tensor`` callable and always returns ``None``, so the default C++
# allocation is left untouched.  Assign an instance to ``raw_data_callback`` to
# print progress without changing how tensor data is owned.

progress = []

progress_options = onnxl.ParseOptions()
progress_options.raw_data_callback = onnxl.RawDataCallback(
    lambda tensor: progress.append(f"{tensor.name}: {len(tensor.raw_data)} bytes")
)

with_progress = onnxl.ModelProto()
with_progress.ParseFromString(serialized, progress_options)
print("\n".join(progress))
