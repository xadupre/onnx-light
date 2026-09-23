"""
.. _l-example-quantization-profiles:

Uses every portable quantization profile from Python
===================================================

This example quantizes and dequantizes all 40 profiles exposed by
:mod:`onnx_light.onnx_core.quantization`. It shows the parameters to supply
for affine, scalar-codebook, vector-codebook, transformed and cast storage.
The :ref:`profile catalogue <l-quantized-values>` describes their numerical
contracts and limitations.

These are **onnx-light representations, not vendor-compatible files**.
In particular, selecting ``gptq`` or ``awq`` does not run calibration;
selecting ``aqlm`` does not train codebooks. The small synthetic tables below
demonstrate the API, not trained models or recommended quantization quality.
"""

import numpy
from onnx_light import onnx
from onnx_light.onnx import numpy_helper
from onnx_light.onnx_core.quantization import (
    dequantize_tensor_proto,
    make_quantization_plan,
    quantization_formats,
    quantize_tensor_proto,
)

# %%
# Common conversion path
# ----------------------
#
# ``count`` is the total number of scalar elements, not the number of
# channels or vectors. ``block_size`` is measured in those same elements;
# the last block may be shorter. Flattening uses row-major logical order.
# A plan stores encoding parameters; the resulting message also stores what
# the decoder needs, so decoding does not require the plan.

covered = set()


def roundtrip(values, plan):
    """Returns the encoded message and reconstructed NumPy array."""
    source = numpy_helper.from_array(values, name="weights")
    encoded = quantize_tensor_proto(source, plan)
    restored = numpy_helper.to_array(dequantize_tensor_proto(encoded))
    assert restored.shape == values.shape
    assert restored.dtype == values.dtype
    assert numpy.isfinite(restored).all()
    error = float(numpy.max(numpy.abs(restored.astype(numpy.float64) - values)))
    print(f"{plan.format:16s} blocks={len(plan.blocks):2d} max_abs_error={error:.6g}")
    covered.add(plan.format)
    return encoded, restored


# %%
# Affine integers and grouped quantization
# ---------------------------------------
#
# ``int8`` and ``eetq`` start with signed 8-bit codes; ``int4`` starts with
# signed 4-bit codes. ``gptq``, ``awq`` and ``matmulnbits`` start with unsigned
# 4-bit codes and zero point 8. ``q2_k`` through ``q6_k`` start with signed
# 2--6-bit codes, respectively.
#
# The reconstruction is ``scale * (code - zero_point) + offset``.
# The example supplies a range-based scale explicitly; this simple rule is
# not GPTQ, AWQ or K-quant calibration. For imported K-quant parameters,
# supply effective sub-block scales/offsets, not GGUF packed scale bytes.
#
# ``plan.blocks`` and ``plan.block(i)`` return copies. Always assign the
# modified list back, or use ``plan.set_block(i, block)``.

weights = numpy.linspace(-1, 1, 16, dtype=numpy.float32).reshape(4, 4)
for profile in (
    "int8",
    "eetq",
    "int4",
    "gptq",
    "awq",
    "matmulnbits",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
):
    plan = make_quantization_plan(profile, weights.size, block_size=4)
    blocks = plan.blocks
    for block in blocks:
        block.scale = 1.0 / (2 ** (block.bits - 1) - 1)
    plan.blocks = blocks
    roundtrip(weights, plan)


# %%
# Per-channel INT8
# ----------------
#
# ``int8_per_channel`` does not infer an axis. For a matrix whose channels
# are columns, first gather column values into contiguous blocks. Each block
# then has one column's scale. Decode automatically restores the original
# ordering and shape. A zero-valued channel uses scale 1, since scales must
# be strictly positive.

weights = numpy.array([[-1, -10, 0], [1, 10, 0]], dtype=numpy.float32)
plan = make_quantization_plan("int8_per_channel", weights.size, block_size=weights.shape[0])
plan.permutation = numpy.arange(weights.size).reshape(weights.shape).T.ravel().tolist()
for channel in range(weights.shape[1]):
    block = plan.block(channel)
    maximum = float(numpy.max(numpy.abs(weights[:, channel])))
    block.scale = maximum / 127 if maximum > 0 else 1.0
    plan.set_block(channel, block)
_, restored = roundtrip(weights, plan)
numpy.testing.assert_allclose(restored, weights, rtol=0, atol=1e-6)


# %%
# Mixed precision: HQQ, EXL2 and EXL3 families
# ------------------------------------------
#
# These profiles start with signed 4-bit affine blocks. The caller supplies
# the bit allocation and quantization parameters; the profile name does not
# select a trained allocation, an EXL vendor layout or an HQQ optimizer.

weights = numpy.array([-1, 0, 1, -2, 0, 2], dtype=numpy.float32)
for profile in ("hqq", "exl2", "exl3"):
    plan = make_quantization_plan(profile, weights.size, block_size=3)
    blocks = plan.blocks
    blocks[0].bits, blocks[0].scale = 2, 1.0
    blocks[1].bits, blocks[1].scale = 5, 0.5
    plan.blocks = blocks
    _, restored = roundtrip(weights, plan)
    numpy.testing.assert_array_equal(restored, weights)


# %%
# Fixed scalar codebooks and low-bit floating-point levels
# -------------------------------------------------------
#
# ``nf4`` provides 16 normal-float levels; ``iq4_nl`` provides 16 integer
# levels. ``log`` provides zero and signed powers of two from 1/8 to 8.
# ``binary`` uses [-1, 1]. The ternary families use [-1, 0, 1]: five trits
# per byte, except ``tq2_0``, which uses two bits per index.
#
# ``mxfp4``/``nvfp4`` use E2M1 levels; ``mxfp6``/``fp6_llm`` use E3M2
# levels; ``fp8_e4m3`` uses finite E4M3FN levels. They store codebook indices,
# not the corresponding vendor float bit patterns. MX/NV scale rounding and
# products of multiple scale levels must be supplied by the caller.
#
# All these profiles reconstruct ``scale * codebook[index]``. Inspecting
# ``block.codebook`` gives the actual unscaled levels. Here a NumPy nearest-
# level reference also checks the decoder's output.

weights = numpy.linspace(-1, 1, 17, dtype=numpy.float32)
for profile in (
    "nf4",
    "iq4_nl",
    "log",
    "binary",
    "ternary",
    "tq1_0",
    "tq2_0",
    "bitnet",
    "paretoq",
    "tequila",
    "mxfp4",
    "nvfp4",
    "mxfp6",
    "fp6_llm",
    "fp8_e4m3",
):
    plan = make_quantization_plan(profile, weights.size, block_size=weights.size)
    block = plan.block(0)
    block.scale = 1.0 / 127 if profile == "iq4_nl" else 1.0
    plan.set_block(0, block)
    levels = numpy.array(block.codebook) * block.scale
    indices = numpy.abs(weights[:, None] - levels[None, :]).argmin(axis=1)
    _, restored = roundtrip(weights, plan)
    numpy.testing.assert_array_equal(restored, levels[indices].astype(weights.dtype))


# %%
# Supplied vector and additive codebooks
# -------------------------------------
#
# Defaults are 32 entries of width 4 for ``stq1_0``, 256 entries of width 8
# for ``iq1_s``/``quip_sharp``, and two such 256-entry books for ``aqlm``.
# A codebook is flattened in [books, entries, vector_size] order.
# Reconstruction sums one selected vector from each book, then multiplies
# by ``scale``. The encoder chooses books greedily against the residual.
#
# The synthetic tables below retain those default dimensions. Replace them
# with trained tables in real applications. QuIP# additionally needs a
# forward/inverse transform: identity is used here solely to demonstrate
# the required fields, not as a useful QuIP# rotation.

weights = numpy.linspace(-1, 1, 16, dtype=numpy.float32)
for profile in ("stq1_0", "iq1_s", "aqlm", "quip_sharp"):
    plan = make_quantization_plan(profile, weights.size, block_size=weights.size)
    block = plan.block(0)
    table = numpy.empty((block.books, block.entries, block.vector_size))
    for book in range(block.books):
        levels = numpy.linspace(-1, 1, block.entries) / (book + 1)
        table[book] = levels[:, None]
    block.codebook = table.ravel().tolist()
    plan.set_block(0, block)
    if profile == "quip_sharp":
        plan.transform_size = 8
        plan.forward = numpy.eye(8).ravel().tolist()
        plan.inverse = plan.forward
    roundtrip(weights, plan)


# %%
# Sparse outliers: SpQR and SqueezeLLM families
# -------------------------------------------
#
# ``spqr`` starts with signed 4-bit affine blocks. ``squeezellm`` starts
# with a scalar codebook whose 16 levels must be supplied.
# ``outliers`` contains flattened indices in the ORIGINAL tensor, before
# permutation or transformation. Their original values are stored separately
# and restored exactly; selection of these indices is not automatic.

weights = numpy.array([0.125, 1000, -0.25, 0.5], dtype=numpy.float32)
for profile in ("spqr", "squeezellm"):
    plan = make_quantization_plan(profile, weights.size)
    plan.outliers = [1]
    block = plan.block(0)
    if profile == "spqr":
        block.scale = 0.125
    else:
        block.codebook = numpy.linspace(-1, 1, block.entries).tolist()
    plan.set_block(0, block)
    _, restored = roundtrip(weights, plan)
    assert restored[1] == weights[1]


# %%
# Rotations and rescaling: QuaRot and SmoothQuant families
# ------------------------------------------------------
#
# ``quarot`` starts with signed 4-bit affine blocks; ``smoothquant`` with
# signed 8-bit blocks. Both require an explicit inverse pair.
# Consecutive row vectors are multiplied by the forward matrix, then
# quantized. Decode multiplies by the inverse. Matrices are row-major.
# The example supplies an orthogonal rotation or a diagonal rescaling;
# neither is calibrated automatically.

weights = numpy.array([1, 2, -1, -2], dtype=numpy.float32)
for profile in ("quarot", "smoothquant"):
    plan = make_quantization_plan(profile, weights.size)
    matrix = (
        numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2)
        if profile == "quarot"
        else numpy.diag([2.0, 0.5])
    )
    plan.transform_size = 2
    plan.forward = matrix.ravel().tolist()
    plan.inverse = numpy.linalg.inv(matrix).ravel().tolist()
    block = plan.block(0)
    block.scale = 0.5
    plan.set_block(0, block)
    roundtrip(weights, plan)


# %%
# Cast storage, tiles and column-major order
# -----------------------------------------
#
# ``tiled_float`` and ``column_major`` default to FLOAT cast storage,
# without an implicit reordering. Supply the permutation explicitly.
# Here the first plan groups 2-by-2 tiles, the second groups columns;
# FLOAT16 storage demonstrates an optional precision reduction.
# The reconstructed tensor retains its original FLOAT logical dtype.

weights = numpy.arange(16, dtype=numpy.float32).reshape(4, 4) / 8
indices = numpy.arange(weights.size).reshape(weights.shape)
for profile in ("tiled_float", "column_major"):
    plan = make_quantization_plan(profile, weights.size, block_size=4)
    if profile == "tiled_float":
        plan.permutation = indices.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).ravel().tolist()
    else:
        plan.permutation = indices.T.ravel().tolist()
    blocks = plan.blocks
    for block in blocks:
        block.cast_type = onnx.TensorProto.FLOAT16
    plan.blocks = blocks
    encoded, restored = roundtrip(weights, plan)
    numpy.testing.assert_array_equal(restored, weights)


# %%
# Serialization and a model-scoped type catalogue
# -----------------------------------------------
#
# Save the encoded message, not the plan. Parsing the bytes preserves the
# inline layout and all parameters. A model catalogue can instead hold the
# layout under an identifier; decoding then needs that model as well.
# This does not register an ONNX operator or make ordinary tensor kernels
# accept an encoded input.

wire = encoded.SerializeToString()
loaded = onnx.EncodedValueProto()
loaded.ParseFromString(wire)
numpy.testing.assert_array_equal(numpy_helper.to_array(dequantize_tensor_proto(loaded)), weights)

model = onnx.ModelProto()
declaration = model.struct_types.add()
declaration.CopyFrom(loaded.struct_type)
declaration.type_id = 91
loaded.struct_type = onnx.StructTypeProto(type_ref=91)
numpy.testing.assert_array_equal(
    numpy_helper.to_array(dequantize_tensor_proto(loaded, model=model)), weights
)

# %%
# Coverage
# --------
#
# The example fails if a public profile has not been demonstrated.

assert covered == set(quantization_formats()), set(quantization_formats()) - covered
print(f"Demonstrated {len(covered)} quantization profiles.")
