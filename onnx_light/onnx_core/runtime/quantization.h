// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <array>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Selects the numerical reconstruction of an independently encoded block. */
enum class QuantizationMethod { kAffine = 0, kCodebook = 1, kCast = 2 };

/**
 * Describes one contiguous block in quantization order.
 *
 * Affine reconstruction is scale * (code - zero_point) + offset. Codebook reconstruction
 * is scale * sum(book[index]), with tables ordered [books, entries, vector_size].
 * Quantization uses nearest-even affine rounding or greedy nearest-residual
 * codebook selection, not calibration or codebook training.
 */
struct QuantizationBlock {
  /// Counts scalar elements in this block, not bytes or codebook vectors (at most UINT32_MAX).
  uint64_t count = 0;
  /// Selects affine reconstruction, codebook reconstruction or floating-point cast storage.
  QuantizationMethod method = QuantizationMethod::kAffine;
  /// Sets the affine code or codebook index width, from 1 to 16; does not size cast storage.
  uint32_t bits = 4;
  /// Selects signed affine codes; does not change codebook indices or casts.
  bool signed_codes = true;
  /// Multiplies reconstructed values for all methods; must be finite and strictly positive.
  double scale = 1;
  /// Sets an integer affine zero point in the code range; must be zero for other methods.
  double zero_point = 0;
  /// Adds a finite affine reconstruction offset; must be zero for other methods.
  double offset = 0;
  /// Counts additive codebooks; must be positive for codebook reconstruction.
  uint32_t books = 1;
  /// Counts entries per codebook; must be positive and no greater than 2^bits.
  uint32_t entries = 0;
  /// Counts components per codebook entry; must be positive for codebook reconstruction.
  uint32_t vector_size = 1;
  /// Stores finite values in [books, entries, vector_size] order; must be empty for other methods.
  std::vector<double> codebook;
  /// Packs five trits per byte; requires one scalar codebook with exactly three entries.
  bool base3 = false;
  /// Selects FLOAT, DOUBLE, FLOAT16 or BFLOAT16 physical storage for the cast method only.
  int32_t cast_type = TensorProto::FLOAT16;
};

/**
 * Describes a portable onnx-light quantization, not a vendor packing ABI.
 *
 * @par Construction and conversion
 * Calls MakeQuantizationPlan(format, count, block_size) to initialize profile defaults,
 * then sets the required block parameters, ordering, transforms and outlier indices.
 * Passes the completed plan to QuantizeTensor() or QuantizeTensorProto().
 * DequantizeTensor() and DequantizeTensorProto() need only the resulting encoded value
 * (plus its type catalogue when referenced), not this plan.
 *
 * The source must be a finite FLOAT, DOUBLE, FLOAT16 or BFLOAT16 tensor with a concrete
 * shape. Scalars and empty tensors are supported. The logical output shape and dtype
 * remain those of the source, regardless of the physical code or cast dtype.
 * The converters do not modify source storage; encoded and decoded results own their data.
 * External TensorProto payloads must be loaded before quantization.
 * TensorProto::LoadExternalData() may retain EXTERNAL metadata; loaded raw_data is accepted.
 *
 * @par Block coverage and numerical methods
 * Counts and block sizes measure scalar elements, not bytes or codebook vectors.
 * Blocks cover the flattened tensor exactly, in order; each count is at most UINT32_MAX.
 * The factory uses blocks of at most block_size elements and a possibly shorter final block.
 * It returns no blocks for count=0. It does not infer channels, axes, tiles or scales.
 * All profiles start with scale=1, offset=0, and zero_point=0, except the unsigned
 * GPTQ/AWQ/MatMulNBits defaults, whose zero_point is 8.
 *
 * - Affine reconstruction is scale * (code - zero_point) + offset. Signed b-bit codes
 *   range from -2^(b-1) to 2^(b-1)-1; unsigned codes range from 0 to 2^b-1.
 *   Encoding clips to this range and rounds halfway cases to even. A scale must be
 *   finite and strictly positive, a zero point an integer in the code range, and an
 *   offset finite. Clipping is permitted; it is not reported as an invalid input.
 * - Codebook reconstruction is scale * sum(codebook[book, index, component]).
 *   Supplies exactly books * entries * vector_size finite numbers in row-major order.
 *   Each book has the same entries and vector_size. The encoder greedily selects the
 *   closest vector to the remaining residual, one book at a time, not a globally
 *   optimal combination. Equal distances select the first entry. A partial final
 *   vector compares and reconstructs only its logical components but stores one
 *   index per book. Zero point and offset must both be zero.
 * - Cast storage converts source/scale to cast_type and reconstructs cast_value*scale.
 *   Supported cast types are FLOAT, DOUBLE, FLOAT16 and BFLOAT16. Zero point and offset
 *   must be zero. Affine and cast blocks must not contain a codebook.
 *
 * Binary codes/indices use 1--16 bits, least-significant bits first. Base-3 packing
 * requires books=1, entries=3 and vector_size=1, and packs five trits per byte.
 * Unused high bits/trits are zero. Codebook vector width, multiple books and metadata
 * mean that bits is not necessarily the total number of stored bits per source value.
 *
 * @par Ordering, transforms and outliers
 * Encoding performs the following operations on the flattened row-major source:
 * 1. Saves values at outliers and replaces them with zero.
 * 2. Gathers source[permutation[i]] into quantization position i.
 * 3. Multiplies consecutive row vectors by forward.
 * 4. Encodes the resulting consecutive blocks.
 *
 * Decoding applies inverse, scatters through permutation, then restores outliers exactly.
 * An empty permutation is identity; otherwise it contains every source index exactly once.
 * Outlier indices are unique, in range, and refer to the original source before reordering.
 * They need not be sorted. With transform_size=N>0, forward and inverse each contain N*N
 * finite row-major entries, N divides the total element count, and forward*inverse must
 * equal identity within absolute tolerance 1e-6 per entry. The matrices need not be
 * orthogonal. With transform_size=0, both matrices must be empty.
 * Transform groups and quantization blocks are independent partitions of the values.
 *
 * @par Profile-by-profile configuration
 * The following recipes assume count=16. They show factory calls and the parameters
 * to set before conversion; apply block settings to every block unless stated otherwise.
 * Scales and tables shown as examples are not calibrated or trained.
 * A profile provides defaults, not a restriction to a vendor's block size or bit allocation.
 *
 * <table>
 * <tr><th>Profile</th><th>Factory call and required configuration</th></tr>
 * <tr><td>int8, eetq</td><td>MakeQuantizationPlan("int8", 16, 4), or "eetq": signed 8-bit blocks.
 * Set each block.scale, for example 1.0/127 for values in [-1,1].</td></tr>
 * <tr><td>int4</td><td>MakeQuantizationPlan("int4", 16, 4): signed 4-bit blocks.
 * Set block.scale, for example 1.0/7.</td></tr>
 * <tr><td>int8_per_channel</td><td>MakeQuantizationPlan("int8_per_channel", 16, 4): signed INT8.
 * Group each channel into a block using permutation when it is not already contiguous,
 * and assign one scale per channel. See the column-grouping example below.</td></tr>
 * <tr><td>gptq, awq, matmulnbits</td><td>MakeQuantizationPlan("gptq", 16, 4), or
 * "awq"/"matmulnbits": unsigned 4-bit blocks, zero_point=8. Supply each group's scale and integer
 * zero_point; optionally supply a real offset. No Hessian calculation or activation calibration
 * runs.</td></tr> <tr><td>q2_k, q3_k, q4_k, q5_k, q6_k</td><td>MakeQuantizationPlan("q2_k", 16, 4),
 * and analogously for the other names: signed 2, 3, 4, 5 or 6-bit blocks. Supply effective
 * sub-block scale and offset, including any hierarchical scale products. No GGUF scale
 * packing.</td></tr> <tr><td>hqq, exl2, exl3</td><td>MakeQuantizationPlan("exl2", 16, 8), or
 * "hqq"/"exl3": signed 4-bit defaults. Override blocks[0].bits=3 and blocks[1].bits=5, for example,
 * then supply their respective scales/zero points/offsets. No bit allocation is inferred.</td></tr>
 * <tr><td>nf4</td><td>MakeQuantizationPlan("nf4", 16): one supplied-by-default 16-level
 * normal-float scalar table with 4-bit indices. Set scale; 1 preserves its [-1,1] levels.</td></tr>
 * <tr><td>iq4_nl</td><td>MakeQuantizationPlan("iq4_nl", 16): a fixed 16-entry signed integer
 * scalar table, 4-bit indices. Set scale, for example 1.0/127.</td></tr>
 * <tr><td>binary</td><td>MakeQuantizationPlan("binary", 16): one-bit indices into [-1,1].
 * Set scale to the desired reconstructed magnitude.</td></tr>
 * <tr><td>ternary, tq1_0, bitnet, paretoq, tequila</td><td>MakeQuantizationPlan("ternary", 16),
 * or any other listed name: [-1,0,1], base3=true. Set scale for the nonzero magnitude.
 * Represents ternary values, not the associated training algorithms.</td></tr>
 * <tr><td>tq2_0</td><td>MakeQuantizationPlan("tq2_0", 16): the same [-1,0,1] table with
 * base3=false and two-bit indices. Set scale.</td></tr>
 * <tr><td>stq1_0</td><td>MakeQuantizationPlan("stq1_0", 16): books=1, entries=32, vector_size=4,
 * bits=5. Supply 128 table values and scale. No code/sign splitting or scatter layout.</td></tr>
 * <tr><td>iq1_s</td><td>MakeQuantizationPlan("iq1_s", 16): books=1, entries=256, vector_size=8,
 * bits=8. Supply 2048 table values and scale.</td></tr>
 * <tr><td>quip_sharp</td><td>MakeQuantizationPlan("quip_sharp", 16): the same table defaults as
 * iq1_s. Supply 2048 table values, scale, and a nonzero transform_size with both matrices.
 * The transform width need not equal the codebook vector width.</td></tr>
 * <tr><td>aqlm</td><td>MakeQuantizationPlan("aqlm", 16): books=2, entries=256, vector_size=8,
 * bits=8. Supply 4096 values ordered [2,256,8] and scale. See the codebook example below.</td></tr>
 * <tr><td>spqr</td><td>MakeQuantizationPlan("spqr", 16): signed 4-bit affine base. Set scale and
 * optionally zero_point/offset, then set outliers, for example {0,15}.</td></tr>
 * <tr><td>squeezellm</td><td>MakeQuantizationPlan("squeezellm", 16): books=1, entries=16,
 * vector_size=1, bits=4. Supply 16 scalar levels and scale; set outliers as needed.</td></tr>
 * <tr><td>log</td><td>MakeQuantizationPlan("log", 16): a 15-entry table containing zero and signed
 * powers of two from 2^-3 through 2^3, using four-bit indices. Set scale; replace
 * codebook and entries (and bits if necessary) to change the logarithmic range/rule.</td></tr>
 * <tr><td>fp6_llm, mxfp6</td><td>MakeQuantizationPlan("fp6_llm", 16), or "mxfp6": scalar E3M2
 * finite levels with six-bit indices. Supply effective scale.</td></tr>
 * <tr><td>mxfp4, nvfp4</td><td>MakeQuantizationPlan("mxfp4", 16, 8), or "nvfp4": scalar E2M1
 * finite levels with four-bit indices. Supply each block's effective scale; any E8M0/FP8
 * scale rounding or multiplication of global and local scales is the caller's job.</td></tr>
 * <tr><td>fp8_e4m3</td><td>MakeQuantizationPlan("fp8_e4m3", 16): finite E4M3FN scalar levels,
 * with nonfinite entries excluded, using eight-bit indices. Supply scale.</td></tr>
 * <tr><td>quarot</td><td>MakeQuantizationPlan("quarot", 16): signed 4-bit affine blocks.
 * Supply scale, a nonzero transform_size, forward and inverse. See the rotation example.</td></tr>
 * <tr><td>smoothquant</td><td>MakeQuantizationPlan("smoothquant", 16): signed 8-bit affine blocks.
 * Supply scale and an explicit forward/inverse rescaling pair; no activation statistics
 * are collected. See the rescaling example.</td></tr>
 * <tr><td>tiled_float</td><td>MakeQuantizationPlan("tiled_float", 16, 4): FLOAT cast blocks.
 * Supply permutation to group tiles and optionally change each block.cast_type,
 * for example to TensorProto::FLOAT16. No tiling is inferred from the name.</td></tr>
 * <tr><td>column_major</td><td>MakeQuantizationPlan("column_major", 16): FLOAT cast storage.
 * Supply a column-major gather permutation; decoding restores the original logical order.</td></tr>
 * </table>
 *
 * @par Basic affine conversion
 * @code{.cpp}
 * auto input = Tensor::FromFloat("weight", {4}, {-1, -0.5f, 0.5f, 1});
 * auto plan = MakeQuantizationPlan("int4", 4);
 * plan.blocks[0].scale = 0.25;
 * auto encoded = QuantizeTensor(input, plan);
 * auto restored = DequantizeTensor(encoded);
 * @endcode
 *
 * @par Column grouping and tiled storage
 * For a 4-by-4 row-major matrix whose channels are columns:
 * @code{.cpp}
 * auto channels = MakeQuantizationPlan("int8_per_channel", 16, 4);
 * channels.permutation = {0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15};
 * for (auto &block : channels.blocks)
 *   block.scale = 1.0 / 127;  // Replace with the corresponding channel's scale.
 * auto columns = MakeQuantizationPlan("column_major", 16);
 * columns.permutation = channels.permutation;
 * auto tiles = MakeQuantizationPlan("tiled_float", 16, 4);
 * tiles.permutation = {0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15};
 * for (auto &block : tiles.blocks)
 *   block.cast_type = TensorProto::FLOAT16;
 * @endcode
 *
 * @par Supplied scalar, vector and additive tables
 * The same assignment pattern applies to stq1_0, iq1_s, quip_sharp, aqlm and squeezellm.
 * Use their dimensions from the table above, or explicitly change books, entries,
 * vector_size and bits together. This small AQLM-family example overrides the default
 * dimensions with two synthetic books, four two-component entries each:
 * @code{.cpp}
 * auto additive = MakeQuantizationPlan("aqlm", 16);
 * auto &block = additive.blocks[0];
 * block.books = 2;
 * block.entries = 4;
 * block.vector_size = 2;
 * block.bits = 2;
 * block.codebook = {-1,-1, 0,0, 1,1, 2,2,
 *                  -0.25,-0.25, 0,0, 0.25,0.25, 0.5,0.5};
 * block.scale = 1;
 * @endcode
 * Real applications supply trained tables; this example does not train a codebook.
 * QuIP# additionally requires a transform pair, even when a table has been supplied.
 *
 * @par Rotations, rescaling and sparse exceptions
 * The following mutually inverse matrices act on consecutive pairs, before quantization.
 * The first pair is a scaled Hadamard transform; the second is diagonal rescaling.
 * @code{.cpp}
 * auto rotated = MakeQuantizationPlan("quarot", 16);
 * rotated.transform_size = 2;
 * rotated.forward = {1,1, 1,-1};
 * rotated.inverse = {0.5,0.5, 0.5,-0.5};
 * rotated.blocks[0].scale = 0.5;
 * auto rescaled = MakeQuantizationPlan("smoothquant", 16);
 * rescaled.transform_size = 2;
 * rescaled.forward = {2,0, 0,0.5};
 * rescaled.inverse = {0.5,0, 0,2};
 * rescaled.blocks[0].scale = 0.125;
 * auto sparse = MakeQuantizationPlan("spqr", 16);
 * sparse.blocks[0].scale = 0.125;
 * sparse.outliers = {0,15};
 * @endcode
 * Outliers are also allowed with other profiles. They are stored exactly and do not
 * require a special quantization method.
 *
 * @par Python usage
 * Uses the same fields through onnx_light.onnx_core.quantization. Unlike the C++ vector,
 * Python plan.blocks is a list of copies: assign the modified list back. Similarly,
 * plan.block(i) returns a copy, and plan.set_block(i, block) replaces the stored block.
 * @code{.py}
 * import numpy
 * from onnx_light.onnx import numpy_helper
 * from onnx_light.onnx_core.quantization import (
 *     make_quantization_plan, quantize_tensor_proto, dequantize_tensor_proto,
 * )
 * weights = numpy.array([-1, 0, 1], dtype=numpy.float32)
 * plan = make_quantization_plan("int4", weights.size)
 * block = plan.block(0)
 * block.scale = 0.25
 * plan.set_block(0, block)
 * encoded = quantize_tensor_proto(numpy_helper.from_array(weights), plan)
 * restored = numpy_helper.to_array(dequantize_tensor_proto(encoded))
 * @endcode
 * The Python guide docs/howto/quantized_values.rst contains configuration examples for
 * every profile; docs/examples/runtime/plot_quantization_profiles.py executes all of them.
 *
 * @par Validation, persistence and limitations
 * Quantization validates the completed plan, not just the factory name. Missing learned
 * codebooks, required transforms, invalid coverage/indices/parameters, nonfinite values
 * and cast overflow raise std::invalid_argument (ValueError in Python).
 * The factory can therefore return a plan that is not yet ready to encode.
 * The selected profile name is saved after the prefix @c onnx_light.quantization.v1/;
 * changing format alone does not update existing block parameters.
 * Encoding and decoding reject names not returned by QuantizationFormats(), even after
 * editing a plan or an encoded layout.
 *
 * The encoded value stores the tables, scales, transforms, permutation and exact outliers,
 * with an inline layout or a model-scoped type reference. Serialize the EncodedValueProto
 * and retain its catalogue when referenced. A plan is an encoding configuration, not
 * a persistent cache or an ONNX operator schema. Tensor-only operators require explicit
 * dequantization or a schema/kernel supporting encoded values.
 *
 * Profile names do not promise vendor packing, published bits-per-weight figures,
 * calibration/training, or accuracy. In particular, float-like profiles store nearest-level
 * codebook indices, not vendor float bit patterns, and their first-entry tie rule may
 * differ from vendor rounding. Parameters/codebooks occupy payload space; consecutive
 * blocks with identical layouts share a descriptor, not their parameter values.
 *
 * @see MakeQuantizationPlan, QuantizationBlock, QuantizationFormats, QuantizeTensor,
 * QuantizeTensorProto, DequantizeTensor, DequantizeTensorProto
 */
struct QuantizationPlan {
  /// Names the profile in the encoded layout; changing it does not reinitialize blocks.
  std::string format;
  /// Covers the tensor with consecutive blocks in post-permutation, post-transform order.
  std::vector<QuantizationBlock> blocks;
  /// Maps quantization positions to original flattened indices; empty selects identity.
  std::vector<int64_t> permutation;
  /// Sets the row-vector transform width; zero requires empty forward/inverse matrices.
  uint32_t transform_size = 0;
  /// Stores the finite transform_size-by-transform_size forward matrix in row-major order.
  std::vector<double> forward;
  /// Stores the corresponding inverse matrix in row-major order.
  std::vector<double> inverse;
  /// Selects unique original flattened indices whose exact values bypass quantization.
  std::vector<int64_t> outliers;
};

/** Returns the portable profile names as a compile-time array without dynamic allocation. */
constexpr std::array<std::string_view, 40> QuantizationFormats() {
  return {"int8",        "int8_per_channel",
          "int4",        "gptq",
          "awq",         "eetq",
          "matmulnbits", "q2_k",
          "q3_k",        "q4_k",
          "q5_k",        "q6_k",
          "hqq",         "exl2",
          "exl3",        "nf4",
          "iq4_nl",      "binary",
          "ternary",     "tq1_0",
          "tq2_0",       "bitnet",
          "paretoq",     "tequila",
          "stq1_0",      "iq1_s",
          "aqlm",        "quip_sharp",
          "spqr",        "squeezellm",
          "log",         "fp6_llm",
          "fp8_e4m3",    "mxfp4",
          "mxfp6",       "nvfp4",
          "quarot",      "smoothquant",
          "tiled_float", "column_major"};
}

/**
 * Creates block defaults for a catalogue profile.
 *
 * See @ref QuantizationPlan for the complete profile-by-profile configuration table,
 * required scales/codebooks/transforms, numerical rules, validation constraints,
 * and C++/Python usage examples. The returned plan may require additional parameters
 * before it can be passed to QuantizeTensor() or QuantizeTensorProto().
 *
 * @param format Selects one of the names returned by QuantizationFormats().
 * @param count Counts the source tensor's scalar elements, not bytes or codebook vectors.
 * @param block_size Sets the maximum elements per block, in [1, UINT32_MAX].
 * The final block may be shorter; count=0 yields no blocks.
 * @returns A plan with profile defaults, scale=1, no permutation, no transforms and
 * no outliers. Fixed codebooks are populated; learned codebooks must be supplied.
 * @throws std::invalid_argument If the profile is unknown or block_size is invalid.
 * @see QuantizationPlan
 */
QuantizationPlan MakeQuantizationPlan(const std::string &format, uint64_t count,
                                      uint64_t block_size = 128);

/** Quantizes a finite floating-point tensor into an owned, self-describing encoded value. */
RuntimeValue QuantizeTensor(const Tensor &tensor, const QuantizationPlan &plan);

/** Dequantizes a supported encoded runtime value to its declared floating-point tensor type. */
Tensor DequantizeTensor(const RuntimeValue &value, const StructTypeCatalogue &catalogue = {},
                        RawBufferAllocator *allocator = nullptr);

/** Quantizes a loaded TensorProto without changing the source message. */
EncodedValueProto QuantizeTensorProto(const TensorProto &tensor, const QuantizationPlan &plan);

/** Dequantizes a supported encoded message into an owned TensorProto. */
TensorProto DequantizeTensorProto(const EncodedValueProto &value,
                                  const StructTypeCatalogue &catalogue = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
