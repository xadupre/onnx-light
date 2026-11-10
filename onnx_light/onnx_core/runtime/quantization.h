// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <array>
#include <optional>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Selects the numerical reconstruction of an independently encoded block. */
enum class QuantizationMethod { kAffine = 0, kCodebook = 1, kCast = 2 };

/** Selects a quantization format, including native GGML and ORT-compatible storage. */
enum class QuantizationFormat {
  kInt8,
  kInt8PerChannel,
  kInt4,
  kGptq,
  kAwq,
  kEetq,
  kMatmulnbits,
  kHqq,
  kExl2,
  kExl3,
  kNf4,
  kBinary,
  kTernary,
  kBitnet,
  kParetoq,
  kTequila,
  kStq10,
  kAqlm,
  kQuipSharp,
  kSpqr,
  kSqueezellm,
  kLog,
  kFp6Llm,
  kFp8E4m3,
  kMxfp4,
  kMxfp6,
  kNvfp4,
  kQuarot,
  kSmoothquant,
  kTiledFloat,
  kColumnMajor,
  kOrtMatmulnbitsInt2,
  kOrtMatmulnbitsInt4,
  kOrtMatmulnbitsInt8,
  kMxfp8,
  kMxfp6E2m3,
  kQ40,
  kQ80,
  kQ4K,
  kQ5K,
  kQ6K,
  kQ2K,
  kQ3K,
  kIq4Nl,
  kIq1S,
  kTq10,
  kTq20,
};

/** Returns whether the format uses exact native GGML block storage. */
constexpr bool IsGgufQuantizationFormat(QuantizationFormat format) {
  return format >= QuantizationFormat::kQ40 && format <= QuantizationFormat::kTq20;
}

/**
 * Describes the shared type of a run of contiguous blocks.
 * Corresponds to the nine-element parameters constant in a StructTypeProto array element.
 */
struct QuantizationBlockLayout {
  /// Counts scalar elements in this block, not bytes or codebook vectors (at most UINT32_MAX).
  uint64_t count = 0;
  /// Selects affine reconstruction, codebook reconstruction or floating-point cast storage.
  QuantizationMethod method = QuantizationMethod::kAffine;
  /// Sets the affine code or codebook index width, from 1 to 16; does not size cast storage.
  uint32_t bits = 4;
  /// Selects signed affine codes; does not change codebook indices or casts.
  bool signed_codes = true;
  /// Counts additive codebooks; must be positive for codebook reconstruction.
  uint32_t books = 1;
  /// Counts entries per codebook; must be positive and no greater than 2^bits.
  uint32_t entries = 0;
  /// Counts components per codebook entry; must be positive for codebook reconstruction.
  uint32_t vector_size = 1;
  /// Packs five trits per byte; requires one scalar codebook with exactly three entries.
  bool base3 = false;
  /// Must be FLOAT, DOUBLE, FLOAT16 or BFLOAT16 for every method.
  /// Selects physical storage only for the cast method.
  int32_t cast_type = TensorProto::FLOAT16;
};

/** Stores the per-block numerical parameters carried in the encoded payload. */
struct QuantizationBlockParameters {
  /// Multiplies reconstructed values; portable profiles require a strictly positive value.
  double scale = 1;
  /// Sets an affine zero point; ORT profiles also accept floating-point zero points.
  double zero_point = 0;
  /// Adds a finite affine reconstruction offset; must be zero for other methods.
  double offset = 0;
  /// Stores finite values in [books, entries, vector_size] order; must be empty for other methods.
  std::vector<double> codebook;
};

/**
 * Groups consecutive blocks under one shared layout, like a StructTypeProto array.
 * The array dimension is blocks.size(); every block covers layout.count scalar elements.
 * Codebook values remain per-block payload data, not shared type constants.
 */
struct QuantizationRun {
  QuantizationBlockLayout layout;
  std::vector<QuantizationBlockParameters> blocks;
};

/**
 * Describes portable quantization or explicitly ORT-compatible MatMulNBits inputs.
 *
 * The coverage and numerical rules below describe portable profiles. For the three
 * kOrtMatmulnbitsInt* profiles, use MakeMatMulNBitsPlan and its matrix-specific contract.
 * kMxfp8 instead selects the complete fixed-32 QuantizeMxfp8Weights contract, with
 * automatic E8M0 scales, native E4M3FN bytes and fixed FLOAT reconstruction.
 * kMxfp6E2m3 selects QuantizeMxfp6E2m3Weights, with packed E2M3 codes and automatic
 * group-32 E8M0 scales; it does not change the portable E3M2 kMxfp6 profile.
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
 * Proto conversions write directly into exactly sized output raw_data buffers.
 * Dequantization reads EncodedValueProto by const reference, without copying its payload;
 * the RuntimeValue overload delegates to the same decoder. Numerical workspace is still
 * allocated for reconstructed values, transforms and codebook search.
 * External TensorProto payloads must be loaded before quantization.
 * TensorProto::LoadExternalData() may retain EXTERNAL metadata; loaded raw_data is accepted.
 *
 * @par Block coverage and numerical methods
 * Counts and block sizes measure scalar elements, not bytes or codebook vectors.
 * Runs follow the StructTypeProto array representation: each QuantizationRun stores one
 * QuantizationBlockLayout and an array of QuantizationBlockParameters. The layout is
 * shared by every block of the run; parameters remain independent payload values.
 * Runs cover the flattened tensor exactly, in order. Each layout.count is in [1, UINT32_MAX];
 * sum(run.layout.count * run.blocks.size()) must equal the tensor's element count.
 * The factory groups full blocks into one run and a shorter final block into a second run.
 * It returns no runs for count=0. Empty runs are invalid.
 * To use different bit widths or methods, supply separate runs with their own layouts.
 * Adjacent compatible runs share one serialized descriptor. The factory does not infer
 * channels, axes, tiles or scales.
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
 * Decoding applies inverse, scatters through permutation, then restores outliers
 * (rounded to reconstruction_dtype for v2).
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
 * <tr><td>int8, eetq</td><td>MakeQuantizationPlan(QuantizationFormat::kInt8, 16, 4), or "eetq":
 * signed 8-bit blocks. Set each block.scale, for example 1.0/127 for values in [-1,1].</td></tr>
 * <tr><td>int4</td><td>MakeQuantizationPlan(QuantizationFormat::kInt4, 16, 4): signed 4-bit blocks.
 * Set block.scale, for example 1.0/7.</td></tr>
 * <tr><td>int8_per_channel</td><td>MakeQuantizationPlan(QuantizationFormat::kInt8PerChannel, 16,
 * 4): signed INT8. Group each channel into a block using permutation when it is not already
 * contiguous, and assign one scale per channel. See the column-grouping example below.</td></tr>
 * <tr><td>gptq, awq, matmulnbits</td><td>MakeQuantizationPlan(QuantizationFormat::kGptq, 16, 4), or
 * "awq"/"matmulnbits": unsigned 4-bit blocks, zero_point=8. Supply each group's scale and integer
 * zero_point; optionally supply a real offset. No Hessian calculation or activation calibration
 * runs.</td></tr>
 * <tr><td>hqq, exl2,
 * exl3</td><td>MakeQuantizationPlan(QuantizationFormat::kExl2, 16, 8), or "hqq"/"exl3": signed
 * 4-bit defaults. Split the two blocks into separate runs and set runs[0].layout.bits=3 and
 * runs[1].layout.bits=5, then supply each block's scale/zero point/offset.
 * No bit allocation is inferred.</td></tr>
 * <tr><td>nf4</td><td>MakeQuantizationPlan(QuantizationFormat::kNf4, 16): one supplied-by-default
 * 16-level normal-float scalar table with 4-bit indices. Set scale; 1 preserves its [-1,1]
 * levels.</td></tr>
 * <tr><td>binary</td><td>MakeQuantizationPlan(QuantizationFormat::kBinary, 16): one-bit indices
 * into [-1,1]. Set scale to the desired reconstructed magnitude.</td></tr> <tr><td>ternary,
 * bitnet, paretoq, tequila</td><td>MakeQuantizationPlan(QuantizationFormat::kTernary, 16), or any
 * other listed name: [-1,0,1], base3=true. Set scale for the nonzero magnitude. Represents ternary
 * values, not the associated training algorithms.</td></tr>
 * <tr><td>stq1_0</td><td>MakeQuantizationPlan(QuantizationFormat::kStq10, 16): books=1, entries=32,
 * vector_size=4, bits=5. Supply 128 table values and scale. No code/sign splitting or scatter
 * layout.</td></tr>
 * <tr><td>quip_sharp</td><td>MakeQuantizationPlan(QuantizationFormat::kQuipSharp, 16):
 * books=1, entries=256, vector_size=8. Supply 2048 table values, scale, and a nonzero
 * transform_size with both matrices. The transform width need not equal the codebook vector
 * width.</td></tr> <tr><td>aqlm</td><td>MakeQuantizationPlan(QuantizationFormat::kAqlm, 16):
 * books=2, entries=256, vector_size=8, bits=8. Supply 4096 values ordered [2,256,8] and scale. See
 * the codebook example below.</td></tr>
 * <tr><td>spqr</td><td>MakeQuantizationPlan(QuantizationFormat::kSpqr, 16): signed 4-bit affine
 * base. Set scale and optionally zero_point/offset, then set outliers, for example
 * {0,15}.</td></tr>
 * <tr><td>squeezellm</td><td>MakeQuantizationPlan(QuantizationFormat::kSqueezellm, 16): books=1,
 * entries=16, vector_size=1, bits=4. Supply 16 scalar levels and scale; set outliers as
 * needed.</td></tr> <tr><td>log</td><td>MakeQuantizationPlan(QuantizationFormat::kLog, 16): a
 * 15-entry table containing zero and signed powers of two from 2^-3 through 2^3, using four-bit
 * indices. Set scale; replace codebook and entries (and bits if necessary) to change the
 * logarithmic range/rule.</td></tr> <tr><td>fp6_llm,
 * mxfp6</td><td>MakeQuantizationPlan(QuantizationFormat::kFp6Llm, 16), or "mxfp6": scalar E3M2
 * finite levels with six-bit indices. Supply effective scale.</td></tr>
 * <tr><td>mxfp6_e2m3</td><td>MakeQuantizationPlan(QuantizationFormat::kMxfp6E2m3, 32):
 * complete E2M3/group-32/E8M0 matrix codec with automatic scales and FLOAT reconstruction.
 * See QuantizeMxfp6E2m3Weights; portable block customizations are rejected.</td></tr>
 * <tr><td>mxfp4, nvfp4</td><td>MakeQuantizationPlan(QuantizationFormat::kMxfp4, 16, 8), or "nvfp4":
 * scalar E2M1 finite levels with four-bit indices. Supply each block's effective scale; any
 * E8M0/FP8 scale rounding or multiplication of global and local scales is the caller's
 * job.</td></tr> <tr><td>fp8_e4m3</td><td>MakeQuantizationPlan(QuantizationFormat::kFp8E4m3, 16):
 * finite E4M3FN scalar levels, with nonfinite entries excluded, using eight-bit indices. Supply
 * scale.</td></tr> <tr><td>quarot</td><td>MakeQuantizationPlan(QuantizationFormat::kQuarot, 16):
 * signed 4-bit affine blocks. Supply scale, a nonzero transform_size, forward and inverse. See the
 * rotation example.</td></tr>
 * <tr><td>smoothquant</td><td>MakeQuantizationPlan(QuantizationFormat::kSmoothquant, 16): signed
 * 8-bit affine blocks. Supply scale and an explicit forward/inverse rescaling pair; no activation
 * statistics are collected. See the rescaling example.</td></tr>
 * <tr><td>tiled_float</td><td>MakeQuantizationPlan(QuantizationFormat::kTiledFloat, 16, 4): FLOAT
 * cast blocks. Supply permutation to group tiles and optionally change each run.layout.cast_type,
 * for example to TensorProto::FLOAT16. No tiling is inferred from the name.</td></tr>
 * <tr><td>column_major</td><td>MakeQuantizationPlan(QuantizationFormat::kColumnMajor, 16): FLOAT
 * cast storage. Supply a column-major gather permutation; decoding restores the original logical
 * order.</td></tr>
 * </table>
 *
 * @par Basic affine conversion
 * @code{.cpp}
 * auto input = Tensor::FromFloat("weight", {4}, {-1, -0.5f, 0.5f, 1});
 * auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 4);
 * plan.runs[0].blocks[0].scale = 0.25;
 * auto encoded = QuantizeTensor(input, plan);
 * auto restored = DequantizeTensor(encoded);
 * @endcode
 *
 * @par Column grouping and tiled storage
 * For a 4-by-4 row-major matrix whose channels are columns:
 * @code{.cpp}
 * auto channels = MakeQuantizationPlan(QuantizationFormat::kInt8PerChannel, 16, 4);
 * channels.permutation = {0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15};
 * for (auto &block : channels.runs[0].blocks)
 *   block.scale = 1.0 / 127;  // Replace with the corresponding channel's scale.
 * auto columns = MakeQuantizationPlan(QuantizationFormat::kColumnMajor, 16);
 * columns.permutation = channels.permutation;
 * auto tiles = MakeQuantizationPlan(QuantizationFormat::kTiledFloat, 16, 4);
 * tiles.permutation = {0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15};
 * for (auto &run : tiles.runs)
 *   run.layout.cast_type = TensorProto::FLOAT16;
 * @endcode
 *
 * @par Supplied scalar, vector and additive tables
 * The same assignment pattern applies to stq1_0, quip_sharp, aqlm and squeezellm.
 * Use their dimensions from the table above, or explicitly change books, entries,
 * vector_size and bits together. This small AQLM-family example overrides the default
 * dimensions with two synthetic books, four two-component entries each:
 * @code{.cpp}
 * auto additive = MakeQuantizationPlan(QuantizationFormat::kAqlm, 16);
 * auto &layout = additive.runs[0].layout;
 * auto &block = additive.runs[0].blocks[0];
 * layout.books = 2;
 * layout.entries = 4;
 * layout.vector_size = 2;
 * layout.bits = 2;
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
 * auto rotated = MakeQuantizationPlan(QuantizationFormat::kQuarot, 16);
 * rotated.transform_size = 2;
 * rotated.forward = {1,1, 1,-1};
 * rotated.inverse = {0.5,0.5, 0.5,-0.5};
 * rotated.runs[0].blocks[0].scale = 0.5;
 * auto rescaled = MakeQuantizationPlan(QuantizationFormat::kSmoothquant, 16);
 * rescaled.transform_size = 2;
 * rescaled.forward = {2,0, 0,0.5};
 * rescaled.inverse = {0.5,0, 0,2};
 * rescaled.runs[0].blocks[0].scale = 0.125;
 * auto sparse = MakeQuantizationPlan(QuantizationFormat::kSpqr, 16);
 * sparse.runs[0].blocks[0].scale = 0.125;
 * sparse.outliers = {0,15};
 * @endcode
 * Outliers are also allowed with other profiles. They are stored exactly and do not
 * require a special quantization method.
 *
 * @par Python usage
 * Uses the same fields through onnx_light.onnx_core.quantization. Unlike the C++ vector,
 * Python plan.runs and run.blocks are lists of copies: assign modified lists back.
 * plan.run(i) and run.block(j) return copies; set_run and set_block replace stored values.
 * @code{.py}
 * import numpy
 * import onnx_light.onnx.numpy_helper as onh
 * from onnx_light.onnx_core.quantization import (
 *     QuantizationFormat, make_quantization_plan, quantize_tensor_proto, dequantize_tensor_proto,
 * )
 * weights = numpy.array([-1, 0, 1], dtype=numpy.float32)
 * plan = make_quantization_plan(QuantizationFormat.INT4, weights.size)
 * run = plan.run(0)
 * block = run.block(0)
 * block.scale = 0.25
 * run.set_block(0, block)
 * plan.set_run(0, run)
 * encoded = quantize_tensor_proto(onh.from_array(weights), plan)
 * restored = onh.to_array(dequantize_tensor_proto(encoded))
 * @endcode
 * The Python guide docs/howto/quantized_values.rst contains configuration examples for
 * every profile; docs/examples/runtime/plot_quantization_profiles.py executes all of them.
 *
 * @par Validation, persistence and limitations
 * Quantization validates the completed plan, not just the factory name. Missing learned
 * codebooks, required transforms, invalid coverage/indices/parameters, nonfinite values
 * and cast overflow raise std::invalid_argument (ValueError in Python).
 * The factory can therefore return a plan that is not yet ready to encode.
 * The selected profile name is saved after the prefix @c onnx_light.quantization.v1/
 * (DOUBLE reconstruction, the default) or @c onnx_light.quantization.v2/ (FLOAT).
 * This choice is independent of the logical/output dtype and preserved in shared parameters;
 * changing format alone does not update existing block parameters.
 * Encoding rejects invalid QuantizationFormat values. Decoding uses ParseQuantizationFormat
 * to reject unknown profile names. QuantizationFormatName supplies each enum's stable wire name.
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
 * @par Reconstruction precision
 * All portable profiles use the selected contract for affine, codebook and cast blocks.
 * v1 retains DOUBLE operands and block arithmetic, with long-double inverse-transform
 * accumulation rounded to DOUBLE per component. ORT MatMulNBits retains its v1 DOUBLE
 * decoder; this is not a promise about ORT kernel arithmetic.
 * v2 converts each operand to FLOAT, rounds each subtraction, multiplication and addition
 * separately to FLOAT (no fused multiply-add), and accumulates codebooks in book order.
 * Inverse transforms accumulate in increasing row index with separate FLOAT products
 * and sums. Outliers round once to FLOAT; permutations do not change values.
 * Under IEEE round-to-nearest-even with gradual underflow, subnormals are retained;
 * conversions may round tiny operands to zero. Nonfinite operands/results and overflow
 * are rejected, including overflow when converting to the requested output dtype.
 * Output conversion follows reconstruction; FLOAT16/BFLOAT16 convert via FLOAT as in v1.
 * v2 uses float-valued reconstruction/permutation/transform workspaces, even for DOUBLE
 * outputs. Parameter serialization and encoding/calibration arithmetic remain unchanged.
 * GGML formats use the exact native block schema in gguf.h. Their former generic
 * v1/v2 profile names are rejected; codecs without a GGUF counterpart are unchanged.
 *
 * @see MakeQuantizationPlan, QuantizationRun, QuantizationBlockLayout,
 * QuantizationBlockParameters, QuantizationFormats, QuantizeTensor,
 * QuantizeTensorProto, DequantizeTensor, DequantizeTensorProto
 */
struct QuantizationPlan {
  /// Selects the profile; changing it does not reinitialize runs.
  QuantizationFormat format = QuantizationFormat::kInt4;
  /// Covers the tensor with consecutive runs in post-permutation, post-transform order.
  std::vector<QuantizationRun> runs;
  /// Records the required [K,N] shape for ORT profiles; remains empty for portable profiles.
  Shape matrix_shape;
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
  /// Selects reconstruction independently of output dtype: DOUBLE (v1) or FLOAT (v2).
  /// ORT-compatible profiles retain their v1 DOUBLE decoder and reject FLOAT.
  int32_t reconstruction_dtype = TensorProto::DOUBLE;
};

/** Returns all supported formats, including native GGML block storage. */
constexpr auto QuantizationFormats() {
  std::array<QuantizationFormat, static_cast<size_t>(QuantizationFormat::kTq20) + 1> formats{};
  for (size_t i = 0; i < formats.size(); ++i)
    formats[i] = static_cast<QuantizationFormat>(i);
  return formats;
}

/** Returns the stable wire name of a profile; rejects invalid enum values. */
std::string_view QuantizationFormatName(QuantizationFormat format);

/** Parses a stable wire name; rejects unknown names. */
QuantizationFormat ParseQuantizationFormat(std::string_view name);

/**
 * Creates run layouts and block parameters for a catalogue profile.
 * Native GGML formats require already-quantized blocks through the import APIs in gguf.h;
 * floating-point-to-GGML quantization is not implemented and is rejected.
 *
 * See @ref QuantizationPlan for the complete profile-by-profile configuration table,
 * required scales/codebooks/transforms, numerical rules, validation constraints,
 * and C++/Python usage examples. The returned plan may require additional parameters
 * before it can be passed to QuantizeTensor() or QuantizeTensorProto().
 * kMxfp8 and kMxfp6E2m3 require a positive count divisible by 32 and block_size=32 or the default
 * 128, both selecting fixed groups of 32. It selects automatic scales and FLOAT
 * reconstruction; modifying its block parameters, grouping, ordering or transforms
 * is rejected. MakeQuantizationType and shared parameters do not support this profile.
 *
 * @param format Selects one of the enum values returned by QuantizationFormats().
 * @param count Counts the source tensor's scalar elements, not bytes or codebook vectors.
 * @param block_size Sets the maximum elements per block, in [1, UINT32_MAX].
 * Full blocks share a run; a shorter final block has its own run. count=0 yields no runs.
 * @returns A plan with profile defaults, scale=1, no permutation, no transforms and
 * no outliers. Fixed codebooks are populated; learned codebooks must be supplied.
 * @throws std::invalid_argument If the profile is unknown or block_size is invalid.
 * @see QuantizationPlan
 */
QuantizationPlan MakeQuantizationPlan(QuantizationFormat format, uint64_t count,
                                      uint64_t block_size = 128);

/**
 * Creates an ORT MatMulNBits input-packing plan for a logical [K,N] weight matrix.
 *
 * Accepts kOrtMatmulnbitsInt2, kOrtMatmulnbitsInt4 and kOrtMatmulnbitsInt8. K and N
 * must be positive; block_size is a power of two in [16, UINT32_MAX].
 * Creates one shared unsigned affine layout of block_size elements and
 * N * ceil(K/block_size) parameter blocks, in [column, K-block] order.
 * The last block of every column includes padding, unlike portable profiles.
 * Scales default to one and zero points to 2^(bits-1). No calibration is performed.
 *
 * QuantizeTensor/QuantizeTensorProto require a rank-two FLOAT, FLOAT16 or BFLOAT16
 * source. Scales and floating zero points are rounded to that dtype before encoding.
 * Finite negative scales are supported. A zero scale requires an all-zero source block;
 * nonzero scales that round to zero are rejected.
 * All-midpoint zero points are omitted; in-range integer zero points are packed UINT8;
 * otherwise zero points use the source floating dtype. Offsets, codebooks, permutations,
 * transforms and outliers are not supported by these profiles.
 *
 * The encoded structure stores a [bits,block_size] constant, B, scales and optional
 * zero_points tensors in their ORT input layouts. It does not contain an EP-specific
 * prepacked buffer, g_idx or bias. ExportMatMulNBitsInputs extracts ready-to-use inputs.
 */
QuantizationPlan MakeMatMulNBitsPlan(QuantizationFormat format, uint64_t k, uint64_t n,
                                     uint64_t block_size = 128);

/** Stores owned input tensors and attributes for com.microsoft::MatMulNBits, version 1. */
struct MatMulNBitsInputs {
  uint64_t k = 0;
  uint64_t n = 0;
  uint64_t block_size = 0;
  uint32_t bits = 0;
  TensorProto weights;
  TensorProto scales;
  std::optional<TensorProto> zero_points;
};

/**
 * Packs signed codes as portable little-endian bit streams, or unpacks them.
 * The unpacker rejects nonzero padding and an incorrect byte extent.
 * Callers must bound the input and logical code count before calling.
 */
std::string PackQuantizationCodes(std::string_view codes, uint32_t bits);
std::string UnpackQuantizationCodes(std::string_view packed, size_t count, uint32_t bits);

/**
 * Converts signed codes to and from compressed-tensors offset-binary INT32
 * words. The matrix is row-major and axis must be 0 or 1. Unpacking rejects
 * incorrect extents and nonzero padding bits. Callers must bound the input
 * and logical matrix size before calling.
 */
std::string PackCompressedTensorInt32(std::string_view codes, size_t rows, size_t columns,
                                      uint32_t bits, uint32_t axis);
std::string UnpackCompressedTensorInt32(std::string_view packed, size_t rows, size_t columns,
                                        uint32_t bits, uint32_t axis);

/** Rejects an affine block if its actual codes overflow FLOAT reconstruction. */
void ValidateCompressedTensorAffine(std::string_view codes, float scale, int32_t zero_point);

/** Extracts ORT input tensors without dequantizing, after validating the encoded layout. */
MatMulNBitsInputs ExportMatMulNBitsInputs(const EncodedValueProto &value,
                                          const StructTypeCatalogue &catalogue = {});

/** Stores owned NVFP4 WEIGHT inputs, with low-nibble-first E2M1 weights and E4M3FN scales. */
struct Nvfp4Weights {
  Tensor weight_packed;
  Tensor weight_scale;
  float weight_global_scale = 0;
};

/** Stores owned MXFP4 weights with low-nibble-first E2M1 values and E8M0 scale bytes. */
struct Mxfp4Weights {
  Tensor weight_packed;
  Tensor weight_scale;
};

/** Stores owned native E4M3FN weights [rows,cols] and E8M0 bytes [rows,cols/32]. */
struct Mxfp8Weights {
  Tensor weight;
  Tensor weight_scale;
};

/** Stores owned compressed-tensors E2M3 bytes [rows,3*cols/4] and E8M0 [rows,cols/32]. */
struct Mxfp6E2m3Weights {
  Tensor weight_packed;
  Tensor weight_scale;
};

/** Describes the resolved weight-only metadata accepted by the compressed-tensors adapter. */
struct CompressedTensorsQuantizationArgs {
  uint32_t num_bits = 0;
  std::string type;
  bool symmetric = false;
  std::string strategy;
  std::optional<int64_t> group_size;
  std::optional<std::string> scale_dtype;
  std::optional<std::string> zp_dtype;
  bool dynamic = false;
  bool actorder = false;
  bool block_structure = false;
};

/** Carries one module's loaded tensors for the pinned compressed-tensors adapter. */
struct CompressedTensorsWeights {
  std::string compression_format;
  CompressedTensorsQuantizationArgs quantization_args;
  std::unordered_map<std::string, Tensor> tensors;
  std::string upstream_revision = "eb058d630f7344099459b6795b1e3eeffdda4b40";
};

/** Imports one pinned compressed-tensors checkpoint without dequantizing its weights. */
EncodedValueProto ImportCompressedTensorsWeights(const CompressedTensorsWeights &checkpoint,
                                                 size_t max_bytes = size_t{1} << 30);

/** Exports codes and parameters to one explicitly requested compressed-tensors layout. */
CompressedTensorsWeights ExportCompressedTensorsWeights(
    const EncodedValueProto &value, std::string compression_format,
    CompressedTensorsQuantizationArgs quantization_args,
    std::string upstream_revision = "eb058d630f7344099459b6795b1e3eeffdda4b40",
    size_t max_bytes = size_t{1} << 30);

/** Carries a pinned native GPTQModel AWQ checkpoint, distinct from portable kAwq. */
struct AwqWeights {
  std::string variant;
  int64_t in_features = 0, out_features = 0, group_size = 128;
  uint32_t bits = 4;
  int64_t group_axis = 1;
  std::unordered_map<std::string, Tensor> tensors;
  std::string upstream_revision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
};

/** Imports native AWQ tensors without repacking or requantization. */
EncodedValueProto ImportAwqWeights(const AwqWeights &weights, size_t max_bytes = size_t{1} << 30);

/** Exports the validated original AWQ tensors and metadata byte-for-byte. */
AwqWeights ExportAwqWeights(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                            const StructTypeCatalogue &catalogue = {});

/** Unpacks native AWQ unsigned four-bit codes into UINT8 [N,K]. */
Tensor UnpackAwqCodes(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                      const StructTypeCatalogue &catalogue = {});

/** Reconstructs FLOAT [N,K] weights without simulating accelerator arithmetic. */
Tensor ReconstructAwqWeights(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                             const StructTypeCatalogue &catalogue = {});

/**
 * Carries one GPTQModel EXL3 module, distinct from the portable kExl3 profile.
 *
 * Pins GPTQModel's canonical checkpoint contract: INT16 trellis [K/16,N/16,16*bits],
 * FLOAT16 suh [K], svh [N], and optional scalar INT32 mcg or mul1 codebook marker.
 * K and N are positive multiples of 128; bits is an integer from 1 through 8.
 * The pinned producer floors fractional per-module bit settings before packing; the
 * physical rate is inferred from trellis shape, not configuration prose. Different
 * modules may have different rates. Other revisions, custom codebooks, legacy packed
 * signs, transforms and side tensors are rejected.
 */
struct Exl3Weights {
  std::unordered_map<std::string, Tensor> tensors;
  std::string upstream_revision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
};

/** Imports EXL3 streams and side tensors losslessly into onnx_light.exl3_weights.v1. */
EncodedValueProto ImportExl3Weights(const Exl3Weights &weights, size_t max_bytes = size_t{1} << 30);

/** Exports the validated native checkpoint state without scalar recoding. */
Exl3Weights ExportExl3Weights(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                              const StructTypeCatalogue &catalogue = {});

/** Unpacks cyclic INT16 state bit patterns in [K/16,N/16,256] tensor-core lane order. */
Tensor UnpackExl3States(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                        const StructTypeCatalogue &catalogue = {});

/**
 * Reconstructs FLOAT [K,N] weights using binary16 codebook rounding and block-128
 * normalized FLOAT butterflies, with suh after the input transform and svh after
 * the output transform. FLOAT16 reconstruction additionally rounds each transform
 * and scale multiplication to binary16, following native materialization boundaries.
 * Uses deterministic ascending butterflies, not backend-dependent dense GEMM reductions;
 * neither mode promises bit-identical accelerator inference or materialization.
 * Bounds both payload and logical FLOAT workspace by max_bytes.
 * Does not implement quantization, training, bias or accelerated matrix multiplication.
 */
Tensor ReconstructExl3Weights(const EncodedValueProto &value, size_t max_bytes = size_t{1} << 30,
                              const StructTypeCatalogue &catalogue = {},
                              int32_t reconstruction_dtype = TensorProto::FLOAT);

/**
 * Quantizes a finite FLOAT matrix using complete groups of 32 columns along axis 1.
 *
 * Implements compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40's E2M3
 * contract, distinct from the unchanged portable kMxfp6/kFp6Llm E3M2 profiles.
 * Each code uses sign bit 5, exponent bits 4:3 (bias 1), mantissa bits 2:0.
 * Exponent zero decodes as mantissa/8; other magnitudes are (1+mantissa/8)*2^(exponent-1).
 * Four codes form c0|(c1<<6)|(c2<<12)|(c3<<18), stored as three little-endian bytes.
 * For group max_abs m, rounds its FP32 bits by adding 2^19 then masking with
 * 0x7f800000; subtracts 127+2 from the resulting exponent and clamps to [-127,125].
 * This rounds upward at mantissa 1.9375, not a generic nearest-power-of-two threshold.
 * Zero blocks use scale byte 0 (2^-127); bytes 0--254 decode as 2^(byte-127).
 * Byte 255 is rejected. The upper exponent clamp is an explicit finite-reconstruction
 * extension to upstream; imports accept all finite E8M0 scales only when reconstruction
 * stays finite. Divides in FP32, rounds E2M3 to nearest even and saturates to +/-7.5,
 * preserving signed zero. Reconstruction multiplies decoded E2M3 by its scale in FP32.
 * Rejects nonfinite sources, incomplete groups, empty shapes, other dtypes and malformed
 * payloads/schemas. No padding, transforms, outliers or shared parameters are supported.
 * Stores onnx_light.mxfp6_e2m3_weights.v1 with packed weights followed by E8M0 bytes.
 * MakeQuantizationPlan(kMxfp6E2m3,count) selects the same automatic-scale codec.
 */
EncodedValueProto QuantizeMxfp6E2m3Weights(const Tensor &tensor);

/** Extracts owned packed E2M3 weights and scales after validating the exact schema. */
Mxfp6E2m3Weights ExportMxfp6E2m3Weights(const EncodedValueProto &value,
                                        const StructTypeCatalogue &catalogue = {});

/** Imports packed E2M3 weights losslessly, rejecting invalid scales and reconstruction. */
EncodedValueProto ImportMxfp6E2m3Weights(const Mxfp6E2m3Weights &weights);

/**
 * Quantizes a finite FLOAT matrix with positive dimensions and cols divisible by 32.
 *
 * Uses compressed-tensors eb058d630f7344099459b6795b1e3eeffdda4b40's num_bits=8 rule:
 * ((FP32 max_abs bits + 2^21) & 0x7f800000)'s exponent minus 127 minus 8.
 * Zero blocks use scale one. Explicit extensions clamp exponents to [-127,119],
 * guaranteeing finite FP32 reconstruction even at the saturated E4M3FN maximum 448.
 * E4M3FN encoding rounds to nearest even and saturates; reconstruction multiplies in FP32.
 * Stores an exact versioned schema, native FP8 bytes, then row-major E8M0 bytes.
 * MakeQuantizationPlan(kMxfp8,count) selects this codec with automatic scales; only
 * unaltered fixed-32 plans are accepted, without transforms, permutations or outliers.
 */
EncodedValueProto QuantizeMxfp8Weights(const Tensor &tensor);

/** Extracts owned MXFP8 tensors after validating the exact schema and reconstruction. */
Mxfp8Weights ExportMxfp8Weights(const EncodedValueProto &value,
                                const StructTypeCatalogue &catalogue = {});

/** Imports native MXFP8 tensors losslessly, rejecting NaNs and nonfinite reconstruction. */
EncodedValueProto ImportMxfp8Weights(const Mxfp8Weights &weights);

/**
 * Quantizes FLOAT [rows,cols] weights using groups of 16 along axis 1.
 *
 * Requires positive dimensions, cols divisible by 16, finite weights, positive finite global
 * scale and FLOAT or FLOAT8E4M3FN local scales shaped [rows,cols/16].
 * FLOAT local scales round to E4M3FN nearest-even with finite saturation at 448.
 * This follows calibration's rounded-local contract, not arbitrary compressor calls that
 * normalize with unrounded FLOAT scales before converting their storage to FP8.
 * Nonpositive/nonfinite local scales, including positive values rounding to zero, are rejected.
 * Each effective scale is separately rounded FP32(local_fp8/global); zero/nonfinite effective
 * scales and nonfinite reconstructed outputs are rejected, including for all-zero groups.
 * Normalization rounds to FP32 before clamping to [-6,6] and E2M1 nearest-even quantization.
 * Negative nonzero normalized values rounding to E2M1 zero retain their sign.
 * Input -0 and values whose FP32 normalization underflows to -0 become positive zero.
 * The distinct v1 WEIGHT schema preserves local FP8 bytes and the FP32 global scale exactly.
 */
EncodedValueProto QuantizeNvfp4Weights(const Tensor &tensor, const Tensor &local_scales,
                                       float global_scale);

/** Extracts owned NVFP4 inputs after validating the complete schema and reconstruction. */
Nvfp4Weights ExportNvfp4Weights(const EncodedValueProto &value,
                                const StructTypeCatalogue &catalogue = {});

/** Imports NVFP4 inputs losslessly with logical FLOAT shape [rows,packed_cols*2]. */
EncodedValueProto ImportNvfp4Weights(const Nvfp4Weights &weights);

/**
 * Quantizes a finite FLOAT [rows,cols] matrix using groups of 32 along axis 1.
 *
 * Requires positive dimensions and cols divisible by 32. For each nonzero group maximum m,
 * starts with e = floor(log2(m)) and increments e when m >= 1.75 * 2^e (the E2M1
 * mantissa threshold). Divides the resulting power of two by four by subtracting 2 from e,
 * then clamps the scale exponent to [-127,125] to keep FLOAT reconstruction finite.
 * Stores the exponent with E8M0 bias 127; zero groups use scale one.
 * E2M1 quantization saturates to [-6,6] and rounds midpoint ties to even.
 * Nonfinite inputs and incomplete groups are
 * rejected. The versioned payload stores low-nibble-first values followed by row-major E8M0 scale
 * bytes; it does not emit a hardware-swizzled kernel layout.
 */
EncodedValueProto QuantizeMxfp4Weights(const Tensor &tensor);

/** Extracts owned MXFP4 inputs after validating the complete schema and reconstruction. */
Mxfp4Weights ExportMxfp4Weights(const EncodedValueProto &value,
                                const StructTypeCatalogue &catalogue = {});

/** Imports MXFP4 inputs losslessly with logical FLOAT shape [rows,packed_cols*2]. */
EncodedValueProto ImportMxfp4Weights(const Mxfp4Weights &weights);

/** Quantizes a finite floating-point tensor into an owned, self-describing encoded value. */
RuntimeValue QuantizeTensor(const Tensor &tensor, const QuantizationPlan &plan);

/** Supplies optional numerical inputs to automatic quantization; tensors are borrowed. */
struct QuantizationParameters {
  const Tensor *scales = nullptr;
  const Tensor *zero_points = nullptr;
  const Tensor *offsets = nullptr;
  const Tensor *codebooks = nullptr;
  const Tensor *permutation = nullptr;
  const Tensor *forward = nullptr;
  const Tensor *inverse = nullptr;
  const Tensor *outliers = nullptr;
};

/** Owns model-local immutable fixed numerical parameters, independently of storage type IDs. */
class QuantizationParameterCatalogue {
public:
  struct MemoryEstimate {
    /// Bounds retained plans, common payloads and range buffers, including capacity slack.
    size_t retained_bytes = 0;
    /// Bounds peak numerical construction work beyond retained storage.
    size_t temporary_bytes = 0;
  };
  struct Entry {
    QuantizationPlan plan;
    StructTypeProto storage_type, local_type;
    TypeProto logical_type;
    std::vector<std::pair<size_t, size_t>> local_ranges;
    std::vector<uint8_t> common;
    size_t full_size = 0;
  };
  /** Bounds numerical storage from compact descriptors, without expanding block arrays.
   * Excludes proto metadata and parser allocations. Numerical tensors may remain external.
   */
  static MemoryEstimate EstimateMemory(const ModelProto &model);
  static std::shared_ptr<const QuantizationParameterCatalogue> Build(const ModelProto &model);
  const Entry &Get(const std::string &name) const;
  /** Validates a compact value without materializing it and returns its shared parameters. */
  const Entry &Validate(const EncodedValueProto &value,
                        const StructTypeCatalogue &catalogue = {}) const;

private:
  std::unordered_map<std::string, Entry> entries_;
};

/** Quantizes with fixed shared parameters and retains their immutable catalogue. */
RuntimeValue QuantizeTensorShared(const Tensor &tensor, const StructTypeProto &type,
                                  const std::string &parameter_ref,
                                  std::shared_ptr<const QuantizationParameterCatalogue> parameters,
                                  const StructTypeCatalogue &catalogue = {});

/** Returns an independent self-contained encoded message, resolving shared parameters. */
EncodedValueProto
MaterializeQuantizedValue(const EncodedValueProto &value,
                          const QuantizationParameterCatalogue *parameters = nullptr,
                          const StructTypeCatalogue &catalogue = {});
EncodedValueProto MaterializeQuantizedValue(const RuntimeValue &value,
                                            const StructTypeCatalogue &catalogue = {});

/** Returns the portable plan's storage type without requiring trained numerical parameters. */
StructTypeProto MakeQuantizationType(const QuantizationPlan &plan);

/** Returns the truthful compact schema for codes and local outlier values. */
StructTypeProto MakeSharedQuantizationType(const StructTypeProto &type);

/**
 * Quantizes into a portable storage type, calibrating omitted scales from the source.
 * Scalar parameters broadcast; otherwise scales, zero points and offsets have one entry
 * per block. Codebooks are concatenated in run/block order. Learned or vector codebooks
 * and transformations require explicit inputs; no training algorithm is implied.
 * ORT layouts calibrate per column/K-block and must match the resulting input-packing schema.
 */
RuntimeValue QuantizeTensor(const Tensor &tensor, const StructTypeProto &type,
                            const QuantizationParameters &parameters = {},
                            const StructTypeCatalogue &catalogue = {});

/** Dequantizes a supported encoded runtime value to its declared floating-point tensor type. */
Tensor DequantizeTensor(const RuntimeValue &value, const StructTypeCatalogue &catalogue = {},
                        RawBufferAllocator *allocator = nullptr);

/** Dequantizes a message by const reference without copying its encoded payload. */
Tensor DequantizeTensor(const EncodedValueProto &value, const StructTypeCatalogue &catalogue = {},
                        RawBufferAllocator *allocator = nullptr,
                        int32_t output_dtype = TensorProto::UNDEFINED);

/** Quantizes a loaded TensorProto without changing the source message. */
EncodedValueProto QuantizeTensorProto(const TensorProto &tensor, const QuantizationPlan &plan);

/** Dequantizes a supported encoded message into an owned TensorProto. */
TensorProto DequantizeTensorProto(const EncodedValueProto &value,
                                  const StructTypeCatalogue &catalogue = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
