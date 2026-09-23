// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"

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
  uint64_t count = 0;
  QuantizationMethod method = QuantizationMethod::kAffine;
  uint32_t bits = 4;
  bool signed_codes = true;
  double scale = 1;
  double zero_point = 0;
  double offset = 0;
  uint32_t books = 1;
  uint32_t entries = 0;
  uint32_t vector_size = 1;
  std::vector<double> codebook;
  bool base3 = false;
  int32_t cast_type = TensorProto::FLOAT16;
};

/**
 * Describes a portable onnx-light quantization, not a vendor packing ABI.
 *
 * Removes selected outliers, gathers the permutation, applies the forward
 * matrix to consecutive row vectors, then encodes consecutive blocks.
 * Decoding reverses these steps and restores the outliers exactly.
 * Matrices are row-major, square and explicitly supplied as an inverse pair.
 * An empty permutation/matrix selects the identity. No calibration is inferred.
 */
struct QuantizationPlan {
  std::string format;
  std::vector<QuantizationBlock> blocks;
  std::vector<int64_t> permutation;
  uint32_t transform_size = 0;
  std::vector<double> forward;
  std::vector<double> inverse;
  std::vector<int64_t> outliers;
};

/** Returns the names of the portable catalogue profiles. */
std::vector<std::string> QuantizationFormats();

/**
 * Creates block defaults for a catalogue profile.
 *
 * Scales default to one and must be supplied by the caller when required.
 * Learned/vector codebooks and rotations are never invented: quantization
 * rejects a plan until these required parameters have been supplied.
 * Mixed precision and tiling use explicit block overrides and permutations.
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
