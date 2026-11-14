// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include <span>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Describes a fixed, standard-domain DequantizePacked wire profile. */
struct PackedQuantizationProfile {
  std::string_view format;
  size_t block_size;
  size_t block_bytes;
  int ggml_type;
};

/** Returns the 27 supported fixed wire profiles. */
std::span<const PackedQuantizationProfile> PackedQuantizationProfiles();

/** Resolves a case-sensitive format, rejecting unsupported profiles. */
const PackedQuantizationProfile &PackedQuantizationLayout(std::string_view format);

/** Validates dimensions, alignment, dtype and overflow; returns the packed byte count. */
size_t ValidatePackedQuantizationShape(const PackedQuantizationProfile &profile, const Shape &shape,
                                       int64_t output_dtype);

/** Decodes a validated raw UINT8 vector, preserving nonfinite FP32 arithmetic. */
Tensor DequantizePackedTensor(const Tensor &input, std::string_view format, const Shape &shape,
                              int64_t output_dtype, RawBufferAllocator *allocator = nullptr);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
