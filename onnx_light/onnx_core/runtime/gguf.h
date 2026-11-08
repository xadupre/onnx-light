// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

inline constexpr char kGgufLayoutVersion[] = "ggml-blocks-v1";
inline constexpr char kGgufUpstreamRevision[] = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
inline constexpr char kGgufReferenceRevision[] = "0c1e57098bba43ac29e6e3b677cdceebdd22334f";

inline constexpr char kGgufTypePrefix[] = "onnx_light.gguf.ggml-blocks-v1/";

struct GgufBlockLayout {
  std::string subtype;
  size_t block_size;
  size_t type_size;
};

/** Stores effective FLOAT32 subgroup multipliers and subtractive minima in logical order.
 * Q4_0 and Q8_0 have one scale; Q4_K and Q5_K have eight scales and minima;
 * Q6_K has sixteen signed scales. Symmetric formats have no minima.
 */
struct GgufBlockParameters {
  std::vector<float> scales;
  std::vector<float> minima;
};

/** Stores an extracted GPTQModel GGUF safetensors qweight and explicit layout metadata.
 * Supports only nonempty matrices with block-aligned logical input features.
 * qweight is UINT8 [out_features, in_features / block_size * type_size].
 * This stage rejects upstream row padding and partial original dimensions;
 * it does not load an entire checkpoint or infer subtype metadata.
 */
struct GptqModelGgufWeights {
  Tensor qweight;
  Shape shape;
  std::string subtype;
  std::string layout_version = kGgufLayoutVersion;
  std::string upstream_revision = kGgufUpstreamRevision;
};

/** Returns the supported, exact block layouts. */
const std::vector<GgufBlockLayout> &GgufBlockLayouts();
/** Returns a canonical layout; rejects aliases and unsupported subtypes. */
const GgufBlockLayout &GgufLayout(std::string_view subtype);

/** Stores exact GGML bytes in EncodedValueProto, without repacking or requantization.
 * Records the logical FLOAT shape and native subtype in the standard structured type.
 * Bounds both packed and FLOAT32 decoded sizes.
 */
EncodedValueProto ImportGgufTensorPayload(std::string_view data, Shape shape, std::string subtype,
                                          std::string layout_version = kGgufLayoutVersion,
                                          std::string upstream_revision = kGgufUpstreamRevision,
                                          size_t max_bytes = size_t{1} << 30);
/** Validates and returns unchanged payload bytes, with the same size limits as import. */
std::string ExportGgufTensorPayload(const EncodedValueProto &payload,
                                    size_t max_bytes = size_t{1} << 30,
                                    const StructTypeCatalogue &catalogue = {});
/** Validates and reconstructs an owning FLOAT32 tensor in logical row-major order. */
Tensor DequantizeGgufTensorPayload(const EncodedValueProto &payload,
                                   size_t max_bytes = size_t{1} << 30,
                                   const StructTypeCatalogue &catalogue = {},
                                   RawBufferAllocator *allocator = nullptr);
/** Validates the entire payload and returns one block's effective numerical parameters. */
GgufBlockParameters ReadGgufBlockParameters(const EncodedValueProto &payload, size_t block_index,
                                            size_t max_bytes = size_t{1} << 30,
                                            const StructTypeCatalogue &catalogue = {});

/** Imports a block-aligned UINT8 checkpoint matrix as a native EncodedValueProto.
 * Preserves qweight bytes and is decoded by the standard DequantizeTensor APIs.
 */
EncodedValueProto ImportGptqModelGgufWeights(const GptqModelGgufWeights &weights,
                                             size_t max_bytes = size_t{1} << 30);
/** Validates and exports unchanged blocks as an owning UINT8 checkpoint matrix. */
GptqModelGgufWeights ExportGptqModelGgufWeights(const EncodedValueProto &payload,
                                                size_t max_bytes = size_t{1} << 30,
                                                const StructTypeCatalogue &catalogue = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
