// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/gguf.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

struct ValidatedLayout {
  const GgufBlockLayout &layout;
  size_t blocks;
  size_t decoded_bytes;
};

size_t CheckedMultiply(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "GGUF size overflow.");
  return a * b;
}

ValidatedLayout ValidateLayout(std::string_view data, const Shape &shape, std::string_view subtype,
                               std::string_view version, std::string_view revision,
                               size_t max_bytes) {
  EXT_ENFORCE_INVALID(version == kGgufLayoutVersion, "Unsupported GGUF layout version.");
  EXT_ENFORCE_INVALID(revision == kGgufUpstreamRevision, "Unsupported GGUF upstream revision.");
  const auto &layout = GgufLayout(subtype);
  EXT_ENFORCE_INVALID(!shape.empty(), "GGUF requires a non-scalar shape.");
  for (const auto dim : shape)
    EXT_ENFORCE_INVALID(dim > 0, "GGUF requires positive dimensions.");
  EXT_ENFORCE_INVALID(static_cast<uint64_t>(shape[shape.size() - 1]) % layout.block_size == 0,
                      "GGUF rows must contain complete blocks.");
  size_t count = 1;
  for (const auto dim : shape) {
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max(),
                        "GGUF dimension exceeds size_t.");
    count = CheckedMultiply(count, static_cast<size_t>(dim));
  }
  const size_t blocks = count / layout.block_size;
  const size_t packed_bytes = CheckedMultiply(blocks, layout.type_size);
  const size_t decoded_bytes = CheckedMultiply(count, sizeof(float));
  EXT_ENFORCE_INVALID(packed_bytes <= max_bytes && decoded_bytes <= max_bytes,
                      "GGUF packed or decoded byte limit exceeded.");
  EXT_ENFORCE_INVALID(count <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                      "GGUF tensor element count exceeds int64.");
  EXT_ENFORCE_INVALID(data.size() == packed_bytes, "GGUF payload byte count mismatch.");
  return {layout, blocks, decoded_bytes};
}

float Half(const uint8_t *p) {
  const auto bits = static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
  const float value = Float16BitsToFloat(bits);
  EXT_ENFORCE_INVALID(std::isfinite(value), "GGUF contains a nonfinite FLOAT16 parameter.");
  return value;
}

int SignedByte(uint8_t value) { return value < 128 ? value : static_cast<int>(value) - 256; }

struct BlockParameters {
  std::array<float, 16> scales{};
  std::array<float, 8> minima{};
  size_t scale_count = 0;
  size_t minimum_count = 0;
};

BlockParameters Parameters(const uint8_t *p, std::string_view subtype) {
  BlockParameters result;
  if (subtype == "Q6_K") {
    const float d = Half(p + 208);
    result.scale_count = 16;
    for (size_t i = 0; i < 16; ++i)
      result.scales[i] = d * static_cast<float>(SignedByte(p[192 + i]));
  } else {
    const float d = Half(p);
    if (subtype == "Q4_0" || subtype == "Q8_0") {
      result.scale_count = 1;
      result.scales[0] = d;
    } else {
      const float dmin = Half(p + 2);
      const uint8_t *sc = p + 4;
      result.scale_count = result.minimum_count = 8;
      for (size_t i = 0; i < 8; ++i) {
        const int scale = i < 4 ? sc[i] & 63 : (sc[i + 4] & 15) | ((sc[i - 4] >> 6) << 4);
        const int minimum = i < 4 ? sc[i + 4] & 63 : (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
        result.scales[i] = d * static_cast<float>(scale);
        result.minima[i] = dmin * static_cast<float>(minimum);
      }
    }
  }
  return result;
}

// Matches the pinned ggml/src/ggml-quants.c and gguf-py/gguf/quants.py FLOAT32 reconstruction.
void DecodeBlock(const uint8_t *p, const GgufBlockLayout &layout, float *output) {
  const auto parameters = Parameters(p, layout.subtype);
  for (size_t i = 0; i < layout.block_size; ++i) {
    float value;
    if (layout.subtype == "Q4_0") {
      const int code = ((p[2 + i % 16] >> (4 * (i / 16))) & 15) - 8;
      value = parameters.scales[0] * static_cast<float>(code);
    } else if (layout.subtype == "Q8_0") {
      value = parameters.scales[0] * static_cast<float>(SignedByte(p[2 + i]));
    } else if (layout.subtype == "Q6_K") {
      const size_t half = i / 128, quarter = (i % 128) / 32, lane = i % 32;
      const uint8_t low = p[half * 64 + (quarter % 2) * 32 + lane];
      const uint8_t high = p[128 + half * 32 + lane];
      const int code = ((low >> (4 * (quarter / 2))) & 15) | (((high >> (2 * quarter)) & 3) << 4);
      value = parameters.scales[i / 16] * static_cast<float>(code - 32);
    } else {
      const bool q5 = layout.subtype == "Q5_K";
      const size_t qs = q5 ? 48 : 16;
      int code = (p[qs + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
      if (q5)
        code |= ((p[16 + i % 32] >> (i / 32)) & 1) << 4;
      // Separate FLOAT32 rounding prevents contraction into a fused multiply-subtract.
      volatile float product = parameters.scales[i / 32] * static_cast<float>(code);
      value = product - parameters.minima[i / 32];
    }
    EXT_ENFORCE_INVALID(std::isfinite(value), "GGUF reconstructs a nonfinite FLOAT32 value.");
    if (output != nullptr)
      output[i] = value;
  }
}

void ValidateBlocks(std::string_view data, const ValidatedLayout &validated) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(data.data());
  for (size_t i = 0; i < validated.blocks; ++i)
    DecodeBlock(bytes + i * validated.layout.type_size, validated.layout, nullptr);
}

ValidatedLayout Validate(const GgufTensorPayload &payload, size_t max_bytes) {
  return ValidateLayout(payload.data, payload.shape, payload.subtype, payload.layout_version,
                        payload.upstream_revision, max_bytes);
}

Shape PackedMatrixShape(const Shape &shape, const GgufBlockLayout &layout) {
  EXT_ENFORCE_INVALID(shape.size() == 2, "GPTQModel GGUF requires a rank-two logical shape.");
  const size_t row_bytes =
      CheckedMultiply(static_cast<size_t>(shape[1]) / layout.block_size, layout.type_size);
  EXT_ENFORCE_INVALID(row_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                      "GPTQModel GGUF packed row exceeds int64.");
  return {shape[0], static_cast<int64_t>(row_bytes)};
}

} // namespace

const std::vector<GgufBlockLayout> &GgufBlockLayouts() {
  static const std::vector<GgufBlockLayout> layouts{{"Q4_0", 32, 18},
                                                    {"Q8_0", 32, 34},
                                                    {"Q4_K", 256, 144},
                                                    {"Q5_K", 256, 176},
                                                    {"Q6_K", 256, 210}};
  return layouts;
}

const GgufBlockLayout &GgufLayout(std::string_view subtype) {
  for (const auto &layout : GgufBlockLayouts())
    if (layout.subtype == subtype)
      return layout;
  throw std::invalid_argument("Unsupported canonical GGUF subtype: " + std::string(subtype) + ".");
}

GgufTensorPayload ImportGgufTensorPayload(std::string_view data, Shape shape, std::string subtype,
                                          std::string layout_version, std::string upstream_revision,
                                          size_t max_bytes) {
  const auto validated =
      ValidateLayout(data, shape, subtype, layout_version, upstream_revision, max_bytes);
  ValidateBlocks(data, validated);
  return {std::move(subtype), std::move(layout_version), std::move(upstream_revision),
          std::move(shape), std::string(data)};
}

std::string ExportGgufTensorPayload(const GgufTensorPayload &payload, size_t max_bytes) {
  const auto validated = Validate(payload, max_bytes);
  ValidateBlocks(payload.data, validated);
  return payload.data;
}

Tensor DequantizeGgufTensorPayload(const GgufTensorPayload &payload, size_t max_bytes) {
  const auto validated = Validate(payload, max_bytes);
  Tensor output("", TensorProto::FLOAT, payload.shape,
                std::vector<uint8_t>(validated.decoded_bytes));
  std::array<float, 256> block{};
  const auto *bytes = reinterpret_cast<const uint8_t *>(payload.data.data());
  for (size_t i = 0; i < validated.blocks; ++i) {
    DecodeBlock(bytes + i * validated.layout.type_size, validated.layout, block.data());
    std::memcpy(output.mutable_bytes() + i * validated.layout.block_size * sizeof(float),
                block.data(), validated.layout.block_size * sizeof(float));
  }
  return output;
}

GgufBlockParameters ReadGgufBlockParameters(const GgufTensorPayload &payload, size_t block_index,
                                            size_t max_bytes) {
  const auto validated = Validate(payload, max_bytes);
  EXT_ENFORCE_INVALID(block_index < validated.blocks, "GGUF block index out of range.");
  ValidateBlocks(payload.data, validated);
  const auto parameters = Parameters(reinterpret_cast<const uint8_t *>(payload.data.data()) +
                                         block_index * validated.layout.type_size,
                                     validated.layout.subtype);
  return {{parameters.scales.begin(), parameters.scales.begin() + parameters.scale_count},
          {parameters.minima.begin(), parameters.minima.begin() + parameters.minimum_count}};
}

GgufTensorPayload ImportGptqModelGgufWeights(const GptqModelGgufWeights &weights,
                                             size_t max_bytes) {
  EXT_ENFORCE_INVALID(weights.shape.size() == 2,
                      "GPTQModel GGUF requires a rank-two logical shape.");
  EXT_ENFORCE_INVALID(weights.qweight.data_type == TensorProto::UINT8,
                      "GPTQModel GGUF qweight must be UINT8.");
  EXT_ENFORCE_INVALID(weights.qweight.shape.size() == 2,
                      "GPTQModel GGUF qweight must be rank two.");
  for (const auto dim : weights.qweight.shape)
    EXT_ENFORCE_INVALID(dim > 0, "GPTQModel GGUF qweight requires positive dimensions.");
  EXT_ENFORCE_INVALID(weights.qweight.bytes() != nullptr,
                      "GPTQModel GGUF qweight has no byte storage.");
  const std::string_view data(reinterpret_cast<const char *>(weights.qweight.bytes()),
                              weights.qweight.size_bytes());
  const auto validated =
      ValidateLayout(data, weights.shape, weights.subtype, weights.layout_version,
                     weights.upstream_revision, max_bytes);
  EXT_ENFORCE_INVALID(weights.qweight.shape == PackedMatrixShape(weights.shape, validated.layout),
                      "GPTQModel GGUF packed matrix shape mismatch; row padding is unsupported.");
  ValidateBlocks(data, validated);
  return {weights.subtype, weights.layout_version, weights.upstream_revision, weights.shape,
          std::string(data)};
}

GptqModelGgufWeights ExportGptqModelGgufWeights(const GgufTensorPayload &payload,
                                                size_t max_bytes) {
  const auto validated = Validate(payload, max_bytes);
  const auto packed_shape = PackedMatrixShape(payload.shape, validated.layout);
  ValidateBlocks(payload.data, validated);
  Tensor qweight("qweight", TensorProto::UINT8, packed_shape,
                 std::vector<uint8_t>(payload.data.size()));
  std::memcpy(qweight.mutable_bytes(), payload.data.data(), payload.data.size());
  return {std::move(qweight), payload.shape, payload.subtype, payload.layout_version,
          payload.upstream_revision};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
