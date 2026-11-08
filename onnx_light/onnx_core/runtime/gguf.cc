// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/gguf.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

#include "onnx_core/runtime/gguf_iq1s_grid.inc"

constexpr std::array<int, 16> kIq4NlValues{-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};

struct ValidatedLayout {
  const GgufBlockLayout &layout;
  QuantizationFormat format;
  size_t blocks;
  size_t decoded_bytes;
  Shape shape;
};

std::string_view Bytes(const EncodedValueProto &value) {
  return {reinterpret_cast<const char *>(value.raw_data().data()), value.raw_data().size()};
}

size_t CheckedMultiply(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "GGUF size overflow.");
  return a * b;
}

ValidatedLayout ValidateLayout(size_t data_size, const Shape &shape, std::string_view subtype,
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
  EXT_ENFORCE_INVALID(data_size == packed_bytes, "GGUF payload byte count mismatch.");
  return {layout, ParseQuantizationFormat(subtype), blocks, decoded_bytes, shape};
}

float Half(const uint8_t *p) {
  const auto bits = static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
  const float value = Float16BitsToFloat(bits);
  EXT_ENFORCE_INVALID(std::isfinite(value), "GGUF contains a nonfinite FLOAT16 parameter.");
  return value;
}

int SignedByte(uint8_t value) { return value < 128 ? value : static_cast<int>(value) - 256; }

uint16_t LittleEndian16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

struct BlockParameters {
  std::array<float, 16> scales{};
  std::array<float, 16> minima{};
  size_t scale_count = 0;
  size_t minimum_count = 0;
};

BlockParameters Parameters(const uint8_t *p, QuantizationFormat format) {
  BlockParameters result;
  if (format == QuantizationFormat::kQ2K) {
    const float d = Half(p + 80), dmin = Half(p + 82);
    result.scale_count = result.minimum_count = 16;
    for (size_t i = 0; i < 16; ++i) {
      result.scales[i] = d * static_cast<float>(p[i] & 15);
      result.minima[i] = dmin * static_cast<float>(p[i] >> 4);
    }
  } else if (format == QuantizationFormat::kQ3K) {
    const float d = Half(p + 108);
    result.scale_count = 16;
    for (size_t i = 0; i < 16; ++i) {
      const int low = (p[96 + i % 8] >> (4 * (i / 8))) & 15;
      const int high = (p[104 + i % 4] >> (2 * (i / 4))) & 3;
      result.scales[i] = d * static_cast<float>((low | (high << 4)) - 32);
    }
  } else if (format == QuantizationFormat::kIq1S) {
    const float d = Half(p);
    result.scale_count = 8;
    for (size_t i = 0; i < 8; ++i)
      result.scales[i] =
          d * static_cast<float>(2 * ((LittleEndian16(p + 34 + 2 * i) >> 12) & 7) + 1);
  } else if (format == QuantizationFormat::kTq10 || format == QuantizationFormat::kTq20) {
    result.scale_count = 1;
    result.scales[0] = Half(p + (format == QuantizationFormat::kTq10 ? 52 : 64));
  } else if (format == QuantizationFormat::kQ6K) {
    const float d = Half(p + 208);
    result.scale_count = 16;
    for (size_t i = 0; i < 16; ++i)
      result.scales[i] = d * static_cast<float>(SignedByte(p[192 + i]));
  } else {
    const float d = Half(p);
    if (format == QuantizationFormat::kQ40 || format == QuantizationFormat::kQ80 ||
        format == QuantizationFormat::kIq4Nl) {
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
void DecodeBlock(const uint8_t *p, const GgufBlockLayout &layout, QuantizationFormat format,
                 float *output) {
  const auto parameters = Parameters(p, format);
  for (size_t i = 0; i < layout.block_size; ++i) {
    float value;
    if (format == QuantizationFormat::kQ40) {
      const int code = ((p[2 + i % 16] >> (4 * (i / 16))) & 15) - 8;
      value = parameters.scales[0] * static_cast<float>(code);
    } else if (format == QuantizationFormat::kQ80) {
      value = parameters.scales[0] * static_cast<float>(SignedByte(p[2 + i]));
    } else if (format == QuantizationFormat::kQ2K || format == QuantizationFormat::kQ3K) {
      const size_t base = format == QuantizationFormat::kQ2K ? 16 : 32;
      int code = (p[base + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
      if (format == QuantizationFormat::kQ3K)
        code -= ((p[i % 32] >> (i / 32)) & 1) ? 0 : 4;
      volatile float product = parameters.scales[i / 16] * static_cast<float>(code);
      value = format == QuantizationFormat::kQ2K ? product - parameters.minima[i / 16] : product;
    } else if (format == QuantizationFormat::kIq4Nl) {
      const int code = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
      value = parameters.scales[0] * static_cast<float>(kIq4NlValues[code]);
    } else if (format == QuantizationFormat::kIq1S) {
      const uint16_t high = LittleEndian16(p + 34 + 2 * (i / 32));
      const size_t index = p[2 + i / 8] | (((high >> (3 * ((i / 8) % 4))) & 7) << 8);
      const int code = ((kIq1SGrid[index] >> (2 * (i % 8))) & 3) - 1;
      const float delta = (high & 0x8000) ? -0.125f : 0.125f;
      value = parameters.scales[i / 32] * (static_cast<float>(code) + delta);
    } else if (format == QuantizationFormat::kTq10) {
      constexpr std::array<int, 5> powers{1, 3, 9, 27, 81};
      const size_t index = i < 160 ? i % 32 : i < 240 ? 32 + (i - 160) % 16 : 48 + (i - 240) % 4;
      const size_t digit = i < 160 ? i / 32 : i < 240 ? (i - 160) / 16 : (i - 240) / 4;
      const auto packed = static_cast<uint8_t>(p[index] * powers[digit]);
      const int code = static_cast<int>((static_cast<unsigned>(packed) * 3) >> 8) - 1;
      value = parameters.scales[0] * static_cast<float>(code);
    } else if (format == QuantizationFormat::kTq20) {
      const int code = ((p[(i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3) - 1;
      value = parameters.scales[0] * static_cast<float>(code);
    } else if (format == QuantizationFormat::kQ6K) {
      const size_t half = i / 128, quarter = (i % 128) / 32, lane = i % 32;
      const uint8_t low = p[half * 64 + (quarter % 2) * 32 + lane];
      const uint8_t high = p[128 + half * 32 + lane];
      const int code = ((low >> (4 * (quarter / 2))) & 15) | (((high >> (2 * quarter)) & 3) << 4);
      value = parameters.scales[i / 16] * static_cast<float>(code - 32);
    } else {
      const bool q5 = format == QuantizationFormat::kQ5K;
      const size_t qs = q5 ? 48 : 16;
      int code = (p[qs + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
      if (q5)
        code |= ((p[16 + i % 32] >> (i / 32)) & 1) << 4;
      // Separate FLOAT32 rounding prevents contraction into a fused multiply-subtract.
      volatile float product = parameters.scales[i / 32] * static_cast<float>(code);
      value = product - parameters.minima[i / 32];
    }
    EXT_ENFORCE_INVALID(std::isfinite(value), "GGUF reconstructs a nonfinite FLOAT32 value.");
    output[i] = value;
  }
}

void ValidateBlocks(std::string_view data, const ValidatedLayout &validated) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(data.data());
  // Finite binary16 fields and bounded integer scales/codes cannot overflow FLOAT32.
  for (size_t i = 0; i < validated.blocks; ++i)
    Parameters(bytes + i * validated.layout.type_size, validated.format);
}

StructTypeProto Schema(const ValidatedLayout &validated) {
  StructTypeProto type;
  type.set_name(std::string(kGgufTypePrefix) + validated.layout.subtype);
  auto *revision = type.mutable_structure()->add_field();
  revision->set_name("upstream_revision");
  auto *constant = revision->mutable_constant();
  constant->set_data_type(TensorProto::UINT8);
  constant->add_dims(std::strlen(kGgufUpstreamRevision));
  constant->set_raw_data(kGgufUpstreamRevision);
  auto *field = type.mutable_structure()->add_field();
  field->set_name("blocks");
  auto *blocks = field->mutable_type()->mutable_tensor_type();
  blocks->set_elem_type(TensorProto::UINT8);
  blocks->mutable_shape()->add_dim()->set_dim_value(validated.blocks);
  blocks->mutable_shape()->add_dim()->set_dim_value(validated.layout.type_size);
  return type;
}

EncodedValueProto MakeValue(std::string_view data, const ValidatedLayout &validated) {
  EncodedValueProto value;
  *value.mutable_struct_type() = Schema(validated);
  auto *logical = value.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  for (const auto dim : validated.shape)
    logical->mutable_shape()->add_dim()->set_dim_value(dim);
  value.set_raw_data(data.data(), data.size());
  return value;
}

ValidatedLayout Validate(const EncodedValueProto &payload, size_t max_bytes,
                         const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!payload.has_parameter_ref(), "GGUF does not accept shared parameters.");
  EXT_ENFORCE_INVALID(payload.has_raw_data() && payload.raw_data().size() <= max_bytes,
                      "GGUF requires loaded bytes within max_bytes.");
  const auto storage = catalogue.ValidateEncodedValue(payload);
  EXT_ENFORCE_INVALID(!storage.external && storage.root && storage.record_count == 1,
                      "GGUF requires one loaded encoded record.");
  const std::string name = storage.root->name().value();
  EXT_ENFORCE_INVALID(name.starts_with(kGgufTypePrefix), "Unsupported GGUF encoded layout.");
  EXT_ENFORCE_INVALID(payload.has_logical_type() && payload.logical_type().has_tensor_type(),
                      "GGUF requires a logical FLOAT tensor.");
  const auto &logical = payload.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(logical.elem_type() == TensorProto::FLOAT && logical.has_shape(),
                      "GGUF requires a concrete logical FLOAT shape.");
  Shape shape;
  for (const auto &dim : logical.shape().dim()) {
    EXT_ENFORCE_INVALID(dim.has_dim_value(), "GGUF requires concrete dimensions.");
    shape.push_back(dim.dim_value());
  }
  const auto validated =
      ValidateLayout(payload.raw_data().size(), shape, name.substr(std::strlen(kGgufTypePrefix)),
                     kGgufLayoutVersion, kGgufUpstreamRevision, max_bytes);
  auto actual = *storage.root;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(validated), &difference),
                      "GGUF descriptor does not match its native schema: ", difference);
  return validated;
}

Shape PackedMatrixShape(const Shape &shape, const GgufBlockLayout &layout) {
  EXT_ENFORCE_INVALID(shape.size() == 2, "GPTQModel GGUF requires a rank-two logical shape.");
  const size_t row_bytes =
      CheckedMultiply(static_cast<size_t>(shape[1]) / layout.block_size, layout.type_size);
  EXT_ENFORCE_INVALID(row_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                      "GPTQModel GGUF packed row exceeds int64.");
  return {shape[0], static_cast<int64_t>(row_bytes)};
}

ValidatedLayout ValidateCheckpointMetadata(const Shape &packed_shape, size_t packed_bytes,
                                           const Shape &shape, std::string_view subtype,
                                           std::string_view version, std::string_view revision,
                                           size_t max_bytes) {
  EXT_ENFORCE_INVALID(shape.size() == 2, "GPTQModel GGUF requires a rank-two logical shape.");
  const auto validated = ValidateLayout(packed_bytes, shape, subtype, version, revision, max_bytes);
  EXT_ENFORCE_INVALID(packed_shape == PackedMatrixShape(shape, validated.layout),
                      "GPTQModel GGUF packed matrix shape mismatch; row padding is unsupported.");
  return validated;
}

} // namespace

const std::vector<GgufBlockLayout> &GgufBlockLayouts() {
  static const std::vector<GgufBlockLayout> layouts{
      {std::string(QuantizationFormatName(QuantizationFormat::kQ40)), 32, 18},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ80)), 32, 34},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ4K)), 256, 144},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ5K)), 256, 176},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ6K)), 256, 210},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ2K)), 256, 84},
      {std::string(QuantizationFormatName(QuantizationFormat::kQ3K)), 256, 110},
      {std::string(QuantizationFormatName(QuantizationFormat::kIq4Nl)), 32, 18},
      {std::string(QuantizationFormatName(QuantizationFormat::kIq1S)), 256, 50},
      {std::string(QuantizationFormatName(QuantizationFormat::kTq10)), 256, 54},
      {std::string(QuantizationFormatName(QuantizationFormat::kTq20)), 256, 66}};
  return layouts;
}

const GgufBlockLayout &GgufLayout(std::string_view subtype) {
  for (const auto &layout : GgufBlockLayouts())
    if (layout.subtype == subtype)
      return layout;
  throw std::invalid_argument("Unsupported canonical GGUF subtype: " + std::string(subtype) + ".");
}

EncodedValueProto ImportGgufTensorPayload(std::string_view data, Shape shape, std::string subtype,
                                          std::string layout_version, std::string upstream_revision,
                                          size_t max_bytes) {
  const auto validated =
      ValidateLayout(data.size(), shape, subtype, layout_version, upstream_revision, max_bytes);
  ValidateBlocks(data, validated);
  return MakeValue(data, validated);
}

std::string ExportGgufTensorPayload(const EncodedValueProto &payload, size_t max_bytes,
                                    const StructTypeCatalogue &catalogue) {
  const auto validated = Validate(payload, max_bytes, catalogue);
  ValidateBlocks(Bytes(payload), validated);
  return std::string(payload.raw_data());
}

Tensor DequantizeGgufTensorPayload(const EncodedValueProto &payload, size_t max_bytes,
                                   const StructTypeCatalogue &catalogue,
                                   RawBufferAllocator *allocator) {
  const auto validated = Validate(payload, max_bytes, catalogue);
  Tensor output =
      MakeOutputTensor(TensorProto::FLOAT, validated.shape, validated.decoded_bytes, allocator);
  output.name = payload.name().value();
  std::array<float, 256> block{};
  const auto *bytes = reinterpret_cast<const uint8_t *>(payload.raw_data().data());
  for (size_t i = 0; i < validated.blocks; ++i) {
    DecodeBlock(bytes + i * validated.layout.type_size, validated.layout, validated.format,
                block.data());
    std::memcpy(output.mutable_bytes() + i * validated.layout.block_size * sizeof(float),
                block.data(), validated.layout.block_size * sizeof(float));
  }
  return output;
}

GgufBlockParameters ReadGgufBlockParameters(const EncodedValueProto &payload, size_t block_index,
                                            size_t max_bytes,
                                            const StructTypeCatalogue &catalogue) {
  const auto validated = Validate(payload, max_bytes, catalogue);
  EXT_ENFORCE_INVALID(block_index < validated.blocks, "GGUF block index out of range.");
  const auto parameters = Parameters(reinterpret_cast<const uint8_t *>(payload.raw_data().data()) +
                                         block_index * validated.layout.type_size,
                                     validated.format);
  return {{parameters.scales.begin(), parameters.scales.begin() + parameters.scale_count},
          {parameters.minima.begin(), parameters.minima.begin() + parameters.minimum_count}};
}

EncodedValueProto ImportGptqModelGgufWeights(const GptqModelGgufWeights &weights,
                                             size_t max_bytes) {
  EXT_ENFORCE_INVALID(weights.qweight.data_type == TensorProto::UINT8,
                      "GPTQModel GGUF qweight must be UINT8.");
  const auto validated = ValidateCheckpointMetadata(
      weights.qweight.shape, weights.qweight.size_bytes(), weights.shape, weights.subtype,
      weights.layout_version, weights.upstream_revision, max_bytes);
  EXT_ENFORCE_INVALID(weights.qweight.bytes() != nullptr,
                      "GPTQModel GGUF qweight has no byte storage.");
  const std::string_view data(reinterpret_cast<const char *>(weights.qweight.bytes()),
                              weights.qweight.size_bytes());
  ValidateBlocks(data, validated);
  auto value = MakeValue(data, validated);
  value.set_name(weights.qweight.name);
  return value;
}

void ValidateGptqModelGgufMetadata(const Shape &packed_shape, size_t packed_bytes,
                                   const Shape &shape, std::string_view subtype,
                                   std::string_view layout_version,
                                   std::string_view upstream_revision, size_t max_bytes) {
  ValidateCheckpointMetadata(packed_shape, packed_bytes, shape, subtype, layout_version,
                             upstream_revision, max_bytes);
}

GptqModelGgufWeights ExportGptqModelGgufWeights(const EncodedValueProto &payload, size_t max_bytes,
                                                const StructTypeCatalogue &catalogue) {
  const auto validated = Validate(payload, max_bytes, catalogue);
  const auto packed_shape = PackedMatrixShape(validated.shape, validated.layout);
  ValidateBlocks(Bytes(payload), validated);
  Tensor qweight =
      MakeOutputTensor(TensorProto::UINT8, packed_shape, payload.raw_data().size(), nullptr);
  qweight.name = "qweight";
  std::memcpy(qweight.mutable_bytes(), payload.raw_data().data(), payload.raw_data().size());
  return {std::move(qweight), validated.shape, validated.layout.subtype, kGgufLayoutVersion,
          kGgufUpstreamRevision};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
