// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr const char *kName = "onnx_light.gptqmodel_fp8_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "GPTQModel FP8 size overflow.");
  return a * b;
}

int32_t Dtype(std::string_view format) {
  if (format == "float8_e4m3fn")
    return TensorProto::FLOAT8E4M3FN;
  if (format == "float8_e4m3fnuz")
    return TensorProto::FLOAT8E4M3FNUZ;
  if (format == "float8_e5m2")
    return TensorProto::FLOAT8E5M2;
  if (format == "float8_e5m2fnuz")
    return TensorProto::FLOAT8E5M2FNUZ;
  if (format == "float8_e8m0fnu")
    return TensorProto::FLOAT8E8M0;
  EXT_THROW_INVALID("GPTQModel FP8 unsupported scalar format.");
}

float Decode(uint8_t code, int32_t type) {
  switch (type) {
  case TensorProto::FLOAT8E4M3FN:
    return Float8E4M3FNBitsToFloat(code);
  case TensorProto::FLOAT8E4M3FNUZ:
    return Float8E4M3FNUZBitsToFloat(code);
  case TensorProto::FLOAT8E5M2:
    return Float8E5M2BitsToFloat(code);
  case TensorProto::FLOAT8E5M2FNUZ:
    return Float8E5M2FNUZBitsToFloat(code);
  default:
    return Float8E8M0BitsToFloat(code);
  }
}

struct Geometry {
  GptqModelFp8Weights metadata;
  int32_t type;
  Shape scale_shape;
  size_t rows, columns, count, scales = 1, total;

  Geometry(GptqModelFp8Weights weights, size_t limit) : metadata(std::move(weights)) {
    EXT_ENFORCE_INVALID(metadata.upstream_revision == kRevision,
                        "GPTQModel FP8 unsupported upstream revision.");
    EXT_ENFORCE_INVALID(metadata.weight_scale_semantics == "inverse",
                        "GPTQModel FP8 requires explicit inverse-scale semantics.");
    EXT_ENFORCE_INVALID(limit > 0 && metadata.in_features > 0 && metadata.out_features > 0,
                        "GPTQModel FP8 requires nonempty [N,K] weights and positive max_bytes.");
    EXT_ENFORCE_INVALID(
        static_cast<uint64_t>(metadata.in_features) <= std::numeric_limits<size_t>::max() &&
            static_cast<uint64_t>(metadata.out_features) <= std::numeric_limits<size_t>::max(),
        "GPTQModel FP8 dimension overflow.");
    rows = static_cast<size_t>(metadata.out_features);
    columns = static_cast<size_t>(metadata.in_features);
    count = Product(rows, columns);
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "GPTQModel FP8 logical FLOAT weights exceed max_bytes.");
    type = Dtype(metadata.format);
    const auto &block = metadata.weight_block_size;
    EXT_ENFORCE_INVALID(block.empty() || (block.size() == 2 && block[0] > 0 && block[1] > 0),
                        "GPTQModel FP8 invalid weight_block_size.");
    EXT_ENFORCE_INVALID(metadata.weight_scale_method == "block" || block.empty(),
                        "GPTQModel FP8 non-block scaling forbids weight_block_size.");
    if (metadata.weight_scale_method == "tensor") {
      scale_shape = {};
    } else if (metadata.weight_scale_method == "row") {
      scale_shape = {metadata.out_features};
      scales = rows;
    } else {
      EXT_ENFORCE_INVALID(metadata.weight_scale_method == "block" && block.size() == 2 &&
                              metadata.out_features % block[0] == 0 &&
                              metadata.in_features % block[1] == 0,
                          "GPTQModel FP8 requires divisible two-axis blocks.");
      scale_shape = {metadata.out_features / block[0], metadata.in_features / block[1]};
      scales = Product(static_cast<size_t>(scale_shape[0]), static_cast<size_t>(scale_shape[1]));
    }
    const size_t scale_bytes = Product(scales, sizeof(float));
    EXT_ENFORCE_INVALID(count <= limit && scale_bytes <= limit - count,
                        "GPTQModel FP8 checkpoint exceeds max_bytes.");
    total = count + scale_bytes;
  }

  size_t ScaleIndex(size_t i) const {
    if (metadata.weight_scale_method == "tensor")
      return 0;
    if (metadata.weight_scale_method == "row")
      return i / columns;
    return (i / columns / static_cast<size_t>(metadata.weight_block_size[0])) *
               static_cast<size_t>(scale_shape[1]) +
           (i % columns / static_cast<size_t>(metadata.weight_block_size[1]));
  }
};

StructTypeProto Schema(const Geometry &g) {
  StructTypeProto result;
  result.set_name(kName);
  auto number = [&](const char *name, int64_t value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::INT64);
    field->mutable_constant()->add_int64_data(value);
  };
  auto text = [&](const char *name, const std::string &value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::UINT8);
    field->mutable_constant()->add_dims(static_cast<int64_t>(value.size()));
    field->mutable_constant()->set_raw_data(value);
  };
  number("in_features", g.metadata.in_features);
  number("out_features", g.metadata.out_features);
  text("format", g.metadata.format);
  text("weight_scale_method", g.metadata.weight_scale_method);
  auto *block = result.mutable_structure()->add_field();
  block->set_name("weight_block_size");
  block->mutable_constant()->set_data_type(TensorProto::INT64);
  block->mutable_constant()->add_dims(static_cast<int64_t>(g.metadata.weight_block_size.size()));
  for (int64_t dim : g.metadata.weight_block_size)
    block->mutable_constant()->add_int64_data(dim);
  text("gptqmodel_revision", kRevision);
  text("weight_scale_semantics", g.metadata.weight_scale_semantics);
  auto tensor = [&](const char *name, int32_t dtype, const Shape &shape) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    auto *type = field->mutable_type()->mutable_tensor_type();
    type->set_elem_type(dtype);
    type->mutable_shape();
    for (int64_t dim : shape)
      type->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  tensor("weight", g.type, {g.metadata.out_features, g.metadata.in_features});
  tensor("weight_scale_inv", TensorProto::FLOAT, g.scale_shape);
  return result;
}

float Scale(const uint8_t *data, size_t index) {
  uint32_t bits = 0;
  for (size_t j = 0; j < sizeof(float); ++j)
    bits |= uint32_t(data[index * sizeof(float) + j]) << (8 * j);
  return std::bit_cast<float>(bits);
}

float Reconstructed(const Geometry &g, const uint8_t *data, size_t i) {
  return Decode(data[i], g.type) / Scale(data + g.count, g.ScaleIndex(i));
}

void CheckData(const Geometry &g, const uint8_t *data) {
  for (size_t i = 0; i < g.scales; ++i) {
    const float scale = Scale(data + g.count, i);
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0,
                        "GPTQModel FP8 inverse scales must be finite and positive.");
  }
  for (size_t i = 0; i < g.count; ++i)
    EXT_ENFORCE_INVALID(std::isfinite(Reconstructed(g, data, i)),
                        "GPTQModel FP8 nonfinite weight or reconstruction.");
}

void CopyScales(uint8_t *destination, const uint8_t *source, size_t bytes) {
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t i = 0; i < bytes; i += sizeof(float))
      std::reverse(destination + i, destination + i + sizeof(float));
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "GPTQModel FP8 requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() == 9,
                      "GPTQModel FP8 unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "GPTQModel FP8 malformed integer metadata.");
    return field.constant().int64_data(0);
  };
  auto text = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().has_raw_data() &&
                            field.constant().raw_data().size() <= 64,
                        "GPTQModel FP8 malformed string metadata.");
    return std::string(field.constant().raw_data());
  };
  GptqModelFp8Weights metadata;
  metadata.in_features = number(0);
  metadata.out_features = number(1);
  metadata.format = text(2);
  metadata.weight_scale_method = text(3);
  const auto &block = type.structure().field(4);
  EXT_ENFORCE_INVALID(block.has_constant() && (block.constant().int64_data_size() == 0 ||
                                               block.constant().int64_data_size() == 2),
                      "GPTQModel FP8 malformed block metadata.");
  for (int i = 0; i < block.constant().int64_data_size(); ++i)
    metadata.weight_block_size.push_back(block.constant().int64_data(i));
  metadata.upstream_revision = text(5);
  metadata.weight_scale_semantics = text(6);
  Geometry g(std::move(metadata), limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference),
                      "GPTQModel FP8 invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
                          logical.shape().dim_size() == 2 &&
                          logical.shape().dim(0).has_dim_value() &&
                          logical.shape().dim(0).dim_value() == g.metadata.out_features &&
                          logical.shape().dim(1).has_dim_value() &&
                          logical.shape().dim(1).dim_value() == g.metadata.in_features,
                      "GPTQModel FP8 invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "GPTQModel FP8 truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "GPTQModel FP8 requires one loaded encoded record.");
  CheckData(g, reinterpret_cast<const uint8_t *>(value.raw_data().data()));
  return g;
}

} // namespace

EncodedValueProto ImportGptqModelFp8Weights(const GptqModelFp8Weights &weights, size_t max_bytes) {
  // Avoid copying loaded tensors while deriving their geometry.
  GptqModelFp8Weights metadata;
  metadata.in_features = weights.in_features;
  metadata.out_features = weights.out_features;
  metadata.format = weights.format;
  metadata.weight_scale_method = weights.weight_scale_method;
  metadata.weight_scale_semantics = weights.weight_scale_semantics;
  metadata.weight_block_size = weights.weight_block_size;
  metadata.upstream_revision = weights.upstream_revision;
  Geometry g(std::move(metadata), max_bytes);
  EXT_ENFORCE_INVALID(weights.tensors.size() == 2 && weights.tensors.contains("weight") &&
                          weights.tensors.contains("weight_scale_inv"),
                      "GPTQModel FP8 requires exactly weight and weight_scale_inv.");
  const Tensor &weight = weights.tensors.at("weight");
  const Tensor &scale = weights.tensors.at("weight_scale_inv");
  EXT_ENFORCE_INVALID(weight.data_type == g.type &&
                          weight.shape == Shape({weights.out_features, weights.in_features}) &&
                          weight.size_bytes() == g.count && weight.bytes() != nullptr &&
                          scale.data_type == TensorProto::FLOAT && scale.shape == g.scale_shape &&
                          scale.size_bytes() == g.total - g.count && scale.bytes() != nullptr,
                      "GPTQModel FP8 invalid tensor dtype, shape or byte extent.");
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(weights.out_features);
  logical->mutable_shape()->add_dim()->set_dim_value(weights.in_features);
  result.mutable_raw_data()->resize(g.total);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  std::memcpy(data, weight.bytes(), g.count);
  CopyScales(data + g.count, scale.bytes(), g.total - g.count);
  CheckData(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

GptqModelFp8Weights ExportGptqModelFp8Weights(const EncodedValueProto &value, size_t max_bytes,
                                              const StructTypeCatalogue &catalogue) {
  auto g = Validate(value, max_bytes, catalogue);
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  g.metadata.tensors.emplace("weight", Tensor("weight", g.type,
                                              {g.metadata.out_features, g.metadata.in_features},
                                              std::vector<uint8_t>(data, data + g.count)));
  std::vector<uint8_t> scales(g.total - g.count);
  CopyScales(scales.data(), data + g.count, scales.size());
  g.metadata.tensors.emplace("weight_scale_inv", Tensor("weight_scale_inv", TensorProto::FLOAT,
                                                        g.scale_shape, std::move(scales)));
  return std::move(g.metadata);
}

Tensor ReconstructGptqModelFp8Weights(const EncodedValueProto &value, size_t max_bytes,
                                      const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::FLOAT,
                {g.metadata.out_features, g.metadata.in_features},
                std::vector<uint8_t>(g.count * sizeof(float)));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t i = 0; i < g.count; ++i) {
    const float reconstructed = Reconstructed(g, data, i);
    std::memcpy(result.mutable_bytes() + i * sizeof(float), &reconstructed, sizeof(float));
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
