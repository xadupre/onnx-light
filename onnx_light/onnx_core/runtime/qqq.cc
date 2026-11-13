// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr const char *kName = "onnx_light.qqq_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "QQQ size overflow.");
  return a * b;
}

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  int64_t k, n, group, bits;
  bool grouped;
  size_t count, total = 0;
  std::vector<Field> fields;
  std::array<size_t, 1024> permutation, inverse;

  Geometry(int64_t rows, int64_t columns, int64_t grouping, int64_t code_bits, size_t limit)
      : k(rows), n(columns), group(grouping), bits(code_bits) {
    EXT_ENFORCE_INVALID(
        limit > 0 && k > 0 && n > 0 && k % 64 == 0 && n % 64 == 0 && bits == 4 &&
            (group == -1 || group == 128),
        "QQQ requires four-bit, unpadded dimensions divisible by 64 and group -1/128.");
    grouped = group == 128 && k != 128;
    EXT_ENFORCE_INVALID(!grouped || k % 128 == 0, "QQQ incomplete 128-element group.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(k) <= std::numeric_limits<size_t>::max() &&
                            static_cast<uint64_t>(n) <= std::numeric_limits<size_t>::max(),
                        "QQQ dimension overflow.");
    count = Product(static_cast<size_t>(k), static_cast<size_t>(n));
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "QQQ logical FLOAT weights exceed max_bytes.");
    EXT_ENFORCE_INVALID(n <= std::numeric_limits<int64_t>::max() / 2,
                        "QQQ packed dimension overflow.");
    Add("B", TensorProto::INT32, {k / 16, 2 * n}, 4, limit);
    Add("s_channel", TensorProto::FLOAT, {1, n}, 4, limit);
    Add("s_group", TensorProto::FLOAT16, grouped ? Shape{k / 128, n} : Shape{0}, 2, limit);
    // Pinned _get_perms gathers a 16-by-64 tile in eight-nibble words.
    constexpr size_t channel[] = {4, 0, 5, 1, 6, 2, 7, 3};
    constexpr size_t grouped_order[] = {0, 2, 4, 6, 1, 3, 5, 7};
    const auto *order = grouped ? grouped_order : channel;
    for (size_t i = 0; i < 32; ++i)
      for (size_t j = 0; j < 4; ++j)
        for (size_t lane = 0; lane < 8; ++lane) {
          const size_t p = order[lane];
          const size_t index = i * 32 + j * 8 + lane;
          permutation[index] = 256 * j + 16 * (4 * (i % 4) + p % 4) + i / 4 + 8 * (p / 4);
          inverse[permutation[index]] = index;
        }
  }

  void Add(const char *name, int32_t type, Shape shape, size_t width, size_t limit) {
    size_t bytes = width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                        "QQQ checkpoint exceeds max_bytes.");
    fields.push_back({name, type, std::move(shape), width, bytes, total});
    total += bytes;
  }

  size_t Position(size_t n_index, size_t k_index) const {
    const size_t tiled = (k_index / 16) * static_cast<size_t>(n) * 16 + (n_index / 16) * 256 +
                         (k_index % 16) * 16 + n_index % 16;
    return (tiled / 1024) * 1024 + inverse[tiled % 1024];
  }
};

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  if (bytes == 0)
    return;
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t i = 0; i < bytes; i += width)
      std::reverse(destination + i, destination + i + width);
}

uint32_t Word(const uint8_t *data, size_t width) {
  uint32_t result = 0;
  for (size_t i = 0; i < width; ++i)
    result |= uint32_t(data[i]) << (8 * i);
  return result;
}

StructTypeProto Schema(const Geometry &g) {
  StructTypeProto result;
  result.set_name(kName);
  auto number = [&](const char *name, int64_t value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::INT64);
    field->mutable_constant()->add_int64_data(value);
  };
  number("in_features", g.k);
  number("out_features", g.n);
  number("group_size", g.group);
  number("bits", g.bits);
  auto *revision = result.mutable_structure()->add_field();
  revision->set_name("gptqmodel_revision");
  revision->mutable_constant()->set_data_type(TensorProto::UINT8);
  revision->mutable_constant()->add_dims(std::strlen(kRevision));
  revision->mutable_constant()->set_raw_data(kRevision);
  for (const auto &spec : g.fields) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(spec.name);
    auto *tensor = field->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(spec.type);
    tensor->mutable_shape();
    for (int64_t dim : spec.shape)
      tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return result;
}

void CheckTensor(const Tensor &tensor, const Field &field) {
  EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                          tensor.size_bytes() == field.bytes &&
                          (field.bytes == 0 || tensor.bytes() != nullptr),
                      "QQQ invalid tensor type, shape or byte extent: ", field.name, ".");
}

size_t ScalePosition(size_t index, bool channel) {
  if (!channel)
    return (index / 64) * 64 + (index % 8) * 8 + (index % 64) / 8;
  const size_t local = index % 32;
  return (index / 32) * 32 + ((local % 8) / 2) * 8 + (local / 8) * 2 + local % 2;
}

float Channel(const Geometry &g, const uint8_t *data, size_t n) {
  return std::bit_cast<float>(Word(data + g.fields[1].offset + ScalePosition(n, true) * 4, 4));
}

float Group(const Geometry &g, const uint8_t *data, size_t n, size_t k) {
  const size_t index = (k / 128) * static_cast<size_t>(g.n) + n;
  return Float16BitsToFloat(
      static_cast<uint16_t>(Word(data + g.fields[2].offset + ScalePosition(index, false) * 2, 2)));
}

uint8_t Code(const Geometry &g, const uint8_t *data, size_t n, size_t k) {
  const size_t position = g.Position(n, k);
  return (data[position / 2] >> ((position % 2) * 4)) & 15;
}

// Products of an integer of magnitude <=128 and a FLOAT are exact in DOUBLE.
// Rounds that exact product to binary32 without consulting the ambient fenv.
float RoundFloat(double value) {
  const uint64_t bits = std::bit_cast<uint64_t>(value);
  const uint32_t sign = static_cast<uint32_t>(bits >> 32) & 0x80000000u;
  if ((bits << 1) == 0)
    return std::bit_cast<float>(sign);
  const int exponent = static_cast<int>((bits >> 52) & 2047) - 1023 + 127;
  const uint64_t mantissa = (bits & ((uint64_t{1} << 52) - 1)) | (uint64_t{1} << 52);
  const int shift = exponent > 0 ? 29 : 30 - exponent;
  uint64_t rounded = 0;
  if (shift < 64) {
    rounded = mantissa >> shift;
    const uint64_t remainder = mantissa & ((uint64_t{1} << shift) - 1);
    const uint64_t half = uint64_t{1} << (shift - 1);
    rounded += remainder > half || (remainder == half && (rounded & 1));
  }
  uint64_t magnitude = exponent > 0 ? (uint64_t(exponent - 1) << 23) + rounded : rounded;
  if (magnitude >= 0x7f800000u)
    magnitude = 0x7f800000u;
  return std::bit_cast<float>(sign | static_cast<uint32_t>(magnitude));
}

float Reconstructed(const Geometry &g, const uint8_t *data, size_t n, size_t k) {
  const int code = Code(g, data, n, k);
  float integer;
  if (g.grouped) {
    // HALF times a four-bit integer is exact in FLOAT: no scale precombination.
    const double scaled = double(code - 8) * double(Group(g, data, n, k));
    const double lower = std::floor(scaled), fraction = scaled - lower;
    const double rounded =
        lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2) != 0));
    integer = rounded == 0 ? std::copysign(0.0f, scaled)
                           : static_cast<float>(std::clamp(rounded, -128.0, 127.0));
  } else {
    integer = static_cast<float>((code >= 8 ? code - 16 : code) * 16);
  }
  const float value = RoundFloat(double(integer) * double(Channel(g, data, n)));
  EXT_ENFORCE_INVALID(std::isfinite(value), "QQQ FLOAT reconstruction overflow.");
  return value;
}

void CheckData(const Geometry &g, const uint8_t *data) {
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    EXT_ENFORCE_INVALID(std::isfinite(Channel(g, data, n)) && Channel(g, data, n) >= 0,
                        "QQQ nonfinite or negative channel scale.");
  for (size_t i = 0; i < g.fields[2].bytes / 2; ++i) {
    const float scale =
        Float16BitsToFloat(static_cast<uint16_t>(Word(data + g.fields[2].offset + i * 2, 2)));
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale >= 0,
                        "QQQ nonfinite or negative group scale.");
  }
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
      Reconstructed(g, data, n, k);
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "QQQ requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() == 8,
                      "QQQ unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "QQQ malformed metadata.");
    return field.constant().int64_data(0);
  };
  Geometry g(number(0), number(1), number(2), number(3), limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference), "QQQ invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(
      logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
          logical.shape().dim_size() == 2 && logical.shape().dim(0).has_dim_value() &&
          logical.shape().dim(0).dim_value() == g.n && logical.shape().dim(1).has_dim_value() &&
          logical.shape().dim(1).dim_value() == g.k,
      "QQQ invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "QQQ truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "QQQ requires one loaded encoded record.");
  CheckData(g, reinterpret_cast<const uint8_t *>(value.raw_data().data()));
  return g;
}

EncodedValueProto EmptyEncoded(const Geometry &g) {
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(g.n);
  logical->mutable_shape()->add_dim()->set_dim_value(g.k);
  result.mutable_raw_data()->resize(g.total);
  std::fill(result.mutable_raw_data()->data(),
            result.mutable_raw_data()->data() + result.raw_data().size(), uint8_t{0});
  return result;
}

} // namespace

EncodedValueProto ImportQqqWeights(const QqqWeights &weights, size_t max_bytes) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision, "QQQ unsupported upstream revision.");
  Geometry g(weights.in_features, weights.out_features, weights.group_size, weights.bits,
             max_bytes);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(), "QQQ unexpected tensor set.");
  for (const auto &field : g.fields) {
    const auto found = weights.tensors.find(field.name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end(), "QQQ missing tensor ", field.name, ".");
    CheckTensor(found->second, field);
  }
  auto result = EmptyEncoded(g);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  for (const auto &field : g.fields)
    CopyWords(data + field.offset, weights.tensors.at(field.name).bytes(), field.bytes,
              field.width);
  CheckData(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

QqqWeights ExportQqqWeights(const EncodedValueProto &value, size_t max_bytes,
                            const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  QqqWeights result;
  result.in_features = g.k;
  result.out_features = g.n;
  result.group_size = g.group;
  result.bits = static_cast<uint32_t>(g.bits);
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (const auto &field : g.fields) {
    std::vector<uint8_t> bytes(field.bytes);
    CopyWords(bytes.data(), data + field.offset, field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackQqqCodes(const EncodedValueProto &value, size_t max_bytes,
                      const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::UINT8, {g.n, g.k},
                std::vector<uint8_t>(g.count));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
      result.mutable_bytes()[n * static_cast<size_t>(g.k) + k] = Code(g, data, n, k);
  return result;
}

Tensor ReconstructQqqWeights(const EncodedValueProto &value, size_t max_bytes,
                             const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::FLOAT, {g.n, g.k},
                std::vector<uint8_t>(g.count * sizeof(float)));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k) {
      const float reconstructed = Reconstructed(g, data, n, k);
      std::memcpy(result.mutable_bytes() + (n * static_cast<size_t>(g.k) + k) * sizeof(float),
                  &reconstructed, sizeof(float));
    }
  return result;
}

EncodedValueProto PackQqqWeights(const QqqWeights &metadata, const Tensor &codes,
                                 const Tensor &s_channel, const Tensor &s_group, size_t max_bytes) {
  EXT_ENFORCE_INVALID(metadata.tensors.empty(), "QQQ packing metadata must not contain tensors.");
  EXT_ENFORCE_INVALID(metadata.upstream_revision == kRevision,
                      "QQQ unsupported upstream revision.");
  Geometry g(metadata.in_features, metadata.out_features, metadata.group_size, metadata.bits,
             max_bytes);
  CheckTensor(codes, {"codes", TensorProto::UINT8, {g.n, g.k}, 1, g.count, 0});
  CheckTensor(s_channel, g.fields[1]);
  CheckTensor(s_group, g.fields[2]);
  for (size_t i = 0; i < g.count; ++i)
    EXT_ENFORCE_INVALID(codes.bytes()[i] <= 15, "QQQ code is out of range.");
  auto result = EmptyEncoded(g);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  CopyWords(data + g.fields[1].offset, s_channel.bytes(), g.fields[1].bytes, 4);
  CopyWords(data + g.fields[2].offset, s_group.bytes(), g.fields[2].bytes, 2);
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k) {
      const size_t position = g.Position(n, k);
      data[position / 2] |= codes.bytes()[n * static_cast<size_t>(g.k) + k] << ((position % 2) * 4);
    }
  CheckData(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
