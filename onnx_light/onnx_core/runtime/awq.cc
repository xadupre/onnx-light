// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr const char *kName = "onnx_light.awq_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "AWQ size overflow.");
  return a * b;
}

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t offset = 0; offset < bytes; offset += width)
      std::reverse(destination + offset, destination + offset + width);
}

uint32_t Word(const uint8_t *bytes, size_t width) {
  uint32_t result = 0;
  for (size_t i = 0; i < width; ++i)
    result |= uint32_t(bytes[i]) << (8 * i);
  return result;
}

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  std::string variant;
  int64_t k, n, group;
  size_t count = 0, total = 0, groups = 0, padded_groups = 0;
  std::vector<Field> fields;

  void Add(const char *name, int32_t type, Shape shape, size_t width, size_t limit) {
    size_t bytes = width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                        "AWQ checkpoint exceeds max_bytes.");
    fields.push_back({name, type, shape, width, bytes, total});
    total += bytes;
  }

  Geometry(std::string kind, int64_t rows, int64_t columns, int64_t grouping, int64_t bits,
           int64_t axis, size_t limit, int32_t scale_type = TensorProto::FLOAT16)
      : variant(std::move(kind)), k(rows), n(columns), group(grouping) {
    EXT_ENFORCE_INVALID(limit > 0 && k > 0 && n > 0 && bits == 4 && axis == 1,
                        "AWQ requires positive dimensions, four bits and group_axis=1.");
    EXT_ENFORCE_INVALID(variant == "gemm" || variant == "gemv" || variant == "gemv_fast" ||
                            variant == "llm-awq",
                        "AWQ unsupported variant.");
    EXT_ENFORCE_INVALID(group == -1 || group > 0, "AWQ unsupported group size.");
    const int64_t effective_group = group == -1 ? k : group;
    EXT_ENFORCE_INVALID(k % effective_group == 0, "AWQ requires complete groups.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(k) <= std::numeric_limits<size_t>::max() &&
                            static_cast<uint64_t>(n) <= std::numeric_limits<size_t>::max(),
                        "AWQ dimension size overflow.");
    count = Product(static_cast<size_t>(k), static_cast<size_t>(n));
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "AWQ logical FLOAT weights exceed max_bytes.");
    groups = static_cast<size_t>(k / effective_group);
    if (variant == "gemm") {
      EXT_ENFORCE_INVALID(
          n % 8 == 0 && (scale_type == TensorProto::FLOAT16 || scale_type == TensorProto::BFLOAT16),
          "AWQ GEMM requires N divisible by eight and FP16/BF16 scales.");
      Add("qweight", TensorProto::INT32, {k, n / 8}, 4, limit);
      Add("qzeros", TensorProto::INT32, {static_cast<int64_t>(groups), n / 8}, 4, limit);
      Add("scales", scale_type, {static_cast<int64_t>(groups), n}, 2, limit);
    } else {
      EXT_ENFORCE_INVALID(effective_group >= 128 || effective_group == 16 ||
                              effective_group == 32 || effective_group == 64,
                          "AWQ invalid effective group size.");
      EXT_ENFORCE_INVALID(scale_type == TensorProto::FLOAT16,
                          "AWQ GEMV requires FLOAT16 parameters.");
      const size_t multiplier = effective_group >= 128 ? 1 : 128 / effective_group;
      const size_t words = groups / 8 + (groups % 8 != 0);
      padded_groups = Product(words / multiplier + (words % multiplier != 0), multiplier * 8);
      EXT_ENFORCE_INVALID(padded_groups <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                          "AWQ padded group overflow.");
      const auto p = static_cast<int64_t>(padded_groups);
      if (variant == "gemv") {
        EXT_ENFORCE_INVALID(k % 8 == 0, "AWQ GEMV requires K divisible by eight.");
        Add("qweight", TensorProto::INT32, {n, k / 8}, 4, limit);
        Add("qzeros", TensorProto::INT32, {n, p / 8}, 4, limit);
        Add("scales", TensorProto::FLOAT16, {n, p}, 2, limit);
      } else {
        EXT_ENFORCE_INVALID(k % 64 == 0 && n % 4 == 0,
                            "AWQ GEMV_FAST requires K divisible by 64 and N by four.");
        Add("qweight", TensorProto::INT16, {n / 4, k}, 2, limit);
        Add(variant == "llm-awq" ? "scaled_zeros" : "qzeros", TensorProto::FLOAT16, {p, n}, 2,
            limit);
        Add("scales", TensorProto::FLOAT16, {p, n}, 2, limit);
      }
    }
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
    field->mutable_constant()->add_dims(value.size());
    field->mutable_constant()->set_raw_data(value);
  };
  text("variant", g.variant);
  number("in_features", g.k);
  number("out_features", g.n);
  number("group_size", g.group);
  number("bits", 4);
  number("group_axis", 1);
  text("gptqmodel_revision", kRevision);
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

float Parameter(const Field &field, const uint8_t *data, size_t index, bool native = false) {
  uint16_t bits = static_cast<uint16_t>(Word(data + index * 2, 2));
  if (native)
    std::memcpy(&bits, data + index * 2, 2);
  return field.type == TensorProto::BFLOAT16 ? Bfloat16BitsToFloat(bits) : Float16BitsToFloat(bits);
}

void CheckParameter(const Geometry &g, const Field &field, const uint8_t *data,
                    bool native = false) {
  if (field.type != TensorProto::FLOAT16 && field.type != TensorProto::BFLOAT16)
    return;
  for (size_t i = 0; i < field.bytes / 2; ++i) {
    const float value = Parameter(field, data, i, native);
    EXT_ENFORCE_INVALID(std::isfinite(value), "AWQ nonfinite parameter.");
    if (std::string_view(field.name) == "scales") {
      const size_t group = g.variant == "gemv" ? i % g.padded_groups : i / static_cast<size_t>(g.n);
      EXT_ENFORCE_INVALID(group >= g.groups || value > 0, "AWQ nonpositive used scale.");
    }
  }
}

void CheckParameters(const Geometry &g, const uint8_t *data) {
  for (const auto &field : g.fields)
    CheckParameter(g, field, data + field.offset);
}

uint8_t Code(const Geometry &g, const uint8_t *data, size_t n, size_t k, bool native = false) {
  size_t word, shift, width = 4;
  if (g.variant == "gemm") {
    constexpr size_t inverse[] = {0, 4, 1, 5, 2, 6, 3, 7};
    word = k * (static_cast<size_t>(g.n) / 8) + n / 8;
    shift = inverse[n % 8] * 4;
  } else if (g.variant == "gemv") {
    word = n * (static_cast<size_t>(g.k) / 8) + k / 8;
    shift = (k % 8) * 4;
  } else {
    // Inverts pack_intweight's two 32-code permutations and four-row/64-code interleave.
    const size_t i = k % 32;
    const size_t first = ((i / 2) % 4) * 8 + (i / 8) * 2 + i % 2;
    const size_t second = (first / 8) * 8 + (first % 2) * 4 + (first % 8) / 2;
    const size_t flat = (n % 4) * 64 + (k % 64) / 32 * 32 + second;
    word = (n / 4) * static_cast<size_t>(g.k) + (k / 64) * 64 + flat / 4;
    shift = (flat % 4) * 4;
    width = 2;
  }
  uint32_t packed = Word(data + word * width, width);
  if (native) {
    if (width == 4)
      std::memcpy(&packed, data + word * width, 4);
    else {
      uint16_t half;
      std::memcpy(&half, data + word * width, 2);
      packed = half;
    }
  }
  return static_cast<uint8_t>((packed >> shift) & 15);
}

float Reconstructed(const Geometry &g, const std::array<const uint8_t *, 3> &data, size_t n,
                    size_t k, bool native = false) {
  const size_t group = k / static_cast<size_t>(g.group == -1 ? g.k : g.group);
  const size_t index =
      g.variant == "gemv" ? n * g.padded_groups + group : group * static_cast<size_t>(g.n) + n;
  const float scale = Parameter(g.fields[2], data[2], index, native);
  const float code = Code(g, data[0], n, k, native);
  float result;
  if (g.variant == "gemm" || g.variant == "gemv") {
    uint32_t zero;
    size_t shift, word;
    if (g.variant == "gemm") {
      constexpr size_t inverse[] = {0, 4, 1, 5, 2, 6, 3, 7};
      word = group * (static_cast<size_t>(g.n) / 8) + n / 8;
      shift = inverse[n % 8] * 4;
    } else {
      word = n * (g.padded_groups / 8) + group / 8;
      shift = (group % 8) * 4;
    }
    zero = Word(data[1] + word * 4, 4);
    if (native)
      std::memcpy(&zero, data[1] + word * 4, 4);
    result = (code - static_cast<float>((zero >> shift) & 15)) * scale;
  } else {
    volatile float product = code * scale;
    result = product + Parameter(g.fields[1], data[1], index, native);
  }
  EXT_ENFORCE_INVALID(std::isfinite(result), "AWQ FLOAT reconstruction overflow.");
  return result;
}

void CheckReconstruction(const Geometry &g, const std::array<const uint8_t *, 3> &data,
                         bool native = false) {
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
      Reconstructed(g, data, n, k, native);
}

Geometry Validate(const AwqWeights &weights, size_t limit) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision, "AWQ unsupported upstream revision.");
  const auto scales = weights.tensors.find("scales");
  EXT_ENFORCE_INVALID(scales != weights.tensors.end(), "AWQ missing scales.");
  Geometry g(weights.variant, weights.in_features, weights.out_features, weights.group_size,
             weights.bits, weights.group_axis, limit, scales->second.data_type);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(), "AWQ unexpected tensors.");
  std::array<const uint8_t *, 3> data;
  size_t index = 0;
  for (const auto &field : g.fields) {
    const auto found = weights.tensors.find(field.name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end(), "AWQ missing tensor ", field.name, ".");
    const auto &tensor = found->second;
    EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                            tensor.size_bytes() == field.bytes && tensor.bytes() != nullptr,
                        "AWQ invalid tensor type, shape or byte extent: ", field.name, ".");
    CheckParameter(g, field, tensor.bytes(), true);
    data[index++] = tensor.bytes();
  }
  CheckReconstruction(g, data, true);
  return g;
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "AWQ requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() == 10,
                      "AWQ unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "AWQ malformed metadata.");
    return field.constant().int64_data(0);
  };
  const auto &variant = type.structure().field(0);
  EXT_ENFORCE_INVALID(variant.has_constant() && variant.constant().has_raw_data() &&
                          variant.constant().raw_data().size() <= 16,
                      "AWQ malformed variant.");
  Geometry g(variant.constant().raw_data(), number(1), number(2), number(3), number(4), number(5),
             limit, type.structure().field(9).type().tensor_type().elem_type());
  std::string difference;
  auto actual = type;
  actual.clear_type_id();
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference), "AWQ invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(
      logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
          logical.shape().dim_size() == 2 && logical.shape().dim(0).has_dim_value() &&
          logical.shape().dim(0).dim_value() == g.n && logical.shape().dim(1).has_dim_value() &&
          logical.shape().dim(1).dim_value() == g.k,
      "AWQ invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "AWQ truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "AWQ requires one loaded encoded record.");
  CheckParameters(g, reinterpret_cast<const uint8_t *>(value.raw_data().data()));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckReconstruction(g, {data, data + g.fields[1].offset, data + g.fields[2].offset});
  return g;
}

} // namespace

EncodedValueProto ImportAwqWeights(const AwqWeights &weights, size_t max_bytes) {
  const auto g = Validate(weights, max_bytes);
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(g.n);
  logical->mutable_shape()->add_dim()->set_dim_value(g.k);
  result.mutable_raw_data()->resize(g.total);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  for (const auto &field : g.fields)
    CopyWords(data + field.offset, weights.tensors.at(field.name).bytes(), field.bytes,
              field.width);
  CheckParameters(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

AwqWeights ExportAwqWeights(const EncodedValueProto &value, size_t max_bytes,
                            const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  AwqWeights result;
  result.variant = g.variant;
  result.in_features = g.k;
  result.out_features = g.n;
  result.group_size = g.group;
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (const auto &field : g.fields) {
    std::vector<uint8_t> bytes(field.bytes);
    CopyWords(bytes.data(), data + field.offset, field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackAwqCodes(const EncodedValueProto &value, size_t max_bytes,
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

Tensor ReconstructAwqWeights(const EncodedValueProto &value, size_t max_bytes,
                             const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::FLOAT, {g.n, g.k},
                std::vector<uint8_t>(g.count * sizeof(float)));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  const std::array<const uint8_t *, 3> fields{data, data + g.fields[1].offset,
                                              data + g.fields[2].offset};
  for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k) {
      const float reconstructed = Reconstructed(g, fields, n, k);
      std::memcpy(result.mutable_bytes() + (n * static_cast<size_t>(g.k) + k) * sizeof(float),
                  &reconstructed, sizeof(float));
    }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
