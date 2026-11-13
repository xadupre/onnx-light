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

constexpr const char *kName = "onnx_light.bitblas_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
constexpr const char *kLayout = "row-packed-v1";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "BitBLAS size overflow.");
  return a * b;
}

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  BitblasWeights metadata;
  size_t k, n, groups, count, total = 0;
  bool signed_codes;
  std::vector<Field> fields;

  Geometry(const BitblasWeights &weights, int32_t scale_type, size_t limit) {
    EXT_ENFORCE_INVALID(weights.tensors.empty(), "BitBLAS geometry requires metadata only.");
    metadata = weights;
    EXT_ENFORCE_INVALID(weights.layout == kLayout && weights.upstream_revision == kRevision,
                        "BitBLAS unsupported layout or upstream revision.");
    EXT_ENFORCE_INVALID(weights.method == "gptq" || weights.method == "awq",
                        "BitBLAS requires an explicit GPTQ/AWQ originating method.");
    const auto rows = weights.in_features, columns = weights.out_features;
    const int64_t output_alignment = weights.method == "gptq" ? 32 : 16;
    EXT_ENFORCE_INVALID(limit > 0 && rows > 0 && columns > 0 && rows % 32 == 0 &&
                            columns % output_alignment == 0,
                        "BitBLAS row-packed checkpoint requires aligned, unpadded dimensions.");
    EXT_ENFORCE_INVALID(weights.bits >= 2 && weights.bits <= 8 &&
                            (weights.method == "awq" || weights.bits == 2 || weights.bits == 4 ||
                             weights.bits == 8),
                        "BitBLAS unsupported bit width for the originating method.");
    EXT_ENFORCE_INVALID(weights.group_size == -1 || weights.group_size == rows ||
                            weights.group_size == 32 || weights.group_size == 64 ||
                            weights.group_size == 128,
                        "BitBLAS unsupported group size.");
    const int64_t group = weights.group_size == -1 ? rows : weights.group_size;
    EXT_ENFORCE_INVALID(rows % group == 0, "BitBLAS incomplete group.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(rows) <= std::numeric_limits<size_t>::max() &&
                            static_cast<uint64_t>(columns) <= std::numeric_limits<size_t>::max(),
                        "BitBLAS dimension overflow.");
    k = static_cast<size_t>(rows);
    n = static_cast<size_t>(columns);
    groups = k / static_cast<size_t>(group);
    EXT_ENFORCE_INVALID(groups <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                        "BitBLAS group index overflow.");
    count = Product(k, n);
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "BitBLAS logical FLOAT weights exceed max_bytes.");
    Product(count, weights.bits);
    EXT_ENFORCE_INVALID(scale_type == TensorProto::FLOAT16,
                        "BitBLAS row-packed checkpoint reader requires FP16 scales.");
    signed_codes = weights.method == "gptq" && weights.symmetric;
    Add("qweight", TensorProto::INT8, {columns, rows / 8 * weights.bits}, 1, limit);
    Add("scales", scale_type, {columns, static_cast<int64_t>(groups)}, 2, limit);
    Add("qzeros", TensorProto::INT8,
        signed_codes ? Shape{0} : Shape{static_cast<int64_t>(groups), columns / 8 * weights.bits},
        1, limit);
  }

  void Add(const char *name, int32_t type, Shape shape, size_t width, size_t limit) {
    size_t bytes = width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                        "BitBLAS checkpoint exceeds max_bytes.");
    fields.push_back({name, type, std::move(shape), width, bytes, total});
    total += bytes;
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

void CheckTensor(const Tensor &tensor, const Field &field) {
  EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                          tensor.size_bytes() == field.bytes &&
                          (field.bytes == 0 || tensor.bytes() != nullptr),
                      "BitBLAS invalid tensor type, shape or extent: ", field.name, ".");
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
  auto text = [&](const char *name, const std::string &value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::UINT8);
    field->mutable_constant()->add_dims(value.size());
    field->mutable_constant()->set_raw_data(value);
  };
  text("layout", g.metadata.layout);
  text("method", g.metadata.method);
  number("in_features", g.metadata.in_features);
  number("out_features", g.metadata.out_features);
  number("group_size", g.metadata.group_size);
  number("bits", g.metadata.bits);
  number("symmetric", g.metadata.symmetric);
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

BitblasWeights Metadata(const BitblasWeights &weights) {
  BitblasWeights result;
  result.layout = weights.layout;
  result.method = weights.method;
  result.in_features = weights.in_features;
  result.out_features = weights.out_features;
  result.group_size = weights.group_size;
  result.bits = weights.bits;
  result.symmetric = weights.symmetric;
  result.upstream_revision = weights.upstream_revision;
  return result;
}

Geometry Validate(const BitblasWeights &weights, size_t limit) {
  const auto scale = weights.tensors.find("scales");
  EXT_ENFORCE_INVALID(scale != weights.tensors.end(), "BitBLAS missing scales.");
  Geometry g(Metadata(weights), scale->second.data_type, limit);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(), "BitBLAS unexpected tensors.");
  for (const auto &field : g.fields) {
    const auto found = weights.tensors.find(field.name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end(), "BitBLAS missing ", field.name, ".");
    CheckTensor(found->second, field);
  }
  return g;
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "BitBLAS requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() == 11,
                      "BitBLAS unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "BitBLAS malformed numeric metadata.");
    return field.constant().int64_data(0);
  };
  auto text = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().has_raw_data() &&
                            field.constant().raw_data().size() <= 64,
                        "BitBLAS malformed text metadata.");
    return std::string(field.constant().raw_data());
  };
  BitblasWeights metadata;
  metadata.layout = text(0);
  metadata.method = text(1);
  metadata.in_features = number(2);
  metadata.out_features = number(3);
  metadata.group_size = number(4);
  const int64_t bits = number(5), symmetric = number(6);
  EXT_ENFORCE_INVALID(bits >= 2 && bits <= 8 && (symmetric == 0 || symmetric == 1),
                      "BitBLAS invalid bits or symmetry.");
  metadata.bits = static_cast<uint32_t>(bits);
  metadata.symmetric = symmetric != 0;
  metadata.upstream_revision = text(7);
  Geometry g(metadata, type.structure().field(9).type().tensor_type().elem_type(), limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference),
                      "BitBLAS invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
                          logical.shape().dim_size() == 2 &&
                          logical.shape().dim(0).has_dim_value() &&
                          logical.shape().dim(0).dim_value() == metadata.out_features &&
                          logical.shape().dim(1).has_dim_value() &&
                          logical.shape().dim(1).dim_value() == metadata.in_features,
                      "BitBLAS invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "BitBLAS truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "BitBLAS requires one loaded record.");
  return g;
}

void CheckScales(const Geometry &g, const uint8_t *data) {
  const auto &scales = g.fields[1];
  for (size_t offset = 0; offset < scales.bytes; offset += 2) {
    const auto *p = data + scales.offset + offset;
    const uint16_t bits = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const float scale = Float16BitsToFloat(bits);
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0,
                        "BitBLAS nonfinite or nonpositive scale.");
  }
}

Tensor Codes(const Geometry &g, const uint8_t *data) {
  auto codes = UnpackQuantizationCodes({reinterpret_cast<const char *>(data), g.fields[0].bytes},
                                       g.count, g.metadata.bits);
  const uint8_t mask = (1u << g.metadata.bits) - 1;
  std::vector<uint8_t> bytes(g.count);
  for (size_t i = 0; i < g.count; ++i)
    bytes[i] = static_cast<uint8_t>(codes[i]) & mask;
  return Tensor("codes", TensorProto::UINT8, {g.metadata.out_features, g.metadata.in_features},
                std::move(bytes));
}

// Converts the storage representation to the common GPTQ FLOAT reconstruction contract.
EncodedValueProto Affine(const Geometry &g, const uint8_t *data, size_t limit) {
  auto codes = Codes(g, data);
  const auto bits = g.metadata.bits;
  const uint8_t midpoint = 1u << (bits - 1), mask = (1u << bits) - 1;
  if (g.signed_codes)
    for (size_t i = 0; i < g.count; ++i)
      codes.mutable_bytes()[i] ^= midpoint;
  std::vector<uint8_t> zero_bytes(Product(Product(g.groups, g.n), 2));
  std::string unpacked;
  if (!g.signed_codes)
    unpacked = UnpackQuantizationCodes(
        {reinterpret_cast<const char *>(data + g.fields[2].offset), g.fields[2].bytes},
        Product(g.groups, g.n), bits);
  for (size_t i = 0; i < g.groups * g.n; ++i) {
    const int16_t zero = g.signed_codes ? midpoint : static_cast<uint8_t>(unpacked[i]) & mask;
    std::memcpy(zero_bytes.data() + i * 2, &zero, 2);
  }
  Tensor zeros("zeros", TensorProto::INT16,
               {static_cast<int64_t>(g.groups), g.metadata.out_features}, std::move(zero_bytes));
  std::vector<uint8_t> scale_bytes(g.fields[1].bytes);
  for (size_t n = 0; n < g.n; ++n)
    for (size_t group = 0; group < g.groups; ++group)
      CopyWords(scale_bytes.data() + (group * g.n + n) * 2,
                data + g.fields[1].offset + (n * g.groups + group) * 2, 2, 2);
  Tensor scales("scales", g.fields[1].type,
                {static_cast<int64_t>(g.groups), g.metadata.out_features}, std::move(scale_bytes));
  std::vector<uint8_t> group_bytes(Product(g.k, 4));
  for (size_t k = 0; k < g.k; ++k) {
    const int32_t group = static_cast<int32_t>(k / (g.k / g.groups));
    std::memcpy(group_bytes.data() + k * 4, &group, 4);
  }
  Tensor groups("g_idx", TensorProto::INT32, {g.metadata.in_features}, std::move(group_bytes));
  GptqWeights metadata;
  metadata.in_features = g.metadata.in_features;
  metadata.out_features = g.metadata.out_features;
  metadata.group_size = g.metadata.group_size;
  // Widening the already-decoded AWQ codes avoids imposing GPTQ's 32-column
  // planar zero blocks on the AWQ reader's valid 16-column outputs.
  metadata.bits = g.metadata.method == "awq" ? 8 : bits;
  return PackGptqWeights(metadata, codes, zeros, scales, groups, limit);
}

EncodedValueProto EmptyEncoded(const Geometry &g) {
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(g.metadata.out_features);
  logical->mutable_shape()->add_dim()->set_dim_value(g.metadata.in_features);
  result.mutable_raw_data()->resize(g.total);
  return result;
}

} // namespace

EncodedValueProto ImportBitblasWeights(const BitblasWeights &weights, size_t max_bytes) {
  const auto g = Validate(weights, max_bytes);
  auto result = EmptyEncoded(g);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  for (const auto &field : g.fields)
    CopyWords(data + field.offset, weights.tensors.at(field.name).bytes(), field.bytes,
              field.width);
  CheckScales(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

BitblasWeights ExportBitblasWeights(const EncodedValueProto &value, size_t max_bytes,
                                    const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckScales(g, data);
  auto result = g.metadata;
  for (const auto &field : g.fields) {
    std::vector<uint8_t> bytes(field.bytes);
    CopyWords(bytes.data(), data + field.offset, field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackBitblasCodes(const EncodedValueProto &value, size_t max_bytes,
                          const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckScales(g, data);
  return Codes(g, data);
}

Tensor ReconstructBitblasWeights(const EncodedValueProto &value, size_t max_bytes,
                                 const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckScales(g, data);
  return ReconstructGptqWeights(Affine(g, data, max_bytes), max_bytes);
}

EncodedValueProto PackBitblasWeights(const BitblasWeights &metadata, const Tensor &codes,
                                     const Tensor &scales, const Tensor &qzeros, size_t max_bytes) {
  const Geometry g(metadata, scales.data_type, max_bytes);
  CheckTensor(
      codes,
      {"codes", TensorProto::UINT8, {metadata.out_features, metadata.in_features}, 1, g.count, 0});
  CheckTensor(scales, g.fields[1]);
  CheckTensor(qzeros, g.fields[2]);
  const int mask = (1u << metadata.bits) - 1, midpoint = 1u << (metadata.bits - 1);
  std::string signed_codes(g.count, '\0');
  for (size_t i = 0; i < g.count; ++i) {
    const int code = codes.bytes()[i];
    EXT_ENFORCE_INVALID(code <= mask, "BitBLAS raw code out of range.");
    signed_codes[i] = static_cast<char>(code < midpoint ? code : code - (1u << metadata.bits));
  }
  const auto packed = PackQuantizationCodes(signed_codes, metadata.bits);
  BitblasWeights weights = metadata;
  weights.tensors.emplace("qweight", Tensor("qweight", TensorProto::INT8, g.fields[0].shape,
                                            std::vector<uint8_t>(packed.begin(), packed.end())));
  weights.tensors.emplace("scales", Tensor::Borrow("scales", scales.data_type, scales.shape,
                                                   scales.bytes(), scales.size_bytes()));
  weights.tensors.emplace("qzeros", Tensor::Borrow("qzeros", qzeros.data_type, qzeros.shape,
                                                   qzeros.bytes(), qzeros.size_bytes()));
  return ImportBitblasWeights(weights, max_bytes);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
