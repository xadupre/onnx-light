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

constexpr const char *kName = "onnx_light.gptq_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "GPTQ size overflow.");
  return a * b;
}

size_t CeilDiv(size_t n, size_t d) { return n / d + (n % d != 0); }

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  if (bytes == 0)
    return;
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t offset = 0; offset < bytes; offset += width)
      std::reverse(destination + offset, destination + offset + width);
}

uint64_t Word(const uint8_t *data, size_t width) {
  uint64_t result = 0;
  for (size_t i = 0; i < width; ++i)
    result |= uint64_t(data[i]) << (8 * i);
  return result;
}

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  std::string format;
  int64_t k, n, group, bits;
  int32_t word_type, scale_type;
  size_t width, word_bits, count, groups, weight_rows, zero_columns, total = 0;
  bool planar;
  std::vector<Field> fields;

  void Add(const char *name, int32_t type, Shape shape, size_t element_width, size_t limit) {
    size_t bytes = element_width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                        "GPTQ checkpoint exceeds max_bytes.");
    fields.push_back({name, type, shape, element_width, bytes, total});
    total += bytes;
  }

  Geometry(std::string kind, int64_t rows, int64_t columns, int64_t grouping, int64_t code_bits,
           int64_t axis, int64_t storage_type, int32_t parameter_type, size_t limit)
      : format(std::move(kind)), k(rows), n(columns), group(grouping), bits(code_bits),
        word_type(static_cast<int32_t>(storage_type)), scale_type(parameter_type) {
    EXT_ENFORCE_INVALID(format == "gptq" || format == "gptq_v2" || format == "gptq_p",
                        "GPTQ unsupported format.");
    EXT_ENFORCE_INVALID(limit > 0 && k > 0 && n > 0 && bits >= 2 && bits <= 8 && axis == 1,
                        "GPTQ invalid dimensions, bits or group axis.");
    EXT_ENFORCE_INVALID(group == -1 || group > 0, "GPTQ invalid group size.");
    EXT_ENFORCE_INVALID(storage_type == TensorProto::INT8 || storage_type == TensorProto::INT16 ||
                            storage_type == TensorProto::INT32 || storage_type == TensorProto::INT64,
                        "GPTQ unsupported packed word dtype.");
    width = storage_type == TensorProto::INT8    ? 1
            : storage_type == TensorProto::INT16 ? 2
            : storage_type == TensorProto::INT32 ? 4
                                                 : 8;
    word_bits = width * 8;
    EXT_ENFORCE_INVALID(format != "gptq_p" || width == 4,
                        "GPTQ planar checkpoint format requires INT32 words.");
    EXT_ENFORCE_INVALID(format == "gptq_p" || bits == 2 || bits == 3 || bits == 4 || bits == 8,
                        "GPTQ five-, six- and seven-bit codes require the gptq_p format.");
    planar = bits == 5 || bits == 6 || bits == 7 || (format == "gptq_p" && bits == 3);
    if (bits == 3 || planar)
      EXT_ENFORCE_INVALID(width == 4 && k % 32 == 0 && n % 32 == 0,
                          "GPTQ three-bit and planar layouts require INT32 and 32-code blocks.");
    EXT_ENFORCE_INVALID(scale_type == TensorProto::FLOAT16 || scale_type == TensorProto::BFLOAT16 ||
                            scale_type == TensorProto::FLOAT,
                        "GPTQ unsupported scales dtype.");
    EXT_ENFORCE_INVALID(
        static_cast<uint64_t>(k) <= std::numeric_limits<size_t>::max() &&
            static_cast<uint64_t>(n) <= std::numeric_limits<size_t>::max() &&
            (group == -1 || static_cast<uint64_t>(group) <= std::numeric_limits<size_t>::max()),
        "GPTQ dimension overflow.");
    const size_t kk = static_cast<size_t>(k), nn = static_cast<size_t>(n);
    count = Product(kk, nn);
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "GPTQ logical FLOAT weights exceed max_bytes.");
    const size_t effective_group = group == -1 ? kk : static_cast<size_t>(group);
    groups = CeilDiv(kk, effective_group);
    weight_rows = CeilDiv(Product(kk, static_cast<size_t>(bits)), word_bits);
    zero_columns = CeilDiv(Product(nn, static_cast<size_t>(bits)), word_bits);
    EXT_ENFORCE_INVALID(weight_rows <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                            zero_columns <=
                                static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                            groups <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                        "GPTQ packed dimension overflow.");
    EXT_ENFORCE_INVALID(Product(groups, nn) <= limit / sizeof(int16_t),
                        "GPTQ logical zeros exceed max_bytes.");
    Add("qweight", word_type, {static_cast<int64_t>(weight_rows), n}, width, limit);
    Add("qzeros", word_type, {static_cast<int64_t>(groups), static_cast<int64_t>(zero_columns)},
        width, limit);
    Add("scales", scale_type, {static_cast<int64_t>(groups), n},
        scale_type == TensorProto::FLOAT ? 4 : 2, limit);
    Add("g_idx", TensorProto::INT32, {k}, 4, limit);
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
  text("format", g.format);
  number("in_features", g.k);
  number("out_features", g.n);
  number("group_size", g.group);
  number("bits", g.bits);
  number("group_axis", 1);
  number("word_dtype", g.word_type);
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

void CheckTensor(const Tensor &tensor, const Field &field) {
  EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                          tensor.size_bytes() == field.bytes &&
                          (field.bytes == 0 || tensor.bytes() != nullptr),
                      "GPTQ invalid tensor type, shape or byte extent: ", field.name, ".");
}

// Each field contributes contiguous low-to-high bits; planar layouts split those
// bits into whole-word planes within each block of 32 logical codes.
template <typename Callback> void Pieces(const Geometry &g, size_t index, Callback callback) {
  if (!g.planar) {
    const size_t bit = index * static_cast<size_t>(g.bits);
    const size_t shift = bit % g.word_bits;
    const size_t low = std::min(static_cast<size_t>(g.bits), g.word_bits - shift);
    callback(bit / g.word_bits, shift, low, size_t{0});
    if (low != static_cast<size_t>(g.bits))
      callback(bit / g.word_bits + 1, size_t{0}, static_cast<size_t>(g.bits) - low, low);
    return;
  }
  size_t offset = 0, word = (index / 32) * static_cast<size_t>(g.bits);
  for (size_t plane : {size_t{4}, size_t{2}, size_t{1}}) {
    if ((static_cast<size_t>(g.bits) & plane) == 0)
      continue;
    const size_t factor = 32 / plane;
    callback(word + (index % 32) / factor, ((index % 32) % factor) * plane, plane, offset);
    offset += plane;
    word += plane;
  }
}

uint16_t Code(const Geometry &g, const uint8_t *data, size_t index, size_t other, bool zeros) {
  uint16_t result = 0;
  Pieces(g, index, [&](size_t word, size_t shift, size_t bits, size_t offset) {
    const size_t position =
        zeros ? other * g.zero_columns + word : word * static_cast<size_t>(g.n) + other;
    result |= static_cast<uint16_t>(
        ((Word(data + position * g.width, g.width) >> shift) & ((uint64_t{1} << bits) - 1))
        << offset);
  });
  return result;
}

uint16_t Zero(const Geometry &g, const uint8_t *data, size_t group, size_t n) {
  return Code(g, data + g.fields[1].offset, n, group, true) + (g.format == "gptq" ? 1 : 0);
}

float Scale(const Geometry &g, const uint8_t *data, size_t index) {
  const auto &field = g.fields[2];
  const uint64_t bits = Word(data + field.offset + index * field.width, field.width);
  if (g.scale_type == TensorProto::FLOAT) {
    const auto value = static_cast<uint32_t>(bits);
    return std::bit_cast<float>(value);
  }
  return g.scale_type == TensorProto::FLOAT16 ? Float16BitsToFloat(static_cast<uint16_t>(bits))
                                              : Bfloat16BitsToFloat(static_cast<uint16_t>(bits));
}

size_t Group(const Geometry &g, const uint8_t *data, size_t k) {
  return static_cast<size_t>(Word(data + g.fields[3].offset + k * 4, 4));
}

float Reconstructed(const Geometry &g, const uint8_t *data, size_t n, size_t k) {
  const size_t group = Group(g, data, k);
  const float value = (static_cast<float>(Code(g, data, k, n, false)) -
                       static_cast<float>(Zero(g, data, group, n))) *
                      Scale(g, data, group * static_cast<size_t>(g.n) + n);
  EXT_ENFORCE_INVALID(std::isfinite(value), "GPTQ FLOAT reconstruction overflow.");
  return value;
}

void CheckData(const Geometry &g, const uint8_t *data) {
  for (size_t k = 0; k < static_cast<size_t>(g.k); ++k) {
    const size_t group = Group(g, data, k);
    EXT_ENFORCE_INVALID(group <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
                            group < g.groups,
                        "GPTQ invalid group map index.");
  }
  for (size_t i = 0; i < Product(g.groups, static_cast<size_t>(g.n)); ++i) {
    EXT_ENFORCE_INVALID(std::isfinite(Scale(g, data, i)) && Scale(g, data, i) > 0,
                        "GPTQ nonfinite or nonpositive scale.");
    // At the all-ones field, upstream legacy conversion can carry into the
    // adjacent field (or wrap for INT3). Accepts only its unambiguous subset.
    if (g.format == "gptq")
      EXT_ENFORCE_INVALID(
          Code(g, data + g.fields[1].offset, i % static_cast<size_t>(g.n),
               i / static_cast<size_t>(g.n), true) != (uint16_t{1} << g.bits) - 1,
          "GPTQ v1 all-ones zero field has ambiguous upstream conversion semantics.");
  }
  if (g.count != 0 && g.scale_type != TensorProto::FLOAT16)
    for (size_t n = 0; n < static_cast<size_t>(g.n); ++n)
      for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
        Reconstructed(g, data, n, k);
}

Geometry Validate(const GptqWeights &weights, size_t limit) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision,
                      "GPTQ unsupported upstream revision.");
  const auto scales = weights.tensors.find("scales");
  EXT_ENFORCE_INVALID(scales != weights.tensors.end(), "GPTQ missing scales.");
  Geometry g(weights.format, weights.in_features, weights.out_features, weights.group_size,
             weights.bits, weights.group_axis, weights.word_dtype, scales->second.data_type, limit);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(), "GPTQ unexpected tensors.");
  for (const auto &field : g.fields) {
    const auto found = weights.tensors.find(field.name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end(), "GPTQ missing tensor ", field.name, ".");
    CheckTensor(found->second, field);
  }
  return g;
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "GPTQ requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(
      type.name().value() == kName && type.has_structure() &&
          type.structure().field_size() == 12,
      "GPTQ unsupported schema or version (portable GPTQ is not a checkpoint).");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "GPTQ malformed metadata.");
    return field.constant().int64_data(0);
  };
  const auto &format = type.structure().field(0);
  EXT_ENFORCE_INVALID(format.has_constant() && format.constant().has_raw_data() &&
                          format.constant().raw_data().size() <= 16,
                      "GPTQ malformed format.");
  Geometry g(format.constant().raw_data(), number(1), number(2), number(3), number(4), number(5),
             number(6), type.structure().field(10).type().tensor_type().elem_type(), limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference), "GPTQ invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(
      logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
          logical.shape().dim_size() == 2 && logical.shape().dim(0).has_dim_value() &&
          logical.shape().dim(0).dim_value() == g.n && logical.shape().dim(1).has_dim_value() &&
          logical.shape().dim(1).dim_value() == g.k,
      "GPTQ invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "GPTQ truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "GPTQ requires one loaded encoded record.");
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
  return result;
}

void SetCode(const Geometry &g, uint8_t *data, size_t index, size_t other, bool zeros,
             uint16_t code) {
  Pieces(g, index, [&](size_t word, size_t shift, size_t bits, size_t offset) {
    const size_t position =
        zeros ? other * g.zero_columns + word : word * static_cast<size_t>(g.n) + other;
    const uint64_t value = uint64_t((code >> offset) & ((uint16_t{1} << bits) - 1)) << shift;
    for (size_t byte = 0; byte < g.width; ++byte)
      data[position * g.width + byte] |= static_cast<uint8_t>(value >> (byte * 8));
  });
}

} // namespace

EncodedValueProto ImportGptqWeights(const GptqWeights &weights, size_t max_bytes) {
  const auto g = Validate(weights, max_bytes);
  auto result = EmptyEncoded(g);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  for (const auto &field : g.fields)
    if (field.bytes != 0)
      CopyWords(data + field.offset, weights.tensors.at(field.name).bytes(), field.bytes,
                field.width);
  CheckData(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

GptqWeights ExportGptqWeights(const EncodedValueProto &value, size_t max_bytes,
                              const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  GptqWeights result;
  result.format = g.format;
  result.in_features = g.k;
  result.out_features = g.n;
  result.group_size = g.group;
  result.bits = static_cast<uint32_t>(g.bits);
  result.word_dtype = g.word_type;
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (const auto &field : g.fields) {
    std::vector<uint8_t> bytes(field.bytes);
    if (field.bytes != 0)
      CopyWords(bytes.data(), data + field.offset, field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackGptqCodes(const EncodedValueProto &value, size_t max_bytes,
                       const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::UINT8, {g.n, g.k},
                std::vector<uint8_t>(g.count));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t n = 0; g.count != 0 && n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
      result.mutable_bytes()[n * static_cast<size_t>(g.k) + k] =
          static_cast<uint8_t>(Code(g, data, k, n, false));
  return result;
}

Tensor UnpackGptqZeros(const EncodedValueProto &value, size_t max_bytes,
                       const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result("zeros", TensorProto::INT16, {static_cast<int64_t>(g.groups), g.n},
                std::vector<uint8_t>(Product(Product(g.groups, static_cast<size_t>(g.n)), 2)));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t group = 0; group < g.groups; ++group)
    for (size_t n = 0; n < static_cast<size_t>(g.n); ++n) {
      const auto zero = static_cast<int16_t>(Zero(g, data, group, n));
      std::memcpy(result.mutable_bytes() + (group * static_cast<size_t>(g.n) + n) * 2, &zero, 2);
    }
  return result;
}

Tensor ReconstructGptqWeights(const EncodedValueProto &value, size_t max_bytes,
                              const StructTypeCatalogue &catalogue) {
  const auto g = Validate(value, max_bytes, catalogue);
  Tensor result(std::string(value.name().value()), TensorProto::FLOAT, {g.n, g.k},
                std::vector<uint8_t>(g.count * sizeof(float)));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t n = 0; g.count != 0 && n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k) {
      const float reconstructed = Reconstructed(g, data, n, k);
      std::memcpy(result.mutable_bytes() + (n * static_cast<size_t>(g.k) + k) * sizeof(float),
                  &reconstructed, sizeof(float));
    }
  return result;
}

EncodedValueProto PackGptqWeights(const GptqWeights &metadata, const Tensor &codes,
                                  const Tensor &zeros, const Tensor &scales, const Tensor &g_idx,
                                  size_t max_bytes) {
  EXT_ENFORCE_INVALID(metadata.tensors.empty(), "GPTQ packing metadata must not contain tensors.");
  EXT_ENFORCE_INVALID(metadata.upstream_revision == kRevision,
                      "GPTQ unsupported upstream revision.");
  Geometry g(metadata.format, metadata.in_features, metadata.out_features, metadata.group_size,
             metadata.bits, metadata.group_axis, metadata.word_dtype, scales.data_type, max_bytes);
  CheckTensor(codes, {"codes", TensorProto::UINT8, {g.n, g.k}, 1, g.count, 0});
  const size_t zero_count = Product(g.groups, static_cast<size_t>(g.n));
  CheckTensor(zeros, {"zeros",
                      TensorProto::INT16,
                      {static_cast<int64_t>(g.groups), g.n},
                      2,
                      Product(zero_count, 2),
                      0});
  CheckTensor(scales, g.fields[2]);
  CheckTensor(g_idx, g.fields[3]);
  const uint16_t maxq = (uint16_t{1} << g.bits) - 1;
  for (size_t i = 0; i < g.count; ++i)
    EXT_ENFORCE_INVALID(codes.bytes()[i] <= maxq, "GPTQ code is out of range.");
  for (size_t i = 0; i < zero_count; ++i) {
    int16_t zero;
    std::memcpy(&zero, zeros.bytes() + i * 2, 2);
    const int minimum = g.format == "gptq" ? 1 : 0;
    EXT_ENFORCE_INVALID(zero >= minimum && zero <= maxq,
                        "GPTQ zero is not exactly representable in the target format.");
  }
  auto result = EmptyEncoded(g);
  auto *data = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  if (g.fields[2].bytes != 0)
    CopyWords(data + g.fields[2].offset, scales.bytes(), g.fields[2].bytes, g.fields[2].width);
  if (g.fields[3].bytes != 0)
    CopyWords(data + g.fields[3].offset, g_idx.bytes(), g.fields[3].bytes, 4);
  for (size_t n = 0; g.count != 0 && n < static_cast<size_t>(g.n); ++n)
    for (size_t k = 0; k < static_cast<size_t>(g.k); ++k)
      SetCode(g, data, k, n, false, codes.bytes()[n * static_cast<size_t>(g.k) + k]);
  for (size_t group = 0; group < g.groups; ++group)
    for (size_t n = 0; n < static_cast<size_t>(g.n); ++n) {
      int16_t zero;
      std::memcpy(&zero, zeros.bytes() + (group * static_cast<size_t>(g.n) + n) * 2, 2);
      SetCode(g, data + g.fields[1].offset, n, group, true,
              static_cast<uint16_t>(zero - (g.format == "gptq" ? 1 : 0)));
    }
  CheckData(g, data);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
