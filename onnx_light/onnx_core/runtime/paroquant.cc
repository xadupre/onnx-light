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

constexpr const char *kName = "onnx_light.paroquant_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
constexpr const char *kOrder = "channel_scales,ascending_rotations,awq_gemm,bias";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "ParoQuant size overflow.");
  return a * b;
}

size_t Sum(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(a <= std::numeric_limits<size_t>::max() - b, "ParoQuant size overflow.");
  return a + b;
}

void Budget(size_t bytes, size_t limit) {
  EXT_ENFORCE_INVALID(limit > 0 && bytes <= limit, "ParoQuant exceeds max_bytes.");
}

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t i = 0; i < bytes; i += width)
      std::reverse(destination + i, destination + i + width);
}

template <typename T> T Load(const uint8_t *data, bool native) {
  std::array<uint8_t, sizeof(T)> bytes;
  std::memcpy(bytes.data(), data, bytes.size());
  if constexpr (std::endian::native == std::endian::big)
    if (!native)
      std::reverse(bytes.begin(), bytes.end());
  T result;
  std::memcpy(&result, bytes.data(), bytes.size());
  return result;
}

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  int64_t k, n, group, steps, effective_group;
  size_t total = 0, count, float_bytes;
  std::vector<Field> fields;

  void Add(const char *name, int32_t type, Shape shape, size_t width, size_t limit) {
    size_t bytes = width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    const size_t next = Sum(total, bytes);
    Budget(next, limit);
    fields.push_back({name, type, std::move(shape), width, bytes, total});
    total = next;
  }

  Geometry(int64_t rows, int64_t columns, int64_t grouping, int64_t rotations, int64_t bits,
           int64_t axis, const std::array<int32_t, 4> &types, bool bias, size_t limit)
      : k(rows), n(columns), group(grouping), steps(rotations),
        effective_group(grouping == -1 ? rows : grouping) {
    EXT_ENFORCE_INVALID(k > 0 && n > 0 && n % 8 == 0 && steps > 0 && bits == 4 && axis == 1,
                        "ParoQuant requires positive K,N,krot, N divisible by eight, bits=4 "
                        "and group_axis=1.");
    EXT_ENFORCE_INVALID(effective_group > 0 && effective_group <= 32768 &&
                            effective_group % 2 == 0 && k % effective_group == 0,
                        "ParoQuant requires complete even groups of at most 32768 channels.");
    for (int64_t dim : {k, n, steps})
      EXT_ENFORCE_INVALID(static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max(),
                          "ParoQuant dimension size overflow.");
    count = Product(static_cast<size_t>(k), static_cast<size_t>(n));
    float_bytes = Product(count, sizeof(float));
    Budget(float_bytes, limit);
    EXT_ENFORCE_INVALID(types[0] == TensorProto::FLOAT16 || types[0] == TensorProto::BFLOAT16,
                        "ParoQuant AWQ scales require FLOAT16 or BFLOAT16.");
    auto parameter_width = [](int32_t type) -> size_t {
      EXT_ENFORCE_INVALID(type == TensorProto::FLOAT16 || type == TensorProto::BFLOAT16,
                          "ParoQuant side parameters require FLOAT16 or BFLOAT16.");
      return 2;
    };
    const int64_t groups = k / effective_group;
    Add("qweight", TensorProto::INT32, {k, n / 8}, 4, limit);
    Add("qzeros", TensorProto::INT32, {groups, n / 8}, 4, limit);
    Add("scales", types[0], {groups, n}, 2, limit);
    Add("pairs", TensorProto::INT16, {steps, k}, 2, limit);
    Add("theta", types[1], {steps, k / 2}, parameter_width(types[1]), limit);
    Add("channel_scales", types[2], {1, k}, parameter_width(types[2]), limit);
    if (bias)
      Add("bias", types[3], {n}, parameter_width(types[3]), limit);
    // Allows AWQ conversion storage and FLOAT reconstruction to coexist.
    Budget(Sum(Product(total, 2), float_bytes), limit);
  }
};

float Parameter(const Field &field, const uint8_t *data, size_t index, bool native) {
  const auto bits = Load<uint16_t>(data + index * 2, native);
  return field.type == TensorProto::BFLOAT16 ? Bfloat16BitsToFloat(bits) : Float16BitsToFloat(bits);
}

void CheckSide(const Geometry &g, const std::vector<const uint8_t *> &data, bool native) {
  const size_t group = static_cast<size_t>(g.effective_group);
  std::vector<uint8_t> seen(group);
  const auto &pairs = g.fields[3];
  for (size_t base = 0; base < pairs.bytes / 2; base += group) {
    std::fill(seen.begin(), seen.end(), 0);
    for (size_t j = 0; j < group; ++j) {
      const auto index = Load<int16_t>(data[3] + (base + j) * 2, native);
      EXT_ENFORCE_INVALID(index >= 0 && static_cast<size_t>(index) < group &&
                              !seen[static_cast<size_t>(index)],
                          "ParoQuant pairs must be group-local permutations without duplicates.");
      seen[static_cast<size_t>(index)] = 1;
    }
  }
  for (size_t f = 4; f < g.fields.size(); ++f)
    for (size_t i = 0; i < g.fields[f].bytes / g.fields[f].width; ++i)
      EXT_ENFORCE_INVALID(std::isfinite(Parameter(g.fields[f], data[f], i, native)),
                          "ParoQuant nonfinite side parameter: ", g.fields[f].name, ".");
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
  auto text = [&](const char *name, const char *value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::UINT8);
    field->mutable_constant()->add_dims(std::strlen(value));
    field->mutable_constant()->set_raw_data(value);
  };
  number("in_features", g.k);
  number("out_features", g.n);
  number("group_size", g.group);
  number("krot", g.steps);
  number("bits", 4);
  number("group_axis", 1);
  text("gptqmodel_revision", kRevision);
  text("operation_order", kOrder);
  for (const auto &spec : g.fields) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(spec.name);
    auto *type = field->mutable_type()->mutable_tensor_type();
    type->set_elem_type(spec.type);
    type->mutable_shape();
    for (int64_t dim : spec.shape)
      type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  return result;
}

Tensor MaterializedWeights(const EncodedValueProto &awq, int32_t dtype, size_t limit) {
  auto result = ReconstructAwqWeights(awq, limit);
  for (size_t i = 0; i < result.size_bytes(); i += sizeof(float)) {
    float value = Load<float>(result.bytes() + i, true);
    value = dtype == TensorProto::BFLOAT16 ? Bfloat16BitsToFloat(FloatToBfloat16Bits(value))
                                           : Float16BitsToFloat(FloatToFloat16Bits(value));
    EXT_ENFORCE_INVALID(std::isfinite(value), "ParoQuant materialized AWQ weight overflow.");
    std::memcpy(result.mutable_bytes() + i, &value, sizeof(float));
  }
  return result;
}

EncodedValueProto AwqValue(const Geometry &g, const std::vector<const uint8_t *> &data, bool native,
                           size_t limit) {
  AwqWeights weights;
  weights.variant = "gemm";
  weights.in_features = g.k;
  weights.out_features = g.n;
  weights.group_size = g.group;
  for (size_t i = 0; i < 3; ++i) {
    const auto &field = g.fields[i];
    if constexpr (std::endian::native == std::endian::big) {
      if (!native) {
        std::vector<uint8_t> bytes(field.bytes);
        CopyWords(bytes.data(), data[i], field.bytes, field.width);
        weights.tensors.emplace(field.name,
                                Tensor(field.name, field.type, field.shape, std::move(bytes)));
        continue;
      }
    }
    weights.tensors.emplace(
        field.name, Tensor::Borrow(field.name, field.type, field.shape, data[i], field.bytes));
  }
  auto result = ImportAwqWeights(weights, limit);
  // Releases any big-endian conversion buffers before allocating FLOAT workspace.
  weights.tensors.clear();
  // The pinned AWQ consumer materializes weights in the scales dtype before FLOAT use.
  MaterializedWeights(result, g.fields[2].type, limit);
  return result;
}

Geometry Validate(const ParoQuantWeights &weights, size_t limit) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision,
                      "ParoQuant unsupported upstream revision.");
  std::array<int32_t, 4> types{};
  size_t i = 0;
  for (const char *name : {"scales", "theta", "channel_scales", "bias"}) {
    const auto found = weights.tensors.find(name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end() || i == 3,
                        "ParoQuant missing required tensor: ", name, ".");
    if (found != weights.tensors.end())
      types[i] = found->second.data_type;
    ++i;
  }
  Geometry g(weights.in_features, weights.out_features, weights.group_size, weights.krot,
             weights.bits, weights.group_axis, types, weights.tensors.count("bias") != 0, limit);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(), "ParoQuant unexpected tensors.");
  std::vector<const uint8_t *> data;
  for (const auto &field : g.fields) {
    const auto found = weights.tensors.find(field.name);
    EXT_ENFORCE_INVALID(found != weights.tensors.end(), "ParoQuant missing tensor: ", field.name);
    const auto &tensor = found->second;
    EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                            tensor.size_bytes() == field.bytes && tensor.bytes() != nullptr,
                        "ParoQuant invalid tensor type, shape or byte extent: ", field.name);
    data.push_back(tensor.bytes());
  }
  CheckSide(g, data, true);
  return g;
}

struct Parsed {
  Geometry geometry;
  EncodedValueProto awq;
};

Parsed Parse(const EncodedValueProto &value, size_t limit, const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "ParoQuant requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(
      type.name().value() == kName && type.has_structure() &&
          (type.structure().field_size() == 14 || type.structure().field_size() == 15),
      "ParoQuant unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "ParoQuant malformed metadata.");
    return field.constant().int64_data(0);
  };
  auto dtype = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_type() && field.type().has_tensor_type(),
                        "ParoQuant malformed tensor descriptor.");
    return field.type().tensor_type().elem_type();
  };
  const bool bias = type.structure().field_size() == 15;
  Geometry g(number(0), number(1), number(2), number(3), number(4), number(5),
             {dtype(10), dtype(12), dtype(13), bias ? dtype(14) : TensorProto::UNDEFINED}, bias,
             limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference),
                      "ParoQuant invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(
      logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
          logical.shape().dim_size() == 2 && logical.shape().dim(0).has_dim_value() &&
          logical.shape().dim(0).dim_value() == g.n && logical.shape().dim(1).has_dim_value() &&
          logical.shape().dim(1).dim_value() == g.k,
      "ParoQuant invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "ParoQuant truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "ParoQuant requires one loaded encoded record.");
  std::vector<const uint8_t *> data;
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (const auto &field : g.fields)
    data.push_back(raw + field.offset);
  CheckSide(g, data, false);
  auto awq = AwqValue(g, data, false, limit);
  return {std::move(g), std::move(awq)};
}

} // namespace

EncodedValueProto ImportParoQuantWeights(const ParoQuantWeights &weights, size_t max_bytes) {
  const auto g = Validate(weights, max_bytes);
  std::vector<const uint8_t *> data;
  for (const auto &field : g.fields)
    data.push_back(weights.tensors.at(field.name).bytes());
  auto awq = AwqValue(g, data, true, max_bytes);
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  *result.mutable_logical_type() = awq.logical_type();
  *result.mutable_raw_data() = std::move(*awq.mutable_raw_data());
  result.mutable_raw_data()->resize(g.total);
  auto *raw = reinterpret_cast<uint8_t *>(result.mutable_raw_data()->data());
  for (size_t i = 3; i < g.fields.size(); ++i) {
    const auto &field = g.fields[i];
    CopyWords(raw + field.offset, data[i], field.bytes, field.width);
  }
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

ParoQuantWeights ExportParoQuantWeights(const EncodedValueProto &value, size_t max_bytes,
                                        const StructTypeCatalogue &catalogue) {
  const auto parsed = Parse(value, max_bytes, catalogue);
  const auto &g = parsed.geometry;
  auto awq = ExportAwqWeights(parsed.awq, max_bytes);
  ParoQuantWeights result;
  result.in_features = g.k;
  result.out_features = g.n;
  result.group_size = g.group;
  result.krot = g.steps;
  result.tensors = std::move(awq.tensors);
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t i = 3; i < g.fields.size(); ++i) {
    const auto &field = g.fields[i];
    std::vector<uint8_t> bytes(field.bytes);
    CopyWords(bytes.data(), raw + field.offset, field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackParoQuantCodes(const EncodedValueProto &value, size_t max_bytes,
                            const StructTypeCatalogue &catalogue) {
  const auto parsed = Parse(value, max_bytes, catalogue);
  auto result = UnpackAwqCodes(parsed.awq, max_bytes);
  result.name = value.name().value();
  return result;
}

Tensor ReconstructParoQuantWeights(const EncodedValueProto &value, size_t max_bytes,
                                   const StructTypeCatalogue &catalogue) {
  const auto parsed = Parse(value, max_bytes, catalogue);
  auto result = MaterializedWeights(parsed.awq, parsed.geometry.fields[2].type, max_bytes);
  result.name = value.name().value();
  return result;
}

Tensor ApplyParoQuantLinear(const EncodedValueProto &value, const Tensor &input, size_t max_bytes,
                            const StructTypeCatalogue &catalogue) {
  const auto parsed = Parse(value, max_bytes, catalogue);
  const auto &g = parsed.geometry;
  EXT_ENFORCE_INVALID(
      input.data_type == TensorProto::FLOAT && input.shape.size() >= 2 && input.shape.back() == g.k,
      "ApplyParoQuantLinear requires FLOAT input of rank >=2 with last dimension K.");
  size_t rows = 1;
  for (size_t i = 0; i + 1 < input.shape.size(); ++i) {
    EXT_ENFORCE_INVALID(input.shape[i] >= 0 && static_cast<uint64_t>(input.shape[i]) <=
                                                   std::numeric_limits<size_t>::max(),
                        "ParoQuant negative or overflowing input dimension.");
    rows = Product(rows, static_cast<size_t>(input.shape[i]));
  }
  const size_t activation_bytes = Product(Product(rows, static_cast<size_t>(g.k)), sizeof(float));
  const size_t output_bytes = Product(Product(rows, static_cast<size_t>(g.n)), sizeof(float));
  Budget(Sum(Sum(Product(g.total, 2), g.float_bytes), Sum(activation_bytes, output_bytes)),
         max_bytes);
  EXT_ENFORCE_INVALID(input.size_bytes() == activation_bytes &&
                          (activation_bytes == 0 || input.bytes() != nullptr),
                      "ParoQuant invalid input byte extent.");
  const auto weight = MaterializedWeights(parsed.awq, g.fields[2].type, max_bytes);
  std::vector<float> activation(activation_bytes / sizeof(float));
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  auto parameter = [&](size_t field, size_t index) {
    return Parameter(g.fields[field], raw + g.fields[field].offset, index, false);
  };
  const size_t k = static_cast<size_t>(g.k), n = static_cast<size_t>(g.n);
  const size_t group = static_cast<size_t>(g.effective_group);
  for (size_t row = 0; row < rows; ++row) {
    float *x = activation.data() + row * k;
    for (size_t j = 0; j < k; ++j) {
      const float source = Load<float>(input.bytes() + (row * k + j) * sizeof(float), true);
      EXT_ENFORCE_INVALID(std::isfinite(source), "ParoQuant requires finite FLOAT input.");
      x[j] = source * parameter(5, j);
      EXT_ENFORCE_INVALID(std::isfinite(x[j]),
                          "ParoQuant requires finite input and finite scaled activations.");
    }
    for (size_t step = 0; step < static_cast<size_t>(g.steps); ++step)
      for (size_t base = 0; base < k; base += group)
        for (size_t pair = 0; pair < group / 2; ++pair) {
          const size_t offset = step * k + base + pair * 2;
          const size_t i = base + static_cast<size_t>(
                                      Load<int16_t>(raw + g.fields[3].offset + offset * 2, false));
          const size_t j = base + static_cast<size_t>(Load<int16_t>(
                                      raw + g.fields[3].offset + (offset + 1) * 2, false));
          const float theta = parameter(4, step * (k / 2) + base / 2 + pair);
          const float c = std::cos(theta), s = std::sin(theta);
          const float xi = x[i], xj = x[j];
          x[i] = xi * c + xj * s;
          x[j] = -xi * s + xj * c;
          EXT_ENFORCE_INVALID(std::isfinite(x[i]) && std::isfinite(x[j]),
                              "ParoQuant FLOAT rotation overflow.");
        }
  }
  Shape shape = input.shape;
  shape.back() = g.n;
  Tensor output(std::string(value.name().value()), TensorProto::FLOAT, std::move(shape),
                std::vector<uint8_t>(output_bytes));
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < n; ++column) {
      float sum = 0;
      for (size_t j = 0; j < k; ++j) {
        const float product = activation[row * k + j] *
                              Load<float>(weight.bytes() + (column * k + j) * sizeof(float), true);
        EXT_ENFORCE_INVALID(std::isfinite(product), "ParoQuant FLOAT product overflow.");
        sum += product;
        EXT_ENFORCE_INVALID(std::isfinite(sum), "ParoQuant FLOAT accumulation overflow.");
      }
      if (g.fields.size() == 7)
        sum += parameter(6, column);
      EXT_ENFORCE_INVALID(std::isfinite(sum), "ParoQuant FLOAT linear output overflow.");
      std::memcpy(output.mutable_bytes() + (row * n + column) * sizeof(float), &sum, sizeof(float));
    }
  return output;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
