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

constexpr const char *kName = "onnx_light.exl3_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
constexpr std::array<const char *, 5> kTensors{"trellis", "suh", "svh", "mcg", "mul1"};

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "EXL3 size overflow.");
  return a * b;
}

uint16_t Word(const uint8_t *bytes) { return uint16_t(bytes[0]) | (uint16_t(bytes[1]) << 8); }

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t offset = 0; offset < bytes; offset += width)
      std::reverse(destination + offset, destination + offset + width);
}

float Rounded(float value) {
  volatile float rounded = value;
  EXT_ENFORCE_INVALID(std::isfinite(rounded), "EXL3 nonfinite reconstruction.");
  return rounded;
}

struct Geometry {
  int64_t k, n;
  uint32_t bits, book;
  size_t count, packed, total;

  Geometry(int64_t rows, int64_t columns, int64_t rate, int64_t codebook, size_t limit)
      : k(rows), n(columns) {
    EXT_ENFORCE_INVALID(limit > 0, "EXL3 max_bytes must be positive.");
    EXT_ENFORCE_INVALID(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0,
                        "EXL3 dimensions must be positive multiples of 128.");
    EXT_ENFORCE_INVALID(rate >= 1 && rate <= 8, "EXL3 unsupported trellis rate.");
    EXT_ENFORCE_INVALID(codebook >= 0 && codebook <= 2, "EXL3 unsupported codebook.");
    bits = static_cast<uint32_t>(rate);
    book = static_cast<uint32_t>(codebook);
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(k) <= std::numeric_limits<size_t>::max() &&
                            static_cast<uint64_t>(n) <= std::numeric_limits<size_t>::max(),
                        "EXL3 dimension size overflow.");
    count = Product(static_cast<size_t>(k), static_cast<size_t>(n));
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "EXL3 logical FLOAT weights exceed max_bytes.");
    packed = Product(count / 8, bits);
    const size_t scales = Product(static_cast<size_t>(k), 2) + Product(static_cast<size_t>(n), 2);
    EXT_ENFORCE_INVALID(packed <= limit && scales <= limit - packed &&
                            (book == 0 || limit - packed - scales >= 4),
                        "EXL3 checkpoint exceeds max_bytes.");
    total = packed + scales + (book == 0 ? 0 : 4);
  }
};

StructTypeProto Schema(const Geometry &g) {
  StructTypeProto result;
  result.set_name(kName);
  auto constant = [&](const char *name, int64_t value) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::INT64);
    field->mutable_constant()->add_int64_data(value);
  };
  constant("bits", g.bits);
  constant("codebook", g.book);
  constant("hadamard_size", 128);
  auto *revision = result.mutable_structure()->add_field();
  revision->set_name("gptqmodel_revision");
  revision->mutable_constant()->set_data_type(TensorProto::UINT8);
  revision->mutable_constant()->add_dims(std::strlen(kRevision));
  revision->mutable_constant()->set_raw_data(kRevision);
  auto array = [&](const char *name, int32_t type, std::initializer_list<int64_t> shape) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    auto *tensor = field->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    tensor->mutable_shape();
    for (int64_t dim : shape)
      tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  array("trellis", TensorProto::INT16, {g.k / 16, g.n / 16, 16 * g.bits});
  array("suh", TensorProto::FLOAT16, {g.k});
  array("svh", TensorProto::FLOAT16, {g.n});
  if (g.book)
    array(kTensors[2 + g.book], TensorProto::INT32, {});
  return result;
}

const Tensor &Require(const Exl3Weights &weights, const char *name) {
  auto found = weights.tensors.find(name);
  EXT_ENFORCE_INVALID(found != weights.tensors.end(), "EXL3 missing tensor ", name, ".");
  return found->second;
}

void CheckTensor(const Tensor &tensor, int32_t type, const Shape &shape, size_t bytes) {
  EXT_ENFORCE_INVALID(tensor.data_type == type && tensor.shape == shape &&
                          tensor.size_bytes() == bytes && tensor.bytes() != nullptr,
                      "EXL3 invalid tensor type, shape or byte extent.");
}

void CheckScales(const uint8_t *bytes, size_t count, bool native = false) {
  for (size_t i = 0; i < count; ++i) {
    uint16_t bits = Word(bytes + i * 2);
    if (native)
      std::memcpy(&bits, bytes + i * 2, 2);
    EXT_ENFORCE_INVALID(std::isfinite(Float16BitsToFloat(bits)), "EXL3 nonfinite scale.");
  }
}

void CheckMarker(const uint8_t *bytes, uint32_t book, bool native = false) {
  if (!book)
    return;
  uint32_t value = uint32_t(Word(bytes)) | (uint32_t(Word(bytes + 2)) << 16);
  if (native)
    std::memcpy(&value, bytes, 4);
  EXT_ENFORCE_INVALID(value == (book == 1 ? 0xcbac1fedu : 0x83dcd12du),
                      "EXL3 invalid codebook marker.");
}

Geometry Validate(const Exl3Weights &weights, size_t limit) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision,
                      "EXL3 unsupported upstream revision.");
  const Tensor &trellis = Require(weights, "trellis");
  EXT_ENFORCE_INVALID(trellis.shape.size() == 3 && trellis.shape[0] > 0 && trellis.shape[1] > 0 &&
                          trellis.shape[0] <= std::numeric_limits<int64_t>::max() / 16 &&
                          trellis.shape[1] <= std::numeric_limits<int64_t>::max() / 16 &&
                          trellis.shape[2] % 16 == 0,
                      "EXL3 invalid trellis dimensions.");
  const uint32_t book = weights.tensors.contains("mcg")    ? 1
                        : weights.tensors.contains("mul1") ? 2
                                                           : 0;
  const Geometry g(trellis.shape[0] * 16, trellis.shape[1] * 16, trellis.shape[2] / 16, book,
                   limit);
  EXT_ENFORCE_INVALID(weights.tensors.size() == (book ? 4 : 3),
                      "EXL3 unexpected side tensors or conflicting codebooks.");
  CheckTensor(trellis, TensorProto::INT16, {g.k / 16, g.n / 16, 16 * g.bits}, g.packed);
  for (size_t i = 1; i <= 2; ++i) {
    const Tensor &tensor = Require(weights, kTensors[i]);
    const size_t count = static_cast<size_t>(i == 1 ? g.k : g.n);
    CheckTensor(tensor, TensorProto::FLOAT16, {static_cast<int64_t>(count)}, count * 2);
    CheckScales(tensor.bytes(), count, true);
  }
  if (book) {
    const Tensor &marker = Require(weights, kTensors[2 + book]);
    CheckTensor(marker, TensorProto::INT32, {}, 4);
    CheckMarker(marker.bytes(), book, true);
  }
  return g;
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "EXL3 requires an independent encoded tensor.");
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
                          logical.shape().dim_size() == 2 &&
                          logical.shape().dim(0).has_dim_value() &&
                          logical.shape().dim(1).has_dim_value(),
                      "EXL3 invalid logical type.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() >= 2,
                      "EXL3 unsupported schema or version.");
  auto number = [&](size_t index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "EXL3 malformed trellis/codebook metadata.");
    return field.constant().int64_data(0);
  };
  const Geometry g(logical.shape().dim(0).dim_value(), logical.shape().dim(1).dim_value(),
                   number(0), number(1), limit);
  std::string difference;
  auto actual = type;
  actual.clear_type_id();
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference), "EXL3 invalid schema: ", difference);
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "EXL3 truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "EXL3 requires one loaded encoded record.");
  const auto *bytes = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckScales(bytes + g.packed, static_cast<size_t>(g.k) + static_cast<size_t>(g.n));
  CheckMarker(bytes + g.total - (g.book ? 4 : 0), g.book);
  return g;
}

float Codebook(uint32_t state, uint32_t book) {
  uint32_t raw;
  float result;
  if (book < 2) {
    raw = book == 0 ? state * 89226354u + 64248484u : state * 0xcbac1fedu;
    raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
    result = Float16BitsToFloat(static_cast<uint16_t>(raw)) +
             Float16BitsToFloat(static_cast<uint16_t>(raw >> 16));
  } else {
    raw = state * 0x83dcd12du;
    const uint32_t sum = (raw & 255) + ((raw >> 8) & 255) + ((raw >> 16) & 255) + (raw >> 24);
    result = std::fma(Float16BitsToFloat(static_cast<uint16_t>(sum + 0x6400)),
                      Float16BitsToFloat(0x1eee), Float16BitsToFloat(0xc931));
  }
  return Float16BitsToFloat(FloatToFloat16Bits(result));
}

uint16_t State(const uint8_t *stream, uint32_t bits, size_t position) {
  const size_t words = 16 * bits;
  const size_t start = ((position + 1) * bits + 256 * bits - 16) % (256 * bits);
  const size_t word = start / 16, shift = start % 16;
  uint32_t state = Word(stream + (word ^ 1) * 2);
  if (shift)
    state = ((state << shift) | (Word(stream + (((word + 1) % words) ^ 1) * 2) >> (16 - shift))) &
            65535;
  return static_cast<uint16_t>(state);
}

void Hadamard(float *values, size_t stride) {
  for (size_t width = 1; width < 128; width *= 2)
    for (size_t base = 0; base < 128; base += 2 * width)
      for (size_t j = base; j < base + width; ++j) {
        const float a = values[j * stride], b = values[(j + width) * stride];
        values[j * stride] = Rounded(a + b);
        values[(j + width) * stride] = Rounded(a - b);
      }
  for (size_t i = 0; i < 128; ++i)
    values[i * stride] = Rounded(values[i * stride] * 0.08838834764831845f);
}

} // namespace

EncodedValueProto ImportExl3Weights(const Exl3Weights &weights, size_t max_bytes) {
  const Geometry g = Validate(weights, max_bytes);
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(g.k);
  logical->mutable_shape()->add_dim()->set_dim_value(g.n);
  auto *raw = result.mutable_raw_data();
  raw->resize(g.total);
  size_t offset = 0;
  for (size_t i = 0; i < 3 + (g.book != 0); ++i) {
    const auto &tensor = Require(weights, kTensors[i < 3 ? i : 2 + g.book]);
    CopyWords(reinterpret_cast<uint8_t *>(raw->data()) + offset, tensor.bytes(),
              tensor.size_bytes(), tensor.data_type == TensorProto::INT32 ? 4 : 2);
    offset += tensor.size_bytes();
  }
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

Exl3Weights ExportExl3Weights(const EncodedValueProto &value, size_t max_bytes,
                              const StructTypeCatalogue &catalogue) {
  const Geometry g = Validate(value, max_bytes, catalogue);
  Exl3Weights result;
  // The canonical schema fixes tensor order independently of unordered-map iteration.
  const auto schema = Schema(g);
  size_t offset = 0;
  for (int i = 4; i < schema.structure().field_size(); ++i) {
    const auto &field = schema.structure().field(i);
    const auto &tensor = field.type().tensor_type();
    Shape shape;
    size_t bytes = tensor.elem_type() == TensorProto::INT32 ? 4 : 2;
    for (const auto &dim : tensor.shape().dim()) {
      shape.push_back(dim.dim_value());
      bytes = Product(bytes, static_cast<size_t>(dim.dim_value()));
    }
    const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data()) + offset;
    const std::string name(field.name().value());
    std::vector<uint8_t> raw(bytes);
    CopyWords(raw.data(), data, bytes, tensor.elem_type() == TensorProto::INT32 ? 4 : 2);
    result.tensors.emplace(name,
                           Tensor(name, tensor.elem_type(), std::move(shape), std::move(raw)));
    offset += bytes;
  }
  return result;
}

Tensor UnpackExl3States(const EncodedValueProto &value, size_t max_bytes,
                        const StructTypeCatalogue &catalogue) {
  const Geometry g = Validate(value, max_bytes, catalogue);
  Tensor result("", TensorProto::INT16, {g.k / 16, g.n / 16, 256},
                std::vector<uint8_t>(g.count * 2));
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  for (size_t tile = 0; tile < g.count / 256; ++tile)
    for (size_t position = 0; position < 256; ++position) {
      const uint16_t state = State(data + tile * 32 * g.bits, g.bits, position);
      std::memcpy(result.mutable_bytes() + (tile * 256 + position) * 2, &state, 2);
    }
  return result;
}

Tensor ReconstructExl3Weights(const EncodedValueProto &value, size_t max_bytes,
                              const StructTypeCatalogue &catalogue, int32_t reconstruction_dtype) {
  EXT_ENFORCE_INVALID(reconstruction_dtype == TensorProto::FLOAT ||
                          reconstruction_dtype == TensorProto::FLOAT16,
                      "EXL3 reconstruction_dtype must be FLOAT or FLOAT16.");
  const Geometry g = Validate(value, max_bytes, catalogue);
  const bool half = reconstruction_dtype == TensorProto::FLOAT16;
  auto round = [&](float number) {
    return Rounded(half ? Float16BitsToFloat(FloatToFloat16Bits(number)) : number);
  };
  Tensor result(std::string(value.name().value()), TensorProto::FLOAT, {g.k, g.n},
                std::vector<uint8_t>(g.count * sizeof(float)));
  auto *output = reinterpret_cast<float *>(result.mutable_bytes());
  const auto *data = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  const size_t columns = static_cast<size_t>(g.n), words = 16 * g.bits;
  for (size_t tile = 0; tile < g.count / 256; ++tile) {
    const uint8_t *stream = data + tile * words * 2;
    for (size_t position = 0; position < 256; ++position) {
      const uint16_t state = State(stream, g.bits, position);
      const size_t lane = position / 8, item = position % 8;
      const size_t row =
          (tile / (columns / 16)) * 16 + (lane % 4) * 2 + (item & 1) + ((item & 2) ? 8 : 0);
      const size_t col = (tile % (columns / 16)) * 16 + lane / 4 + ((item & 4) ? 8 : 0);
      output[row * columns + col] = Codebook(state, g.book);
    }
  }
  const uint8_t *suh = data + g.packed, *svh = suh + g.k * 2;
  for (size_t row = 0; row < static_cast<size_t>(g.k); row += 128)
    for (size_t col = 0; col < columns; ++col) {
      Hadamard(output + row * columns + col, columns);
      if (half)
        for (size_t j = 0; j < 128; ++j)
          output[(row + j) * columns + col] = round(output[(row + j) * columns + col]);
    }
  for (size_t row = 0; row < static_cast<size_t>(g.k); ++row) {
    const float scale = Float16BitsToFloat(Word(suh + row * 2));
    for (size_t col = 0; col < columns; ++col)
      output[row * columns + col] = round(output[row * columns + col] * scale);
    for (size_t col = 0; col < columns; col += 128) {
      Hadamard(output + row * columns + col, 1);
      if (half)
        for (size_t j = 0; j < 128; ++j)
          output[row * columns + col + j] = round(output[row * columns + col + j]);
    }
    for (size_t col = 0; col < columns; ++col)
      output[row * columns + col] =
          round(output[row * columns + col] * Float16BitsToFloat(Word(svh + col * 2)));
  }
  if (half)
    return FloatToFloat16Tensor(result.name, result);
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
