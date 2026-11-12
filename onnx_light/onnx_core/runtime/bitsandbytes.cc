// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr const char *kName = "onnx_light.bitsandbytes_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
constexpr const char *kBnbRevision = "08a9956b939857ae2c4fbd56753960cc370fa7cf";
constexpr size_t kMaxStateBytes = 4096;

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "bitsandbytes size overflow.");
  return a * b;
}

size_t CeilDiv(size_t n, size_t d) { return n / d + (n % d != 0); }

void CopyWords(uint8_t *destination, const uint8_t *source, size_t bytes, size_t width) {
  std::memcpy(destination, source, bytes);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t offset = 0; offset < bytes; offset += width)
      std::reverse(destination + offset, destination + offset + width);
}

float Float(const uint8_t *bytes, bool native) {
  uint32_t bits = 0;
  if (native)
    std::memcpy(&bits, bytes, 4);
  else
    for (size_t i = 0; i < 4; ++i)
      bits |= uint32_t(bytes[i]) << (8 * i);
  return std::bit_cast<float>(bits);
}

float Rounded(float value) {
  volatile float rounded = value;
  EXT_ENFORCE_INVALID(std::isfinite(rounded), "bitsandbytes nonfinite reconstruction.");
  return rounded;
}

struct State {
  int64_t format = -1, rows = 0, columns = 0, block = 0, nested_block = 0;
  int32_t dtype = TensorProto::FLOAT;
  float offset = 0;
};

// QuantState.as_dict(packed=True) emits a small ASCII JSON object, not a general
// checkpoint document. Bound it before parsing and accept only its fixed fields.
class StateParser {
public:
  explicit StateParser(std::string_view text) : text_(text) {}

  State Parse() {
    State result;
    std::unordered_set<std::string_view> keys;
    Expect('{');
    do {
      const auto key = String();
      EXT_ENFORCE_INVALID(keys.insert(key).second, "bitsandbytes duplicate state field.");
      Expect(':');
      if (key == "quant_type") {
        const auto value = String();
        EXT_ENFORCE_INVALID(value == "nf4" || value == "fp4",
                            "bitsandbytes unsupported quant_type.");
        result.format = value == "nf4" ? 0 : 1;
      } else if (key == "blocksize")
        result.block = Integer();
      else if (key == "dtype") {
        const auto value = String();
        EXT_ENFORCE_INVALID(value == "float32" || value == "float16" || value == "bfloat16",
                            "bitsandbytes unsupported source dtype.");
        result.dtype = value == "float32"   ? TensorProto::FLOAT
                       : value == "float16" ? TensorProto::FLOAT16
                                            : TensorProto::BFLOAT16;
      } else if (key == "shape") {
        Expect('[');
        result.rows = Integer();
        Expect(',');
        result.columns = Integer();
        Expect(']');
      } else if (key == "nested_blocksize")
        result.nested_block = Integer();
      else if (key == "nested_dtype") {
        EXT_ENFORCE_INVALID(String() == "float32", "bitsandbytes unsupported nested dtype.");
      } else if (key == "nested_offset") {
        const auto token = Number();
        double value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        EXT_ENFORCE_INVALID(parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size() &&
                                std::isfinite(value) && value >= 0 &&
                                value <= std::numeric_limits<float>::max(),
                            "bitsandbytes invalid nested offset.");
        result.offset = static_cast<float>(value);
      } else
        EXT_THROW_INVALID("bitsandbytes unsupported state field: ", std::string(key), ".");
    } while (Take(','));
    Expect('}');
    Space();
    EXT_ENFORCE_INVALID(position_ == text_.size(), "bitsandbytes trailing state data.");
    for (auto key : {"quant_type", "blocksize", "dtype", "shape"})
      EXT_ENFORCE_INVALID(keys.contains(key), "bitsandbytes missing state field: ", key, ".");
    const bool nested = keys.contains("nested_blocksize") || keys.contains("nested_dtype") ||
                        keys.contains("nested_offset");
    EXT_ENFORCE_INVALID(!nested ||
                            (keys.contains("nested_blocksize") && keys.contains("nested_dtype") &&
                             keys.contains("nested_offset") && result.nested_block == 256),
                        "bitsandbytes incomplete or unsupported nested state.");
    return result;
  }

private:
  std::string_view text_;
  size_t position_ = 0;

  void Space() {
    while (position_ < text_.size() && (text_[position_] == ' ' || text_[position_] == '\n' ||
                                        text_[position_] == '\r' || text_[position_] == '\t'))
      ++position_;
  }

  bool Take(char c) {
    Space();
    if (position_ == text_.size() || text_[position_] != c)
      return false;
    ++position_;
    return true;
  }

  void Expect(char c) { EXT_ENFORCE_INVALID(Take(c), "bitsandbytes malformed JSON state."); }

  std::string_view String() {
    Expect('"');
    const size_t start = position_;
    while (position_ < text_.size() && text_[position_] != '"') {
      const char c = text_[position_++];
      EXT_ENFORCE_INVALID((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_',
                          "bitsandbytes unsupported state string.");
    }
    const size_t end = position_;
    Expect('"');
    return text_.substr(start, end - start);
  }

  std::string_view Number() {
    Space();
    const size_t start = position_;
    Take('-');
    EXT_ENFORCE_INVALID(position_ < text_.size() && text_[position_] >= '0' &&
                            text_[position_] <= '9',
                        "bitsandbytes malformed state number.");
    if (text_[position_] == '0')
      ++position_;
    else
      Digits();
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      Digits();
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-'))
        ++position_;
      Digits();
    }
    return text_.substr(start, position_ - start);
  }

  void Digits() {
    const size_t start = position_;
    while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9')
      ++position_;
    EXT_ENFORCE_INVALID(position_ != start, "bitsandbytes malformed state number.");
  }

  int64_t Integer() {
    const auto token = Number();
    int64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    EXT_ENFORCE_INVALID(parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size(),
                        "bitsandbytes invalid state integer.");
    return value;
  }
};

struct Field {
  const char *name;
  int32_t type;
  Shape shape;
  size_t width, bytes, offset;
};

struct Geometry {
  State state;
  size_t count, blocks, json_size, total = 0;
  std::vector<Field> fields;

  Geometry(State metadata, size_t state_bytes, size_t limit)
      : state(metadata), json_size(state_bytes) {
    EXT_ENFORCE_INVALID(
        limit > 0 && state.rows > 0 && state.columns > 0 &&
            static_cast<uint64_t>(state.rows) <= std::numeric_limits<size_t>::max() &&
            static_cast<uint64_t>(state.columns) <= std::numeric_limits<size_t>::max(),
        "bitsandbytes invalid dimensions or max_bytes.");
    count = Product(static_cast<size_t>(state.rows), static_cast<size_t>(state.columns));
    EXT_ENFORCE_INVALID(count <= limit / sizeof(float),
                        "bitsandbytes logical FLOAT weights exceed max_bytes.");
    if (state.format == 2) {
      EXT_ENFORCE_INVALID(state.block == 0 && state.nested_block == 0 && json_size == 0 &&
                              state.dtype == TensorProto::FLOAT,
                          "bitsandbytes invalid INT8 metadata.");
      blocks = static_cast<size_t>(state.rows);
      Add("weight", TensorProto::INT8, {state.rows, state.columns}, 1, limit);
      Add("weight_scb", TensorProto::FLOAT, {state.rows}, 4, limit);
    } else {
      EXT_ENFORCE_INVALID((state.format == 0 || state.format == 1) &&
                              (state.dtype == TensorProto::FLOAT ||
                               state.dtype == TensorProto::FLOAT16 ||
                               state.dtype == TensorProto::BFLOAT16) &&
                              (state.block == 32 || state.block == 64 || state.block == 128 ||
                               state.block == 256 || state.block == 512 || state.block == 1024 ||
                               state.block == 2048 || state.block == 4096) &&
                              (state.nested_block == 0 || state.nested_block == 256) &&
                              json_size > 0 && json_size <= kMaxStateBytes,
                          "bitsandbytes unsupported four-bit state.");
      blocks = CeilDiv(count, static_cast<size_t>(state.block));
      Add("weight", TensorProto::UINT8, {static_cast<int64_t>(CeilDiv(count, 2)), 1}, 1, limit);
      Add("weight_absmax", state.nested_block ? TensorProto::UINT8 : TensorProto::FLOAT,
          {static_cast<int64_t>(blocks)}, state.nested_block ? 1 : 4, limit);
      Add("weight_quant_map", TensorProto::FLOAT, {16}, 4, limit);
      if (state.nested_block) {
        Add("weight_nested_absmax", TensorProto::FLOAT,
            {static_cast<int64_t>(CeilDiv(blocks, 256))}, 4, limit);
        Add("weight_nested_quant_map", TensorProto::FLOAT, {256}, 4, limit);
      }
      Add("weight_quant_state", TensorProto::UINT8, {static_cast<int64_t>(json_size)}, 1, limit);
    }
  }

  void Add(const char *name, int32_t type, Shape shape, size_t width, size_t limit) {
    size_t bytes = width;
    for (int64_t dim : shape)
      bytes = Product(bytes, static_cast<size_t>(dim));
    EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                        "bitsandbytes checkpoint exceeds max_bytes.");
    fields.push_back({name, type, std::move(shape), width, bytes, total});
    total += bytes;
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
  number("format", g.state.format);
  number("out_features", g.state.rows);
  number("in_features", g.state.columns);
  number("block_size", g.state.block);
  number("nested_block_size", g.state.nested_block);
  number("source_dtype", g.state.dtype);
  number("state_bytes", static_cast<int64_t>(g.json_size));
  for (const auto &[name, value] :
       {std::pair{"gptqmodel_revision", kRevision}, {"bitsandbytes_revision", kBnbRevision}}) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(TensorProto::UINT8);
    field->mutable_constant()->add_dims(std::strlen(value));
    field->mutable_constant()->set_raw_data(value);
  }
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

const Tensor &Require(const BitsAndBytesWeights &weights, const char *name) {
  auto found = weights.tensors.find(name);
  EXT_ENFORCE_INVALID(found != weights.tensors.end(), "bitsandbytes missing tensor: ", name, ".");
  return found->second;
}

void CheckTensor(const Tensor &tensor, const Field &field) {
  EXT_ENFORCE_INVALID(tensor.data_type == field.type && tensor.shape == field.shape &&
                          tensor.size_bytes() == field.bytes && tensor.bytes() != nullptr,
                      "bitsandbytes invalid tensor type, shape or byte extent: ", field.name, ".");
}

const std::array<float, 16> &Codebook(int64_t format) {
  static const std::array<float, 16> nf4 = [] {
    const auto plan = MakeQuantizationPlan(QuantizationFormat::kNf4, 16);
    std::array<float, 16> table;
    const auto &source = plan.runs[0].blocks[0].codebook;
    std::transform(source.begin(), source.end(), table.begin(),
                   [](double value) { return static_cast<float>(value); });
    return table;
  }();
  // bitsandbytes get_4bit_type("fp4"), normalized by 12, in native index order.
  static constexpr std::array<float, 16> fp4{
      0, 1.f / 192,  2.f / 3,  1,  1.f / 3,  0.5f,  1.f / 6,  0.25f,
      0, -1.f / 192, -2.f / 3, -1, -1.f / 3, -0.5f, -1.f / 6, -0.25f};
  return format == 0 ? nf4 : fp4;
}

template <typename Get> float Scale(const Geometry &g, Get get, size_t block, bool native) {
  if (g.state.format == 2)
    return Float(get(1) + block * 4, native);
  if (!g.state.nested_block)
    return Float(get(1) + block * 4, native);
  const auto code = get(1)[block];
  const float value = Float(get(4) + code * 4, native);
  const float maximum = Float(get(3) + (block / 256) * 4, native);
  return Rounded(Rounded(value * maximum) + g.state.offset);
}

template <typename Get> void CheckParameters(Geometry &g, Get get, bool native) {
  for (size_t i = 1; i < g.fields.size(); ++i) {
    const auto &field = g.fields[i];
    if (field.type != TensorProto::FLOAT)
      continue;
    const bool maxima = i == 1 || (g.state.nested_block && i == 3);
    for (size_t j = 0; j < field.bytes / 4; ++j) {
      const float value = Float(get(i) + j * 4, native);
      EXT_ENFORCE_INVALID(std::isfinite(value) && (!maxima || value >= 0),
                          "bitsandbytes nonfinite table or invalid absmax.");
      if (g.state.format != 2 && i == 2)
        EXT_ENFORCE_INVALID(value == Codebook(g.state.format)[j],
                            "bitsandbytes noncanonical NF4/FP4 table.");
    }
  }
  if (g.state.format != 2) {
    const State parsed =
        StateParser(
            std::string_view(reinterpret_cast<const char *>(get(g.fields.size() - 1)), g.json_size))
            .Parse();
    EXT_ENFORCE_INVALID(parsed.format == g.state.format && parsed.rows == g.state.rows &&
                            parsed.columns == g.state.columns && parsed.block == g.state.block &&
                            parsed.nested_block == g.state.nested_block &&
                            parsed.dtype == g.state.dtype,
                        "bitsandbytes JSON state disagrees with schema.");
    g.state.offset = parsed.offset;
  }
  // Validate effective nested scales even on import/export, without dense reconstruction.
  if (g.state.nested_block)
    for (size_t block = 0; block < g.blocks; ++block)
      Scale(g, get, block, native);
}

Geometry Validate(const BitsAndBytesWeights &weights, size_t limit) {
  EXT_ENFORCE_INVALID(weights.upstream_revision == kRevision,
                      "bitsandbytes unsupported upstream revision.");
  const auto &weight = Require(weights, "weight");
  State state;
  size_t json_size = 0;
  if (weight.data_type == TensorProto::INT8) {
    EXT_ENFORCE_INVALID(weight.shape.size() == 2, "bitsandbytes INT8 weight must be rank two.");
    state.format = 2;
    state.rows = weight.shape[0];
    state.columns = weight.shape[1];
  } else {
    const auto &json = Require(weights, "weight_quant_state");
    json_size = json.size_bytes();
    EXT_ENFORCE_INVALID(json_size > 0 && json_size <= kMaxStateBytes,
                        "bitsandbytes invalid JSON state size.");
    CheckTensor(json, {"weight_quant_state",
                       TensorProto::UINT8,
                       {static_cast<int64_t>(json_size)},
                       1,
                       json_size,
                       0});
    state = StateParser(std::string_view(reinterpret_cast<const char *>(json.bytes()), json_size))
                .Parse();
  }
  Geometry g(state, json_size, limit);
  EXT_ENFORCE_INVALID(weights.tensors.size() == g.fields.size(),
                      "bitsandbytes missing or unexpected side tensors.");
  for (const auto &field : g.fields)
    CheckTensor(Require(weights, field.name), field);
  CheckParameters(g, [&](size_t i) { return Require(weights, g.fields[i].name).bytes(); }, true);
  return g;
}

Geometry Validate(const EncodedValueProto &value, size_t limit,
                  const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(!value.has_parameter_ref() && value.has_struct_type() &&
                          value.has_logical_type() && value.logical_type().has_tensor_type(),
                      "bitsandbytes requires an independent encoded tensor.");
  const auto &type = catalogue.Resolve(value.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() >= 7,
                      "bitsandbytes unsupported schema or version.");
  auto number = [&](int index) {
    const auto &field = type.structure().field(index);
    EXT_ENFORCE_INVALID(field.has_constant() && field.constant().int64_data_size() == 1,
                        "bitsandbytes malformed schema metadata.");
    return field.constant().int64_data(0);
  };
  EXT_ENFORCE_INVALID(number(5) == TensorProto::FLOAT || number(5) == TensorProto::FLOAT16 ||
                          number(5) == TensorProto::BFLOAT16,
                      "bitsandbytes unsupported source dtype.");
  EXT_ENFORCE_INVALID(number(6) >= 0 && number(6) <= static_cast<int64_t>(kMaxStateBytes),
                      "bitsandbytes invalid state size.");
  Geometry g(
      {number(0), number(1), number(2), number(3), number(4), static_cast<int32_t>(number(5)), 0},
      static_cast<size_t>(number(6)), limit);
  auto actual = type;
  actual.clear_type_id();
  std::string difference;
  EXT_ENFORCE_INVALID(actual.Equals(Schema(g), &difference),
                      "bitsandbytes invalid schema: ", difference);
  const auto &logical = value.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
                          logical.shape().dim_size() == 2 &&
                          logical.shape().dim(0).has_dim_value() &&
                          logical.shape().dim(1).has_dim_value() &&
                          logical.shape().dim(0).dim_value() == g.state.rows &&
                          logical.shape().dim(1).dim_value() == g.state.columns,
                      "bitsandbytes invalid logical type.");
  EXT_ENFORCE_INVALID(value.has_raw_data() && value.raw_data().size() == g.total,
                      "bitsandbytes truncated or oversized payload.");
  const auto storage = catalogue.ValidateEncodedValue(value);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "bitsandbytes requires one loaded encoded record.");
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  CheckParameters(g, [&](size_t i) { return raw + g.fields[i].offset; }, false);
  return g;
}

} // namespace

EncodedValueProto ImportBitsAndBytesWeights(const BitsAndBytesWeights &weights, size_t max_bytes) {
  const Geometry g = Validate(weights, max_bytes);
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(g);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(g.state.rows);
  logical->mutable_shape()->add_dim()->set_dim_value(g.state.columns);
  auto *raw = result.mutable_raw_data();
  raw->resize(g.total);
  for (const auto &field : g.fields)
    CopyWords(reinterpret_cast<uint8_t *>(raw->data()) + field.offset,
              Require(weights, field.name).bytes(), field.bytes, field.width);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

BitsAndBytesWeights ExportBitsAndBytesWeights(const EncodedValueProto &value, size_t max_bytes,
                                              const StructTypeCatalogue &catalogue) {
  const Geometry g = Validate(value, max_bytes, catalogue);
  BitsAndBytesWeights result;
  for (const auto &field : g.fields) {
    std::vector<uint8_t> bytes(field.bytes);
    CopyWords(bytes.data(),
              reinterpret_cast<const uint8_t *>(value.raw_data().data()) + field.offset,
              field.bytes, field.width);
    result.tensors.emplace(field.name,
                           Tensor(field.name, field.type, field.shape, std::move(bytes)));
  }
  return result;
}

Tensor UnpackBitsAndBytesCodes(const EncodedValueProto &value, size_t max_bytes,
                               const StructTypeCatalogue &catalogue) {
  const Geometry g = Validate(value, max_bytes, catalogue);
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  std::vector<uint8_t> codes(g.count);
  for (size_t i = 0; i < g.count; ++i)
    codes[i] = g.state.format == 2 ? raw[i] : (raw[i / 2] >> (i % 2 ? 0 : 4)) & 15;
  return Tensor("codes", g.state.format == 2 ? TensorProto::INT8 : TensorProto::UINT8,
                {g.state.rows, g.state.columns}, std::move(codes));
}

Tensor ReconstructBitsAndBytesWeights(const EncodedValueProto &value, size_t max_bytes,
                                      const StructTypeCatalogue &catalogue,
                                      int32_t reconstruction_dtype) {
  EXT_ENFORCE_INVALID(reconstruction_dtype == TensorProto::FLOAT ||
                          reconstruction_dtype == TensorProto::FLOAT16 ||
                          reconstruction_dtype == TensorProto::BFLOAT16,
                      "bitsandbytes unsupported reconstruction_dtype.");
  const Geometry g = Validate(value, max_bytes, catalogue);
  const auto *raw = reinterpret_cast<const uint8_t *>(value.raw_data().data());
  auto get = [&](size_t i) { return raw + g.fields[i].offset; };
  const size_t width = reconstruction_dtype == TensorProto::FLOAT ? 4 : 2;
  std::vector<uint8_t> bytes(Product(g.count, width));
  for (size_t block = 0; block < g.blocks; ++block) {
    const size_t block_size =
        static_cast<size_t>(g.state.format == 2 ? g.state.columns : g.state.block);
    const float maximum = Scale(g, get, block, false);
    const float scale = g.state.format == 2 ? Rounded(maximum / 127.f) : maximum;
    const size_t end = std::min(g.count, block * block_size + block_size);
    for (size_t i = block * block_size; i < end; ++i) {
      const float code = g.state.format == 2
                             ? float(raw[i] < 128 ? int(raw[i]) : int(raw[i]) - 256)
                             : Codebook(g.state.format)[(raw[i / 2] >> (i % 2 ? 0 : 4)) & 15];
      const float decoded = Rounded(code * scale);
      if (width == 4)
        std::memcpy(bytes.data() + i * width, &decoded, width);
      else {
        const uint16_t bits = reconstruction_dtype == TensorProto::FLOAT16
                                  ? FloatToFloat16Bits(decoded)
                                  : FloatToBfloat16Bits(decoded);
        EXT_ENFORCE_INVALID(
            (bits & (reconstruction_dtype == TensorProto::FLOAT16 ? 0x7c00 : 0x7f80)) !=
                (reconstruction_dtype == TensorProto::FLOAT16 ? 0x7c00 : 0x7f80),
            "bitsandbytes nonfinite reconstruction.");
        std::memcpy(bytes.data() + i * width, &bits, width);
      }
    }
  }
  return Tensor("weight", reconstruction_dtype, {g.state.rows, g.state.columns}, std::move(bytes));
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
