// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"
#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr const char *kPrefix = "onnx_light.quantization.v1/";

constexpr std::array<std::string_view, 43> kFormatNames = {"int8",
                                                           "int8_per_channel",
                                                           "int4",
                                                           "gptq",
                                                           "awq",
                                                           "eetq",
                                                           "matmulnbits",
                                                           "q2_k",
                                                           "q3_k",
                                                           "q4_k",
                                                           "q5_k",
                                                           "q6_k",
                                                           "hqq",
                                                           "exl2",
                                                           "exl3",
                                                           "nf4",
                                                           "iq4_nl",
                                                           "binary",
                                                           "ternary",
                                                           "tq1_0",
                                                           "tq2_0",
                                                           "bitnet",
                                                           "paretoq",
                                                           "tequila",
                                                           "stq1_0",
                                                           "iq1_s",
                                                           "aqlm",
                                                           "quip_sharp",
                                                           "spqr",
                                                           "squeezellm",
                                                           "log",
                                                           "fp6_llm",
                                                           "fp8_e4m3",
                                                           "mxfp4",
                                                           "mxfp6",
                                                           "nvfp4",
                                                           "quarot",
                                                           "smoothquant",
                                                           "tiled_float",
                                                           "column_major",
                                                           "ort_matmulnbits_int2",
                                                           "ort_matmulnbits_int4",
                                                           "ort_matmulnbits_int8"};
static_assert(static_cast<size_t>(QuantizationFormat::kOrtMatmulnbitsInt8) + 1 ==
              kFormatNames.size());

constexpr uint32_t OrtBits(QuantizationFormat format) {
  switch (format) {
  case QuantizationFormat::kOrtMatmulnbitsInt2:
    return 2;
  case QuantizationFormat::kOrtMatmulnbitsInt4:
    return 4;
  case QuantizationFormat::kOrtMatmulnbitsInt8:
    return 8;
  default:
    return 0;
  }
}

void ValidateOrtDimensions(uint64_t k, uint64_t n, uint64_t block_size) {
  EXT_ENFORCE_INVALID(k > 0 && n > 0 && k <= std::numeric_limits<int64_t>::max() &&
                          n <= std::numeric_limits<int64_t>::max(),
                      "ORT MatMulNBits requires positive K and N dimensions.");
  EXT_ENFORCE_INVALID(block_size >= 16 && block_size <= std::numeric_limits<uint32_t>::max() &&
                          (block_size & (block_size - 1)) == 0,
                      "ORT MatMulNBits block_size must be a power of two in [16, UINT32_MAX].");
}

constexpr double kNf4[] = {-1,
                           -0.6961928009986877,
                           -0.5250730514526367,
                           -0.39491748809814453,
                           -0.28444138169288635,
                           -0.18477343022823334,
                           -0.09105003625154495,
                           0,
                           0.07958029955625534,
                           0.16093020141124725,
                           0.24611230194568634,
                           0.33791524171829224,
                           0.44070982933044434,
                           0.5626170039176941,
                           0.7229568362236023,
                           1};
constexpr double kIq4Nl[] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
constexpr double kBinary[] = {-1, 1};
constexpr double kTernary[] = {-1, 0, 1};
constexpr double kLog[] = {0, 0.125, -0.125, 0.25, -0.25, 0.5, -0.5, 1, -1, 2, -2, 4, -4, 8, -8};

constexpr size_t Product(uint64_t a, uint64_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "Quantization size overflow.");
  return static_cast<size_t>(a * b);
}

constexpr size_t CeilDiv(size_t n, size_t d) { return n / d + (n % d != 0); }

constexpr bool Floating(int32_t type) {
  return type == TensorProto::FLOAT || type == TensorProto::DOUBLE ||
         type == TensorProto::FLOAT16 || type == TensorProto::BFLOAT16;
}

constexpr size_t FloatBytes(int32_t type) {
  EXT_ENFORCE_INVALID(Floating(type), "Quantization requires FLOAT, DOUBLE, FLOAT16 or BFLOAT16.");
  return type == TensorProto::DOUBLE ? 8 : type == TensorProto::FLOAT ? 4 : 2;
}

template <typename T> T Load(const uint8_t *data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

double ReadFloat(const Tensor &tensor, size_t index) {
  const auto *p = tensor.bytes() + index * FloatBytes(tensor.data_type);
  switch (tensor.data_type) {
  case TensorProto::DOUBLE:
    return Load<double>(p);
  case TensorProto::FLOAT:
    return Load<float>(p);
  case TensorProto::FLOAT16:
    return Float16BitsToFloat(Load<uint16_t>(p));
  default:
    return Bfloat16BitsToFloat(Load<uint16_t>(p));
  }
}

void WriteFloat(uint8_t *p, int32_t type, double value) {
  EXT_ENFORCE_INVALID(std::isfinite(value), "Dequantization produced a nonfinite value.");
  if (type == TensorProto::DOUBLE) {
    std::memcpy(p, &value, sizeof(value));
  } else if (type == TensorProto::FLOAT) {
    const float f = static_cast<float>(value);
    EXT_ENFORCE_INVALID(std::isfinite(f), "Dequantization output dtype overflow.");
    std::memcpy(p, &f, sizeof(f));
  } else {
    const uint16_t bits = type == TensorProto::FLOAT16
                              ? FloatToFloat16Bits(static_cast<float>(value))
                              : FloatToBfloat16Bits(static_cast<float>(value));
    EXT_ENFORCE_INVALID(std::isfinite(type == TensorProto::FLOAT16 ? Float16BitsToFloat(bits)
                                                                   : Bfloat16BitsToFloat(bits)),
                        "Dequantization output dtype overflow.");
    std::memcpy(p, &bits, sizeof(bits));
  }
}

struct ByteWriter {
  std::span<uint8_t> data;
  size_t position = 0;

  ByteWriter(utils::ByteSpan &target, size_t size) {
    target.resize(size);
    data = {target.data(), target.size()};
  }

  void Put(uint64_t value, size_t width) {
    EXT_ENFORCE_INVALID(position <= data.size() && width <= data.size() - position,
                        "Quantization payload size exceeded.");
    for (size_t i = 0; i < width; ++i)
      data[position++] = static_cast<uint8_t>((value >> (8 * i)) & 255);
  }
  void PutDouble(double value) { Put(std::bit_cast<uint64_t>(value), 8); }
  void Finish() const {
    EXT_ENFORCE_INVALID(position == data.size(), "Quantization payload size mismatch.");
  }
};

struct ByteReader {
  std::span<const uint8_t> data;
  size_t position = 0;

  uint64_t Get(size_t width) {
    EXT_ENFORCE_INVALID(position <= data.size() && width <= data.size() - position,
                        "Truncated quantization payload.");
    uint64_t value = 0;
    for (size_t i = 0; i < width; ++i)
      value |= uint64_t(static_cast<uint8_t>(data[position++])) << (8 * i);
    return value;
  }
  double GetDouble() { return std::bit_cast<double>(Get(8)); }
};

constexpr size_t CodeCount(const QuantizationBlockLayout &block) {
  return block.method == QuantizationMethod::kCodebook
             ? Product(CeilDiv(block.count, block.vector_size), block.books)
             : block.count;
}

constexpr size_t CodeBytes(const QuantizationBlockLayout &block) {
  if (block.method == QuantizationMethod::kCast)
    return Product(block.count, FloatBytes(block.cast_type));
  if (block.base3)
    return CeilDiv(CodeCount(block), 5);
  return CeilDiv(Product(CodeCount(block), block.bits), 8);
}

constexpr size_t TableSize(const QuantizationBlockLayout &block) {
  return block.method == QuantizationMethod::kCodebook
             ? Product(Product(block.books, block.entries), block.vector_size)
             : 0;
}

constexpr size_t PayloadBytes(const QuantizationPlan &plan) {
  size_t size = 1;
  const auto add = [&size](size_t bytes) {
    EXT_ENFORCE_INVALID(bytes <= std::numeric_limits<size_t>::max() - size,
                        "Quantization size overflow.");
    size += bytes;
  };
  for (size_t count : {plan.permutation.size(), plan.forward.size(), plan.inverse.size(),
                       plan.outliers.size(), plan.outliers.size()})
    add(Product(count, sizeof(double)));
  for (const auto &run : plan.runs) {
    add(Product(run.blocks.size(), 3 * sizeof(double)));
    add(Product(run.blocks.size(), Product(TableSize(run.layout), sizeof(double))));
    add(Product(run.blocks.size(), CodeBytes(run.layout)));
  }
  return size;
}

void ValidateBlock(const QuantizationBlockLayout &layout,
                   const QuantizationBlockParameters &block) {
  EXT_ENFORCE_INVALID(layout.count <= std::numeric_limits<uint32_t>::max(),
                      "A quantization block may contain at most UINT32_MAX elements.");
  EXT_ENFORCE_INVALID(layout.method == QuantizationMethod::kAffine ||
                          layout.method == QuantizationMethod::kCodebook ||
                          layout.method == QuantizationMethod::kCast,
                      "Unknown quantization method.");
  EXT_ENFORCE_INVALID(std::isfinite(block.scale) && block.scale > 0,
                      "Quantization scale must be finite and positive.");
  EXT_ENFORCE_INVALID(std::isfinite(block.zero_point), "Nonfinite quantization zero point.");
  EXT_ENFORCE_INVALID(std::isfinite(block.offset), "Nonfinite quantization offset.");
  EXT_ENFORCE_INVALID(layout.method == QuantizationMethod::kAffine || block.offset == 0,
                      "Only affine quantization accepts an offset.");
  EXT_ENFORCE_INVALID(layout.bits >= 1 && layout.bits <= 16,
                      "Quantization index width must be between 1 and 16.");
  if (layout.method == QuantizationMethod::kCodebook) {
    EXT_ENFORCE_INVALID(layout.books > 0 && layout.entries > 0 && layout.vector_size > 0 &&
                            layout.entries <= (uint64_t{1} << layout.bits),
                        "Invalid codebook dimensions or index width.");
    EXT_ENFORCE_INVALID(block.codebook.size() == TableSize(layout),
                        "Supplied codebook must have books * entries * vector_size values.");
    for (double v : block.codebook)
      EXT_ENFORCE_INVALID(std::isfinite(v), "Codebooks must contain finite values.");
    EXT_ENFORCE_INVALID(block.zero_point == 0, "Codebook zero_point must be zero.");
  } else {
    EXT_ENFORCE_INVALID(block.codebook.empty(), "Only codebook quantization accepts a table.");
  }
  if (layout.method == QuantizationMethod::kAffine) {
    const double lo = layout.signed_codes ? -double(uint64_t{1} << (layout.bits - 1)) : 0;
    const double hi = lo + double(uint64_t{1} << layout.bits) - 1;
    EXT_ENFORCE_INVALID(std::trunc(block.zero_point) == block.zero_point &&
                            block.zero_point >= lo && block.zero_point <= hi,
                        "Affine zero point must be an integer in the code range.");
  }
  EXT_ENFORCE_INVALID(!layout.base3 ||
                          (layout.method == QuantizationMethod::kCodebook && layout.entries == 3 &&
                           layout.books == 1 && layout.vector_size == 1),
                      "Base-3 packing requires a single three-entry scalar codebook.");
  if (layout.method == QuantizationMethod::kCast) {
    FloatBytes(layout.cast_type);
    EXT_ENFORCE_INVALID(block.zero_point == 0, "Cast zero_point must be zero.");
  }
  CodeBytes(layout);
}

void ValidatePlan(const QuantizationPlan &plan, size_t count) {
  QuantizationFormatName(plan.format);
  EXT_ENFORCE_INVALID(plan.matrix_shape.empty(),
                      "Only ORT MatMulNBits profiles accept a matrix_shape.");
  size_t consumed = 0;
  for (const auto &run : plan.runs) {
    EXT_ENFORCE_INVALID(!run.blocks.empty() && run.layout.count > 0,
                        "Quantization runs must contain nonempty blocks.");
    EXT_ENFORCE_INVALID(run.layout.count <= (count - consumed) / run.blocks.size(),
                        "Quantization blocks exceed tensor size.");
    for (const auto &block : run.blocks)
      ValidateBlock(run.layout, block);
    consumed += Product(run.layout.count, run.blocks.size());
  }
  EXT_ENFORCE_INVALID(consumed == count, "Quantization blocks must cover the tensor exactly.");
  EXT_ENFORCE_INVALID(plan.permutation.empty() || plan.permutation.size() == count,
                      "Quantization permutation must cover the tensor.");
  std::unordered_set<int64_t> seen;
  for (int64_t index : plan.permutation)
    EXT_ENFORCE_INVALID(index >= 0 && uint64_t(index) < count && seen.insert(index).second,
                        "Invalid or duplicate permutation index.");
  seen.clear();
  for (int64_t index : plan.outliers)
    EXT_ENFORCE_INVALID(index >= 0 && uint64_t(index) < count && seen.insert(index).second,
                        "Invalid or duplicate outlier index.");
  const size_t n = plan.transform_size;
  EXT_ENFORCE_INVALID(plan.forward.size() == Product(n, n) &&
                          plan.inverse.size() == Product(n, n) && (n == 0 || count % n == 0),
                      "Transform matrices must be square and divide the tensor size.");
  for (double v : plan.forward)
    EXT_ENFORCE_INVALID(std::isfinite(v), "Nonfinite forward transform.");
  for (double v : plan.inverse)
    EXT_ENFORCE_INVALID(std::isfinite(v), "Nonfinite inverse transform.");
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      long double sum = 0;
      for (size_t k = 0; k < n; ++k)
        sum += static_cast<long double>(plan.forward[i * n + k]) * plan.inverse[k * n + j];
      EXT_ENFORCE_INVALID(std::abs(sum - (i == j ? 1.L : 0.L)) <= 1e-6L,
                          "Quantization transforms must be inverse matrices.");
    }
  if (plan.format == QuantizationFormat::kQuarot || plan.format == QuantizationFormat::kQuipSharp ||
      plan.format == QuantizationFormat::kSmoothquant)
    EXT_ENFORCE_INVALID(n != 0, "This profile requires explicit forward/inverse transforms.");
}

void Transform(std::vector<double> &values, const std::vector<double> &matrix, size_t n) {
  if (n == 0)
    return;
  std::vector<double> row(n);
  for (size_t offset = 0; offset < values.size(); offset += n) {
    for (size_t j = 0; j < n; ++j) {
      long double sum = 0;
      for (size_t k = 0; k < n; ++k)
        sum += static_cast<long double>(values[offset + k]) * matrix[k * n + j];
      row[j] = static_cast<double>(sum);
      EXT_ENFORCE_INVALID(std::isfinite(row[j]), "Quantization transform overflow.");
    }
    std::copy(row.begin(), row.end(), values.begin() + offset);
  }
}

using Field = StructTypeProto::Structure::Field;

Field *AddField(StructTypeProto &type, const std::string &name) {
  auto *field = type.mutable_structure()->add_field();
  field->set_name(name);
  return field;
}

void Array(StructTypeProto &type, const std::string &name, int32_t dtype,
           const std::vector<int64_t> &shape) {
  auto *tensor = AddField(type, name)->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  tensor->mutable_shape();
  for (int64_t dim : shape)
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

constexpr std::array<int64_t, 9> BlockHeader(const QuantizationBlockLayout &block) {
  return {int64_t(block.count),        int64_t(block.method), int64_t(block.bits),
          int64_t(block.signed_codes), int64_t(block.books),  int64_t(block.entries),
          int64_t(block.vector_size),  int64_t(block.base3),  int64_t(block.cast_type)};
}

static_assert(Product(7, 4) == 28 && CeilDiv(7, 4) == 2);
static_assert(Floating(TensorProto::FLOAT16) && !Floating(TensorProto::INT8));
static_assert(FloatBytes(TensorProto::DOUBLE) == 8 && FloatBytes(TensorProto::FLOAT) == 4 &&
              FloatBytes(TensorProto::BFLOAT16) == 2);
static_assert([] {
  QuantizationBlockLayout block;
  block.count = 7;
  if (CodeCount(block) != 7 || CodeBytes(block) != 4 || TableSize(block) != 0 ||
      BlockHeader(block)[0] != 7)
    return false;
  block.method = QuantizationMethod::kCodebook;
  block.entries = 3;
  block.base3 = true;
  if (CodeBytes(block) != 2 || TableSize(block) != 3)
    return false;
  block.base3 = false;
  block.books = 2;
  block.vector_size = 4;
  if (CodeCount(block) != 4 || TableSize(block) != 24)
    return false;
  block.method = QuantizationMethod::kCast;
  return CodeBytes(block) == 14;
}());

StructTypeProto Schema(const QuantizationPlan &plan) {
  StructTypeProto root;
  root.set_name(std::string(kPrefix) + std::string(QuantizationFormatName(plan.format)));
  Array(root, "reserved", TensorProto::UINT8, {1});
  Array(root, "permutation", TensorProto::INT64, {int64_t(plan.permutation.size())});
  Array(root, "forward", TensorProto::DOUBLE, {plan.transform_size, plan.transform_size});
  Array(root, "inverse", TensorProto::DOUBLE, {plan.transform_size, plan.transform_size});
  Array(root, "outlier_indices", TensorProto::INT64, {int64_t(plan.outliers.size())});
  Array(root, "outlier_values", TensorProto::DOUBLE, {int64_t(plan.outliers.size())});
  auto *blocks = AddField(root, "blocks")->mutable_type()->mutable_struct_type();
  auto *block_count = AddField(*blocks, "count")->mutable_constant();
  block_count->set_data_type(TensorProto::INT64);
  size_t total_blocks = 0;
  for (const auto &run : plan.runs) {
    EXT_ENFORCE_INVALID(run.blocks.size() <=
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                                total_blocks,
                        "Quantization block count overflow.");
    total_blocks += run.blocks.size();
  }
  block_count->add_int64_data(static_cast<int64_t>(total_blocks));
  size_t first_block = 0;
  for (size_t i = 0; i < plan.runs.size();) {
    const auto &block = plan.runs[i].layout;
    size_t dimension = plan.runs[i].blocks.size();
    size_t end = i + 1;
    while (end < plan.runs.size() && BlockHeader(plan.runs[end].layout) == BlockHeader(block)) {
      dimension += plan.runs[end].blocks.size();
      ++end;
    }
    auto *run = AddField(*blocks, "run_" + std::to_string(first_block))
                    ->mutable_type()
                    ->mutable_struct_type()
                    ->mutable_array();
    run->set_dimension(dimension);
    auto *item = run->mutable_element_type()->mutable_struct_type();
    auto *parameters = AddField(*item, "parameters")->mutable_constant();
    parameters->set_data_type(TensorProto::INT64);
    parameters->add_dims(9);
    for (int64_t p : BlockHeader(block))
      parameters->add_int64_data(p);
    Array(*item, "scale", TensorProto::DOUBLE, {});
    Array(*item, "zero_point", TensorProto::DOUBLE, {});
    Array(*item, "offset", TensorProto::DOUBLE, {});
    Array(*item, "codebook", TensorProto::DOUBLE, {int64_t(TableSize(block))});
    Array(*item, "codes", TensorProto::UINT8, {int64_t(CodeBytes(block))});
    first_block += dimension;
    i = end;
  }
  return root;
}

double NearestEven(double value) {
  const double low = std::floor(value);
  const double fraction = value - low;
  return fraction < 0.5 || (fraction == 0.5 && std::fmod(low, 2) == 0) ? low : low + 1;
}

void Pack(ByteWriter &payload, const std::vector<uint32_t> &codes, uint32_t bits, bool base3) {
  if (base3) {
    for (size_t i = 0; i < codes.size(); i += 5) {
      uint32_t value = 0, multiplier = 1;
      for (size_t j = i; j < std::min(i + 5, codes.size()); ++j, multiplier *= 3)
        value += multiplier * codes[j];
      payload.Put(value, 1);
    }
    return;
  }
  uint64_t buffer = 0;
  uint32_t available = 0;
  for (uint32_t code : codes) {
    buffer |= uint64_t(code) << available;
    available += bits;
    while (available >= 8) {
      payload.Put(buffer & 255, 1);
      buffer >>= 8;
      available -= 8;
    }
  }
  if (available)
    payload.Put(buffer, 1);
}

std::vector<uint32_t> Unpack(ByteReader &payload, size_t count, uint32_t bits, bool base3) {
  std::vector<uint32_t> codes(count);
  if (base3) {
    for (size_t i = 0; i < count; i += 5) {
      uint32_t value = static_cast<uint32_t>(payload.Get(1));
      EXT_ENFORCE_INVALID(value < 243, "Invalid base-3 byte.");
      for (size_t j = i; j < std::min(i + 5, count); ++j) {
        codes[j] = value % 3;
        value /= 3;
      }
      EXT_ENFORCE_INVALID(value == 0, "Nonzero base-3 padding.");
    }
    return codes;
  }
  uint64_t buffer = 0;
  uint32_t available = 0;
  for (auto &code : codes) {
    while (available < bits) {
      buffer |= payload.Get(1) << available;
      available += 8;
    }
    code = static_cast<uint32_t>(buffer & ((uint64_t{1} << bits) - 1));
    buffer >>= bits;
    available -= bits;
  }
  EXT_ENFORCE_INVALID(buffer == 0, "Nonzero quantization code padding.");
  return codes;
}

void EncodeBlock(ByteWriter &payload, const QuantizationBlockLayout &layout,
                 const QuantizationBlockParameters &block, const std::vector<double> &values,
                 size_t offset) {
  payload.PutDouble(block.scale);
  payload.PutDouble(block.zero_point);
  payload.PutDouble(block.offset);
  for (double value : block.codebook)
    payload.PutDouble(value);
  if (layout.method == QuantizationMethod::kCast) {
    for (size_t i = 0; i < layout.count; ++i) {
      const double value = values[offset + i] / block.scale;
      EXT_ENFORCE_INVALID(std::isfinite(value), "Quantization scaling overflow.");
      if (layout.cast_type == TensorProto::DOUBLE)
        payload.PutDouble(value);
      else {
        const float converted = static_cast<float>(value);
        EXT_ENFORCE_INVALID(std::isfinite(converted), "Quantization cast overflow.");
        if (layout.cast_type == TensorProto::FLOAT) {
          payload.Put(std::bit_cast<uint32_t>(converted), 4);
        } else {
          const auto bits = layout.cast_type == TensorProto::FLOAT16
                                ? FloatToFloat16Bits(converted)
                                : FloatToBfloat16Bits(converted);
          EXT_ENFORCE_INVALID(std::isfinite(layout.cast_type == TensorProto::FLOAT16
                                                ? Float16BitsToFloat(bits)
                                                : Bfloat16BitsToFloat(bits)),
                              "Quantization cast overflow.");
          payload.Put(bits, 2);
        }
      }
    }
    return;
  }
  std::vector<uint32_t> codes;
  codes.reserve(CodeCount(layout));
  if (layout.method == QuantizationMethod::kAffine) {
    const int64_t lo = layout.signed_codes ? -(int64_t{1} << (layout.bits - 1)) : 0;
    const int64_t hi = lo + (int64_t{1} << layout.bits) - 1;
    for (size_t i = 0; i < layout.count; ++i) {
      const long double normalized =
          (static_cast<long double>(values[offset + i]) - block.offset) / block.scale +
          block.zero_point;
      const double value = static_cast<double>(
          std::clamp(normalized, static_cast<long double>(lo), static_cast<long double>(hi)));
      const int64_t code = static_cast<int64_t>(NearestEven(value));
      codes.push_back(static_cast<uint32_t>(code) & ((uint32_t{1} << layout.bits) - 1));
    }
  } else {
    std::vector<double> residual(layout.vector_size);
    for (size_t i = 0; i < layout.count; i += layout.vector_size) {
      const size_t width = std::min<size_t>(layout.vector_size, layout.count - i);
      for (size_t j = 0; j < width; ++j) {
        residual[j] = values[offset + i + j] / block.scale;
        EXT_ENFORCE_INVALID(std::isfinite(residual[j]), "Quantization scaling overflow.");
      }
      for (size_t book = 0; book < layout.books; ++book) {
        uint32_t best = 0;
        long double best_error = std::numeric_limits<long double>::infinity();
        const size_t base = book * layout.entries * layout.vector_size;
        for (uint32_t entry = 0; entry < layout.entries; ++entry) {
          long double error = 0;
          for (size_t j = 0; j < width; ++j) {
            const long double diff = static_cast<long double>(residual[j]) -
                                     block.codebook[base + entry * layout.vector_size + j];
            error += diff * diff;
          }
          if (error < best_error) {
            best_error = error;
            best = entry;
          }
        }
        EXT_ENFORCE_INVALID(std::isfinite(best_error), "Codebook distance overflow.");
        codes.push_back(best);
        for (size_t j = 0; j < width; ++j) {
          residual[j] -= block.codebook[base + best * layout.vector_size + j];
          EXT_ENFORCE_INVALID(std::isfinite(residual[j]), "Codebook residual overflow.");
        }
      }
    }
  }
  Pack(payload, codes, layout.bits, layout.base3);
}

void DecodeBlock(ByteReader &payload, const QuantizationBlockLayout &layout,
                 const QuantizationBlockParameters &block, std::vector<double> &values) {
  if (layout.method == QuantizationMethod::kCast) {
    for (size_t i = 0; i < layout.count; ++i) {
      double value;
      if (layout.cast_type == TensorProto::DOUBLE)
        value = payload.GetDouble();
      else if (layout.cast_type == TensorProto::FLOAT)
        value = std::bit_cast<float>(static_cast<uint32_t>(payload.Get(4)));
      else {
        const auto code = static_cast<uint16_t>(payload.Get(2));
        value = layout.cast_type == TensorProto::FLOAT16 ? Float16BitsToFloat(code)
                                                         : Bfloat16BitsToFloat(code);
      }
      values.push_back(value * block.scale);
    }
    return;
  }
  const auto codes = Unpack(payload, CodeCount(layout), layout.bits, layout.base3);
  if (layout.method == QuantizationMethod::kAffine) {
    for (uint32_t code : codes) {
      int64_t signed_code = code;
      if (layout.signed_codes && (code & (uint32_t{1} << (layout.bits - 1))))
        signed_code -= int64_t{1} << layout.bits;
      values.push_back((signed_code - block.zero_point) * block.scale + block.offset);
    }
  } else {
    for (size_t i = 0, group = 0; i < layout.count; i += layout.vector_size, ++group) {
      for (size_t j = 0; j < std::min<size_t>(layout.vector_size, layout.count - i); ++j) {
        double sum = 0;
        for (size_t book = 0; book < layout.books; ++book) {
          const uint32_t entry = codes[group * layout.books + book];
          EXT_ENFORCE_INVALID(entry < layout.entries, "Quantization codebook index out of range.");
          sum += block.codebook[(book * layout.entries + entry) * layout.vector_size + j];
        }
        values.push_back(sum * block.scale);
      }
    }
  }
}

const Field &GetField(const StructTypeProto &type, size_t index, const char *name) {
  EXT_ENFORCE_INVALID(type.has_structure() && index < type.structure().field().size(),
                      "Invalid quantization structure.");
  const auto &field = type.structure().field(index);
  EXT_ENFORCE_INVALID(field.name() == name, "Unexpected quantization field: ", name);
  return field;
}

size_t Extent(const Field &field, int32_t dtype, size_t rank, size_t axis = 0) {
  EXT_ENFORCE_INVALID(field.has_type() && field.type().has_tensor_type(),
                      "Expected tensor field in quantization layout.");
  const auto &type = field.type().tensor_type();
  EXT_ENFORCE_INVALID(type.elem_type() == dtype && type.has_shape() &&
                          type.shape().dim().size() == rank && axis < rank &&
                          type.shape().dim(axis).has_dim_value() &&
                          type.shape().dim(axis).dim_value() >= 0,
                      "Invalid quantization field dtype or dimensions.");
  return static_cast<size_t>(type.shape().dim(axis).dim_value());
}

QuantizationBlockLayout BlockParameters(const StructTypeProto &type) {
  const auto &field = GetField(type, 0, "parameters");
  EXT_ENFORCE_INVALID(field.has_constant(), "Missing quantization parameters.");
  const auto &p = field.constant();
  EXT_ENFORCE_INVALID(p.data_type() == TensorProto::INT64 && p.int64_data().size() == 9 &&
                          p.dims().size() == 1 && p.dims(0) == 9 && !p.has_raw_data(),
                      "Invalid quantization parameters.");
  for (int64_t v : p.int64_data())
    EXT_ENFORCE_INVALID(v >= 0 && uint64_t(v) <= std::numeric_limits<uint32_t>::max(),
                        "Quantization block parameters exceed supported bounds.");
  EXT_ENFORCE_INVALID(p.int64_data(3) <= 1 && p.int64_data(7) <= 1,
                      "Invalid quantization boolean parameter.");
  QuantizationBlockLayout block;
  block.count = p.int64_data(0);
  block.method = static_cast<QuantizationMethod>(p.int64_data(1));
  block.bits = static_cast<uint32_t>(p.int64_data(2));
  block.signed_codes = p.int64_data(3) != 0;
  block.books = static_cast<uint32_t>(p.int64_data(4));
  block.entries = static_cast<uint32_t>(p.int64_data(5));
  block.vector_size = static_cast<uint32_t>(p.int64_data(6));
  block.base3 = p.int64_data(7) != 0;
  block.cast_type = static_cast<int32_t>(p.int64_data(8));
  return block;
}

std::vector<double> ReadDoubles(ByteReader &payload, size_t count) {
  EXT_ENFORCE_INVALID(count <= (payload.data.size() - payload.position) / 8,
                      "Truncated quantization parameter array.");
  std::vector<double> values(count);
  for (double &value : values)
    value = payload.GetDouble();
  return values;
}

std::vector<int64_t> ReadIndices(ByteReader &payload, size_t count) {
  EXT_ENFORCE_INVALID(count <= (payload.data.size() - payload.position) / 8,
                      "Truncated quantization index array.");
  std::vector<int64_t> values(count);
  for (int64_t &value : values)
    value = std::bit_cast<int64_t>(payload.Get(8));
  return values;
}

void SetTable(QuantizationBlockLayout &layout, QuantizationBlockParameters &block,
              std::span<const double> values, uint32_t bits) {
  layout.method = QuantizationMethod::kCodebook;
  layout.signed_codes = false;
  layout.bits = bits;
  layout.entries = static_cast<uint32_t>(values.size());
  block.codebook.assign(values.begin(), values.end());
}

} // namespace

std::string_view QuantizationFormatName(QuantizationFormat format) {
  const size_t index = static_cast<size_t>(format);
  EXT_ENFORCE_INVALID(index < kFormatNames.size(), "Unknown quantization format: ", index);
  return kFormatNames[index];
}

QuantizationFormat ParseQuantizationFormat(std::string_view name) {
  const auto it = std::find(kFormatNames.begin(), kFormatNames.end(), name);
  EXT_ENFORCE_INVALID(it != kFormatNames.end(), "Unknown quantization format: ", std::string(name));
  return static_cast<QuantizationFormat>(it - kFormatNames.begin());
}

QuantizationPlan MakeQuantizationPlan(QuantizationFormat format, uint64_t count,
                                      uint64_t block_size) {
  QuantizationFormatName(format);
  EXT_ENFORCE_INVALID(OrtBits(format) == 0,
                      "Use MakeMatMulNBitsPlan with K and N for ORT MatMulNBits profiles.");
  EXT_ENFORCE_INVALID(block_size > 0 && block_size <= std::numeric_limits<uint32_t>::max(),
                      "Invalid quantization block size.");
  QuantizationPlan plan;
  plan.format = format;
  QuantizationBlockLayout layout;
  QuantizationBlockParameters block;
  if (format == QuantizationFormat::kInt8 || format == QuantizationFormat::kInt8PerChannel ||
      format == QuantizationFormat::kEetq || format == QuantizationFormat::kSmoothquant)
    layout.bits = 8;
  if (format >= QuantizationFormat::kQ2K && format <= QuantizationFormat::kQ6K)
    layout.bits =
        2 + static_cast<uint32_t>(format) - static_cast<uint32_t>(QuantizationFormat::kQ2K);
  if (format == QuantizationFormat::kGptq || format == QuantizationFormat::kAwq ||
      format == QuantizationFormat::kMatmulnbits) {
    layout.signed_codes = false;
    block.zero_point = 8;
  }
  if (format == QuantizationFormat::kNf4)
    SetTable(layout, block, kNf4, 4);
  if (format == QuantizationFormat::kIq4Nl)
    SetTable(layout, block, kIq4Nl, 4);
  if (format == QuantizationFormat::kBinary)
    SetTable(layout, block, kBinary, 1);
  if (format == QuantizationFormat::kTernary || format == QuantizationFormat::kTq10 ||
      format == QuantizationFormat::kTq20 || format == QuantizationFormat::kBitnet ||
      format == QuantizationFormat::kParetoq || format == QuantizationFormat::kTequila) {
    SetTable(layout, block, kTernary, 2);
    layout.base3 = format != QuantizationFormat::kTq20;
  }
  if (format == QuantizationFormat::kStq10 || format == QuantizationFormat::kIq1S ||
      format == QuantizationFormat::kAqlm || format == QuantizationFormat::kQuipSharp) {
    layout.method = QuantizationMethod::kCodebook;
    layout.bits = format == QuantizationFormat::kStq10 ? 5 : 8;
    layout.entries = uint32_t{1} << layout.bits;
    layout.vector_size = format == QuantizationFormat::kStq10 ? 4 : 8;
    layout.books = format == QuantizationFormat::kAqlm ? 2 : 1;
  }
  if (format == QuantizationFormat::kSqueezellm) {
    layout.method = QuantizationMethod::kCodebook;
    layout.entries = 16;
  }
  if (format == QuantizationFormat::kMxfp4 || format == QuantizationFormat::kNvfp4 ||
      format == QuantizationFormat::kMxfp6 || format == QuantizationFormat::kFp6Llm ||
      format == QuantizationFormat::kFp8E4m3) {
    const uint32_t bits =
        format == QuantizationFormat::kFp8E4m3                                          ? 8
        : format == QuantizationFormat::kMxfp6 || format == QuantizationFormat::kFp6Llm ? 6
                                                                                        : 4;
    std::vector<double> table;
    for (uint32_t code = 0; code < (uint32_t{1} << bits); ++code) {
      const double value = bits == 4 ? Float4E2M1NibbleToFloat(static_cast<uint8_t>(code))
                           : bits == 6
                               ? Float6BitsToFloat(static_cast<uint8_t>(code), DataType::FLOAT6E3M2)
                               : Float8E4M3FNBitsToFloat(static_cast<uint8_t>(code));
      if (std::isfinite(value))
        table.push_back(value);
    }
    SetTable(layout, block, table, bits);
  }
  if (format == QuantizationFormat::kLog)
    SetTable(layout, block, kLog, 4);
  if (format == QuantizationFormat::kTiledFloat || format == QuantizationFormat::kColumnMajor) {
    layout.method = QuantizationMethod::kCast;
    layout.cast_type = TensorProto::FLOAT;
  }
  if (count >= block_size) {
    layout.count = block_size;
    plan.runs.push_back(
        {layout, std::vector<QuantizationBlockParameters>(Product(count / block_size, 1), block)});
    count %= block_size;
  }
  if (count != 0) {
    layout.count = std::min(count, block_size);
    plan.runs.push_back({layout, {block}});
  }
  return plan;
}

QuantizationPlan MakeMatMulNBitsPlan(QuantizationFormat format, uint64_t k, uint64_t n,
                                     uint64_t block_size) {
  const uint32_t bits = OrtBits(format);
  EXT_ENFORCE_INVALID(bits != 0, "Expected an ORT MatMulNBits quantization format.");
  ValidateOrtDimensions(k, n, block_size);
  const size_t groups = CeilDiv(k, block_size);
  auto plan = MakeQuantizationPlan(QuantizationFormat::kMatmulnbits,
                                   Product(Product(n, groups), block_size), block_size);
  plan.format = format;
  plan.matrix_shape = {int64_t(k), int64_t(n)};
  plan.runs[0].layout.bits = bits;
  for (auto &block : plan.runs[0].blocks)
    block.zero_point = uint32_t{1} << (bits - 1);
  return plan;
}

namespace {

struct OrtStorage {
  uint64_t k, n, block_size;
  uint32_t bits;
  int32_t type, zero_type;
  size_t groups, blob_size, weight_bytes, scale_bytes, zero_bytes, total_bytes;

  OrtStorage(const Shape &shape, int32_t dtype, QuantizationFormat format, uint64_t size,
             int32_t zero_dtype)
      : block_size(size), bits(OrtBits(format)), type(dtype), zero_type(zero_dtype) {
    EXT_ENFORCE_INVALID(shape.size() == 2 && shape[0] > 0 && shape[1] > 0,
                        "ORT MatMulNBits requires a nonempty rank-two [K,N] tensor.");
    k = shape[0];
    n = shape[1];
    ValidateOrtDimensions(k, n, block_size);
    EXT_ENFORCE_INVALID(bits != 0, "Expected an ORT MatMulNBits quantization format.");
    EXT_ENFORCE_INVALID(type == TensorProto::FLOAT || type == TensorProto::FLOAT16 ||
                            type == TensorProto::BFLOAT16,
                        "ORT MatMulNBits requires FLOAT, FLOAT16 or BFLOAT16 weights.");
    EXT_ENFORCE_INVALID(zero_type == TensorProto::UNDEFINED || zero_type == TensorProto::UINT8 ||
                            zero_type == type,
                        "Invalid ORT MatMulNBits zero-point dtype.");
    groups = CeilDiv(k, block_size);
    blob_size = Product(block_size, bits) / 8;
    weight_bytes = Product(Product(n, groups), blob_size);
    scale_bytes = Product(Product(n, groups), FloatBytes(type));
    zero_bytes = zero_type == TensorProto::UNDEFINED ? 0
                 : zero_type == TensorProto::UINT8   ? Product(n, CeilDiv(groups, 8 / bits))
                                                     : scale_bytes;
    EXT_ENFORCE_INVALID(weight_bytes <= std::numeric_limits<size_t>::max() - scale_bytes &&
                            weight_bytes + scale_bytes <=
                                std::numeric_limits<size_t>::max() - zero_bytes,
                        "ORT MatMulNBits payload size overflow.");
    total_bytes = weight_bytes + scale_bytes + zero_bytes;
  }
};

StructTypeProto OrtSchema(const OrtStorage &storage, QuantizationFormat format) {
  StructTypeProto root;
  root.set_name(std::string(kPrefix) + std::string(QuantizationFormatName(format)));
  auto *parameters = AddField(root, "parameters")->mutable_constant();
  parameters->set_data_type(TensorProto::INT64);
  parameters->add_dims(2);
  parameters->add_int64_data(storage.bits);
  parameters->add_int64_data(storage.block_size);
  Array(root, "B", TensorProto::UINT8,
        {int64_t(storage.n), int64_t(storage.groups), int64_t(storage.blob_size)});
  Array(root, "scales", storage.type, {int64_t(storage.n), int64_t(storage.groups)});
  if (storage.zero_type != TensorProto::UNDEFINED)
    Array(root, "zero_points", storage.zero_type,
          {int64_t(storage.n), int64_t(storage.zero_type == TensorProto::UINT8
                                           ? CeilDiv(storage.groups, 8 / storage.bits)
                                           : storage.groups)});
  return root;
}

double RoundFloat(double value, int32_t type) {
  std::array<uint8_t, sizeof(double)> bytes;
  WriteFloat(bytes.data(), type, value);
  if (type == TensorProto::FLOAT)
    return Load<float>(bytes.data());
  if (type == TensorProto::FLOAT16)
    return Float16BitsToFloat(Load<uint16_t>(bytes.data()));
  return Bfloat16BitsToFloat(Load<uint16_t>(bytes.data()));
}

void PutFloat(ByteWriter &payload, double value, int32_t type) {
  std::array<uint8_t, sizeof(double)> bytes;
  WriteFloat(bytes.data(), type, value);
  const size_t width = FloatBytes(type);
  if (width == 8)
    payload.Put(Load<uint64_t>(bytes.data()), 8);
  else if (width == 4)
    payload.Put(Load<uint32_t>(bytes.data()), 4);
  else
    payload.Put(Load<uint16_t>(bytes.data()), 2);
}

double PackedFloat(std::span<const uint8_t> bytes, int32_t type, size_t index) {
  ByteReader reader{bytes, Product(index, FloatBytes(type))};
  if (type == TensorProto::FLOAT)
    return std::bit_cast<float>(static_cast<uint32_t>(reader.Get(4)));
  const auto code = static_cast<uint16_t>(reader.Get(2));
  return type == TensorProto::FLOAT16 ? Float16BitsToFloat(code) : Bfloat16BitsToFloat(code);
}

void SetLogicalTensor(EncodedValueProto &result, const Tensor &tensor) {
  result.set_name(tensor.name);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(tensor.data_type);
  logical->mutable_shape();
  for (int64_t dim : tensor.shape)
    logical->mutable_shape()->add_dim()->set_dim_value(dim);
  StructTypeCatalogue{}.ValidateEncodedValue(result);
}

EncodedValueProto EncodeOrtTensor(const Tensor &tensor, const QuantizationPlan &plan) {
  EXT_ENFORCE_INVALID(tensor.shape == plan.matrix_shape,
                      "ORT MatMulNBits source shape must match the plan's [K,N].");
  EXT_ENFORCE_INVALID(plan.runs.size() == 1, "ORT MatMulNBits requires one shared block layout.");
  EXT_ENFORCE_INVALID(plan.permutation.empty() && plan.transform_size == 0 &&
                          plan.forward.empty() && plan.inverse.empty() && plan.outliers.empty(),
                      "ORT MatMulNBits does not support permutations, transforms or outliers.");
  const auto &run = plan.runs[0];
  const OrtStorage geometry(tensor.shape, tensor.data_type, plan.format, run.layout.count,
                            TensorProto::UNDEFINED);
  QuantizationBlockLayout expected;
  expected.count = geometry.block_size;
  expected.bits = geometry.bits;
  expected.signed_codes = false;
  EXT_ENFORCE_INVALID(BlockHeader(run.layout) == BlockHeader(expected),
                      "ORT MatMulNBits requires a uniform unsigned affine layout.");
  EXT_ENFORCE_INVALID(run.blocks.size() == Product(geometry.n, geometry.groups),
                      "ORT MatMulNBits requires N * ceil(K/block_size) parameter blocks.");
  std::vector<double> scales, zeros;
  scales.reserve(run.blocks.size());
  zeros.reserve(run.blocks.size());
  const uint32_t midpoint = uint32_t{1} << (geometry.bits - 1);
  const uint32_t maximum = (uint32_t{1} << geometry.bits) - 1;
  bool implicit = true, packed = true;
  for (const auto &block : run.blocks) {
    EXT_ENFORCE_INVALID(block.offset == 0 && block.codebook.empty(),
                        "ORT MatMulNBits does not support offsets or codebooks.");
    const double scale = RoundFloat(block.scale, geometry.type);
    const double zero = RoundFloat(block.zero_point, geometry.type);
    EXT_ENFORCE_INVALID(scale != 0 || block.scale == 0,
                        "ORT MatMulNBits scale underflows the input dtype.");
    scales.push_back(scale);
    zeros.push_back(zero);
    implicit &= zero == midpoint;
    packed &= zero >= 0 && zero <= maximum && std::trunc(zero) == zero;
  }
  const int32_t zero_type = implicit ? TensorProto::UNDEFINED
                            : packed ? TensorProto::UINT8
                                     : geometry.type;
  const OrtStorage storage(tensor.shape, tensor.data_type, plan.format, run.layout.count,
                           zero_type);
  EncodedValueProto result;
  ByteWriter payload(*result.mutable_raw_data(), storage.total_bytes);
  std::vector<uint32_t> codes(storage.block_size);
  for (size_t column = 0; column < storage.n; ++column) {
    for (size_t group = 0; group < storage.groups; ++group) {
      const size_t index = column * storage.groups + group;
      for (size_t j = 0; j < storage.block_size; ++j) {
        const size_t row = group * storage.block_size + j;
        codes[j] = 0;
        if (row >= storage.k)
          continue;
        const double value = ReadFloat(tensor, row * storage.n + column);
        EXT_ENFORCE_INVALID(std::isfinite(value), "Quantization input must be finite.");
        EXT_ENFORCE_INVALID(scales[index] != 0 || value == 0,
                            "ORT MatMulNBits zero scale requires an all-zero source block.");
        const long double normalized =
            scales[index] == 0 ? zeros[index]
                               : static_cast<long double>(value) / scales[index] + zeros[index];
        codes[j] = static_cast<uint32_t>(NearestEven(
            static_cast<double>(std::clamp(normalized, 0.L, static_cast<long double>(maximum)))));
      }
      Pack(payload, codes, storage.bits, false);
    }
  }
  for (double scale : scales)
    PutFloat(payload, scale, storage.type);
  if (zero_type == TensorProto::UINT8) {
    codes.resize(storage.groups);
    for (size_t column = 0; column < storage.n; ++column) {
      for (size_t group = 0; group < storage.groups; ++group)
        codes[group] = static_cast<uint32_t>(zeros[column * storage.groups + group]);
      Pack(payload, codes, storage.bits, false);
    }
  } else if (zero_type != TensorProto::UNDEFINED) {
    for (double zero : zeros)
      PutFloat(payload, zero, storage.type);
  }
  payload.Finish();
  *result.mutable_struct_type() = OrtSchema(storage, plan.format);
  SetLogicalTensor(result, tensor);
  return result;
}

EncodedValueProto EncodeTensor(const Tensor &tensor, const QuantizationPlan &plan) {
  size_t count = 1;
  for (int64_t dim : tensor.shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "Quantization input shape must be concrete.");
    count = Product(count, dim);
  }
  EXT_ENFORCE_INVALID(tensor.size_bytes() == Product(count, FloatBytes(tensor.data_type)) &&
                          (count == 0 || tensor.bytes() != nullptr),
                      "Invalid quantization source tensor storage.");
  if (OrtBits(plan.format) != 0)
    return EncodeOrtTensor(tensor, plan);
  ValidatePlan(plan, count);
  std::vector<double> values(count), exceptions;
  for (size_t i = 0; i < count; ++i) {
    values[i] = ReadFloat(tensor, i);
    EXT_ENFORCE_INVALID(std::isfinite(values[i]), "Quantization input must be finite.");
  }
  for (int64_t index : plan.outliers) {
    exceptions.push_back(values[index]);
    values[index] = 0;
  }
  if (!plan.permutation.empty()) {
    const auto original = values;
    for (size_t i = 0; i < count; ++i)
      values[i] = original[plan.permutation[i]];
  }
  Transform(values, plan.forward, plan.transform_size);
  EncodedValueProto result;
  ByteWriter payload(*result.mutable_raw_data(), PayloadBytes(plan));
  payload.Put(0, 1);
  for (int64_t index : plan.permutation)
    payload.Put(static_cast<uint64_t>(index), 8);
  for (double value : plan.forward)
    payload.PutDouble(value);
  for (double value : plan.inverse)
    payload.PutDouble(value);
  for (int64_t index : plan.outliers)
    payload.Put(static_cast<uint64_t>(index), 8);
  for (double value : exceptions)
    payload.PutDouble(value);
  size_t offset = 0;
  for (const auto &run : plan.runs)
    for (const auto &block : run.blocks) {
      EncodeBlock(payload, run.layout, block, values, offset);
      offset += run.layout.count;
    }
  payload.Finish();
  *result.mutable_struct_type() = Schema(plan);
  SetLogicalTensor(result, tensor);
  return result;
}

struct DecodedValues {
  Shape shape;
  int32_t type;
  std::vector<double> values;
};

struct QuantizationHeader {
  const StructTypeProto *root;
  QuantizationFormat format;
  Shape shape;
  int32_t type;
  size_t count;
};

QuantizationHeader ReadQuantizationHeader(const EncodedValueProto &encoded,
                                          const StructTypeCatalogue &catalogue) {
  const auto layout = catalogue.ValidateEncodedValue(encoded);
  EXT_ENFORCE_INVALID(!layout.external && layout.root && layout.record_count == 1,
                      "Expected one loaded structured quantization record.");
  const auto &root = *layout.root;
  const std::string name = root.name().value();
  EXT_ENFORCE_INVALID(name.starts_with(kPrefix), "Unsupported encoded quantization layout.");
  EXT_ENFORCE_INVALID(encoded.has_logical_type() && encoded.logical_type().has_tensor_type(),
                      "Quantization logical type must be a concrete floating-point tensor.");
  const auto &logical = encoded.logical_type().tensor_type();
  FloatBytes(logical.elem_type());
  EXT_ENFORCE_INVALID(logical.has_shape(), "Missing quantization logical shape.");
  Shape shape;
  size_t count = 1;
  for (const auto &dim : logical.shape().dim()) {
    EXT_ENFORCE_INVALID(dim.has_dim_value() && dim.dim_value() >= 0,
                        "Quantization logical shape must be concrete.");
    shape.push_back(dim.dim_value());
    count = Product(count, dim.dim_value());
  }
  return {&root, ParseQuantizationFormat(std::string_view(name).substr(std::strlen(kPrefix))),
          std::move(shape), logical.elem_type(), count};
}

struct OrtPackedValue {
  OrtStorage storage;
  std::span<const uint8_t> weights, scales, zeros;

  double ZeroPoint(size_t column, size_t group) const {
    if (storage.zero_type == TensorProto::UNDEFINED)
      return uint32_t{1} << (storage.bits - 1);
    if (storage.zero_type != TensorProto::UINT8)
      return PackedFloat(zeros, storage.type, column * storage.groups + group);
    const size_t per_byte = 8 / storage.bits;
    const uint8_t byte = zeros[column * CeilDiv(storage.groups, per_byte) + group / per_byte];
    return (byte >> ((group % per_byte) * storage.bits)) & ((1u << storage.bits) - 1);
  }
};

OrtPackedValue ParseOrtValue(const EncodedValueProto &encoded, const QuantizationHeader &header) {
  EXT_ENFORCE_INVALID(OrtBits(header.format) != 0, "Expected an ORT MatMulNBits encoded value.");
  const auto &root = *header.root;
  const auto &parameters = GetField(root, 0, "parameters");
  EXT_ENFORCE_INVALID(parameters.has_constant() && parameters.constant().int64_data().size() == 2 &&
                          parameters.constant().int64_data(0) == OrtBits(header.format),
                      "Invalid ORT MatMulNBits layout parameters.");
  int32_t zero_type = TensorProto::UNDEFINED;
  if (root.structure().field().size() > 3) {
    const auto &zero = GetField(root, 3, "zero_points");
    EXT_ENFORCE_INVALID(zero.has_type() && zero.type().has_tensor_type(),
                        "Invalid ORT MatMulNBits zero-point field.");
    zero_type = zero.type().tensor_type().elem_type();
  }
  OrtStorage storage(header.shape, header.type, header.format, parameters.constant().int64_data(1),
                     zero_type);
  StructTypeProto actual = root;
  actual.clear_type_id();
  EXT_ENFORCE_INVALID(actual.SerializeAsString() ==
                          OrtSchema(storage, header.format).SerializeAsString(),
                      "ORT MatMulNBits descriptor does not match its versioned schema.");
  EXT_ENFORCE_INVALID(encoded.raw_data().size() == storage.total_bytes,
                      "ORT MatMulNBits payload size mismatch.");
  const std::span<const uint8_t> data{encoded.raw_data().data(), encoded.raw_data().size()};
  OrtPackedValue value{storage, data.first(storage.weight_bytes),
                       data.subspan(storage.weight_bytes, storage.scale_bytes),
                       data.subspan(storage.weight_bytes + storage.scale_bytes)};
  for (size_t column = 0; column < storage.n; ++column)
    for (size_t group = 0; group < storage.groups; ++group) {
      EXT_ENFORCE_INVALID(
          std::isfinite(PackedFloat(value.scales, storage.type, column * storage.groups + group)),
          "Nonfinite ORT MatMulNBits scale.");
      EXT_ENFORCE_INVALID(std::isfinite(value.ZeroPoint(column, group)),
                          "Nonfinite ORT MatMulNBits zero point.");
    }
  return value;
}

DecodedValues DecodeValues(const EncodedValueProto &encoded, const StructTypeCatalogue &catalogue) {
  auto header = ReadQuantizationHeader(encoded, catalogue);
  if (OrtBits(header.format) != 0) {
    const auto packed = ParseOrtValue(encoded, header);
    const auto &storage = packed.storage;
    std::vector<double> values(header.count);
    for (size_t column = 0; column < storage.n; ++column)
      for (size_t row = 0; row < storage.k; ++row) {
        const size_t group = row / storage.block_size;
        const size_t within = row % storage.block_size;
        const size_t index = column * storage.groups + group;
        const uint8_t byte =
            packed.weights[index * storage.blob_size + within / (8 / storage.bits)];
        const uint32_t code =
            (byte >> ((within % (8 / storage.bits)) * storage.bits)) & ((1u << storage.bits) - 1);
        values[row * storage.n + column] = (code - packed.ZeroPoint(column, group)) *
                                           PackedFloat(packed.scales, storage.type, index);
      }
    return {std::move(header.shape), header.type, std::move(values)};
  }
  const auto &root = *header.root;
  const size_t count = header.count;
  QuantizationPlan plan;
  plan.format = header.format;
  ByteReader payload{{encoded.raw_data().data(), encoded.raw_data().size()}};
  EXT_ENFORCE_INVALID(payload.Get(1) == 0, "Nonzero quantization reserved byte.");
  const size_t permutation_size = Extent(GetField(root, 1, "permutation"), TensorProto::INT64, 1);
  EXT_ENFORCE_INVALID(permutation_size == 0 || permutation_size == count,
                      "Invalid quantization permutation size.");
  plan.permutation = ReadIndices(payload, permutation_size);
  const size_t matrix_size = Extent(GetField(root, 2, "forward"), TensorProto::DOUBLE, 2);
  EXT_ENFORCE_INVALID(matrix_size <= std::numeric_limits<uint32_t>::max(),
                      "Quantization transform is too large.");
  plan.transform_size = static_cast<uint32_t>(matrix_size);
  plan.forward = ReadDoubles(payload, Product(matrix_size, matrix_size));
  plan.inverse = ReadDoubles(payload, Product(matrix_size, matrix_size));
  const size_t outlier_count = Extent(GetField(root, 4, "outlier_indices"), TensorProto::INT64, 1);
  EXT_ENFORCE_INVALID(outlier_count <= count, "Too many quantization outliers.");
  plan.outliers = ReadIndices(payload, outlier_count);
  const auto exceptions = ReadDoubles(payload, outlier_count);
  const auto &blocks_field = GetField(root, 6, "blocks");
  EXT_ENFORCE_INVALID(blocks_field.has_type() && blocks_field.type().has_struct_type() &&
                          blocks_field.type().struct_type().has_structure(),
                      "Invalid quantization block list.");
  std::vector<double> values;
  const auto &block_fields = blocks_field.type().struct_type().structure().field();
  GetField(blocks_field.type().struct_type(), 0, "count");
  for (size_t block_index = 1; block_index < block_fields.size(); ++block_index) {
    const auto &field = block_fields[block_index];
    EXT_ENFORCE_INVALID(field.has_type() && field.type().has_struct_type() &&
                            field.type().struct_type().has_array(),
                        "Invalid quantization block run.");
    const auto &run = field.type().struct_type().array();
    EXT_ENFORCE_INVALID(run.dimension() > 0 &&
                            run.dimension() <= (payload.data.size() - payload.position) / 24 &&
                            run.element_type().has_struct_type(),
                        "Invalid quantization block run length or element type.");
    QuantizationRun decoded_run;
    decoded_run.layout = BlockParameters(run.element_type().struct_type());
    const auto &parameters = decoded_run.layout;
    EXT_ENFORCE_INVALID(parameters.count > 0 &&
                            parameters.count <= (count - values.size()) / run.dimension(),
                        "Quantization blocks exceed logical size.");
    decoded_run.blocks.reserve(run.dimension());
    for (uint64_t repeat = 0; repeat < run.dimension(); ++repeat) {
      QuantizationBlockParameters block;
      block.scale = payload.GetDouble();
      block.zero_point = payload.GetDouble();
      block.offset = payload.GetDouble();
      block.codebook = ReadDoubles(payload, TableSize(parameters));
      ValidateBlock(parameters, block);
      EXT_ENFORCE_INVALID(CodeBytes(parameters) <= payload.data.size() - payload.position,
                          "Truncated quantization codes.");
      DecodeBlock(payload, parameters, block, values);
      decoded_run.blocks.push_back(std::move(block));
    }
    plan.runs.push_back(std::move(decoded_run));
  }
  ValidatePlan(plan, count);
  auto expected = Schema(plan);
  StructTypeProto actual = root;
  actual.clear_type_id();
  EXT_ENFORCE_INVALID(actual.SerializeAsString() == expected.SerializeAsString(),
                      "Quantization descriptor does not match its versioned schema.");
  EXT_ENFORCE_INVALID(payload.position == payload.data.size(), "Trailing quantization payload.");
  Transform(values, plan.inverse, plan.transform_size);
  if (!plan.permutation.empty()) {
    const auto reordered = values;
    for (size_t i = 0; i < count; ++i)
      values[plan.permutation[i]] = reordered[i];
  }
  for (size_t i = 0; i < plan.outliers.size(); ++i)
    values[plan.outliers[i]] = exceptions[i];
  return {std::move(header.shape), header.type, std::move(values)};
}

} // namespace

MatMulNBitsInputs ExportMatMulNBitsInputs(const EncodedValueProto &value,
                                          const StructTypeCatalogue &catalogue) {
  const auto packed = ParseOrtValue(value, ReadQuantizationHeader(value, catalogue));
  const auto &storage = packed.storage;
  MatMulNBitsInputs result;
  result.k = storage.k;
  result.n = storage.n;
  result.bits = storage.bits;
  result.block_size = storage.block_size;
  const auto tensor = [](const char *name, int32_t type, const Shape &shape,
                         std::span<const uint8_t> bytes) {
    TensorProto output;
    output.set_name(name);
    output.set_data_type(type);
    for (int64_t dim : shape)
      output.add_dims(dim);
    auto *raw = output.mutable_raw_data();
    raw->resize(bytes.size());
    std::memcpy(raw->data(), bytes.data(), bytes.size());
    return output;
  };
  result.weights = tensor("B", TensorProto::UINT8,
                          {int64_t(storage.n), int64_t(storage.groups), int64_t(storage.blob_size)},
                          packed.weights);
  result.scales =
      tensor("scales", storage.type, {int64_t(storage.n), int64_t(storage.groups)}, packed.scales);
  if (storage.zero_type != TensorProto::UNDEFINED)
    result.zero_points =
        tensor("zero_points", storage.zero_type,
               {int64_t(storage.n), int64_t(storage.zero_type == TensorProto::UINT8
                                                ? CeilDiv(storage.groups, 8 / storage.bits)
                                                : storage.groups)},
               packed.zeros);
  return result;
}

RuntimeValue QuantizeTensor(const Tensor &tensor, const QuantizationPlan &plan) {
  return RuntimeValue(EncodeTensor(tensor, plan));
}

Tensor DequantizeTensor(const EncodedValueProto &value, const StructTypeCatalogue &catalogue,
                        RawBufferAllocator *allocator) {
  const auto decoded = DecodeValues(value, catalogue);
  const size_t width = FloatBytes(decoded.type);
  Tensor result = MakeOutputTensor(decoded.type, decoded.shape,
                                   Product(decoded.values.size(), width), allocator);
  result.name = value.name().value();
  for (size_t i = 0; i < decoded.values.size(); ++i)
    WriteFloat(result.mutable_bytes() + i * width, decoded.type, decoded.values[i]);
  return result;
}

Tensor DequantizeTensor(const RuntimeValue &value, const StructTypeCatalogue &catalogue,
                        RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kEncoded,
                      "Dequantization requires an encoded RuntimeValue.");
  return DequantizeTensor(value.Encoded(), catalogue, allocator);
}

EncodedValueProto QuantizeTensorProto(const TensorProto &tensor, const QuantizationPlan &plan) {
  EXT_ENFORCE_INVALID(tensor.data_location() != TensorProto::EXTERNAL || tensor.is_raw_data(),
                      "Load external TensorProto data before quantization.");
  EncodedValueProto result = EncodeTensor(TensorFromProto(tensor), plan);
  if (!tensor.has_name())
    result.clear_name();
  if (tensor.has_doc_string())
    result.set_doc_string(tensor.doc_string().value());
  return result;
}

TensorProto DequantizeTensorProto(const EncodedValueProto &value,
                                  const StructTypeCatalogue &catalogue) {
  const auto decoded = DecodeValues(value, catalogue);
  TensorProto result;
  if (value.has_name())
    result.set_name(value.name().value());
  if (value.has_doc_string())
    result.set_doc_string(value.doc_string().value());
  result.set_data_type(decoded.type);
  for (int64_t dim : decoded.shape)
    result.add_dims(dim);
  const size_t width = FloatBytes(decoded.type);
  ByteWriter payload(*result.mutable_raw_data(), Product(decoded.values.size(), width));
  for (double number : decoded.values)
    PutFloat(payload, number, decoded.type);
  payload.Finish();
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
