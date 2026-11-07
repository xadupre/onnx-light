// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr std::string_view kRevision = "eb058d630f7344099459b6795b1e3eeffdda4b40";
constexpr std::string_view kV1Prefix = "onnx_light.quantization.v1/";
constexpr std::string_view kV2Prefix = "onnx_light.quantization.v2/";

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "compressed-tensors: size overflow.");
  return a * b;
}

void AppendLittleEndianDouble(std::string &data, double value) {
  const uint64_t bits = std::bit_cast<uint64_t>(value);
  char bytes[sizeof(bits)];
  for (size_t i = 0; i < sizeof(bits); ++i)
    bytes[i] = static_cast<char>((bits >> (8 * i)) & 255);
  data.append(bytes, sizeof(bytes));
}

double ReadLittleEndianDouble(const char *data) {
  uint64_t bits = 0;
  for (size_t i = 0; i < sizeof(bits); ++i)
    bits |= uint64_t(static_cast<uint8_t>(data[i])) << (8 * i);
  return std::bit_cast<double>(bits);
}

void CheckMetadata(const CompressedTensorsWeights &checkpoint, size_t max_bytes) {
  EXT_ENFORCE_INVALID(checkpoint.upstream_revision == kRevision,
                      "compressed-tensors: unsupported upstream revision.");
  EXT_ENFORCE_INVALID(max_bytes > 0, "compressed-tensors: max_bytes must be positive.");
  const auto &args = checkpoint.quantization_args;
  EXT_ENFORCE_INVALID(args.num_bits >= 1 && args.num_bits <= 8,
                      "compressed-tensors: num_bits must be between 1 and 8.");
  EXT_ENFORCE_INVALID(!args.dynamic && !args.actorder && !args.block_structure,
                      "compressed-tensors: dynamic, actorder and block_structure are unsupported.");
  size_t total = 0;
  for (const auto &[name, tensor] : checkpoint.tensors) {
    EXT_ENFORCE_INVALID(!name.empty(), "compressed-tensors: tensor names cannot be empty.");
    EXT_ENFORCE_INVALID(tensor.size_bytes() <= max_bytes - total,
                        "compressed-tensors: checkpoint exceeds max_bytes.");
    total += tensor.size_bytes();
  }
}

const Tensor &RequireTensor(const CompressedTensorsWeights &checkpoint, std::string_view name,
                            int32_t type, const Shape *shape = nullptr) {
  const auto it = checkpoint.tensors.find(std::string(name));
  EXT_ENFORCE_INVALID(it != checkpoint.tensors.end(), "compressed-tensors: missing ",
                      std::string(name), ".");
  const Tensor &tensor = it->second;
  EXT_ENFORCE_INVALID(tensor.data_type == type, "compressed-tensors: ", std::string(name),
                      " has an invalid element type.");
  if (shape)
    EXT_ENFORCE_INVALID(tensor.shape == *shape, "compressed-tensors: ", std::string(name),
                        " has an invalid shape.");
  return tensor;
}

void RequireTensorNames(const CompressedTensorsWeights &checkpoint,
                        const std::vector<std::string_view> &required,
                        const std::vector<std::string_view> &optional = {}) {
  for (std::string_view name : required)
    EXT_ENFORCE_INVALID(checkpoint.tensors.contains(std::string(name)),
                        "compressed-tensors: missing ", std::string(name), ".");
  for (const auto &[name, unused] : checkpoint.tensors) {
    (void)unused;
    const bool known = std::find(required.begin(), required.end(), name) != required.end() ||
                       std::find(optional.begin(), optional.end(), name) != optional.end();
    EXT_ENFORCE_INVALID(known, "compressed-tensors: unexpected tensor ", name, ".");
  }
}

void CheckTensorSize(const Tensor &tensor, size_t expected, std::string_view name) {
  EXT_ENFORCE_INVALID(tensor.size_bytes() == expected, "compressed-tensors: ", std::string(name),
                      " has an invalid byte extent.");
}

size_t TensorBytes(const Shape &shape, size_t element_size) {
  size_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "compressed-tensors: negative tensor dimension.");
    count = Product(count, static_cast<size_t>(dim));
  }
  return Product(count, element_size);
}

Shape CheckMatrix(const Tensor &tensor, int32_t type, std::string_view name, size_t max_bytes) {
  EXT_ENFORCE_INVALID(tensor.shape.size() == 2 && tensor.shape[0] > 0 && tensor.shape[1] > 0,
                      "compressed-tensors: ", std::string(name), " must be a nonempty matrix.");
  const size_t count =
      Product(static_cast<size_t>(tensor.shape[0]), static_cast<size_t>(tensor.shape[1]));
  EXT_ENFORCE_INVALID(count <= max_bytes / 4,
                      "compressed-tensors: logical FLOAT32 weights exceed max_bytes.");
  EXT_ENFORCE_INVALID(tensor.data_type == type, "compressed-tensors: ", std::string(name),
                      " has an invalid element type.");
  return tensor.shape;
}

std::string_view Bytes(const Tensor &tensor) {
  return {reinterpret_cast<const char *>(tensor.bytes()), tensor.size_bytes()};
}

Tensor MakeTensor(std::string name, int32_t type, Shape shape, std::string_view bytes) {
  return Tensor(std::move(name), type, std::move(shape),
                std::vector<uint8_t>(bytes.begin(), bytes.end()));
}

Tensor MakeInt64Tensor(std::string name, const Shape &shape) {
  std::string bytes(shape.size() * sizeof(int64_t), '\0');
  for (size_t i = 0; i < shape.size(); ++i)
    std::memcpy(bytes.data() + i * sizeof(int64_t), &shape[i], sizeof(int64_t));
  return MakeTensor(std::move(name), TensorProto::INT64, {static_cast<int64_t>(shape.size())},
                    bytes);
}

void ValidateArguments(const CompressedTensorsQuantizationArgs &args, std::string_view scale,
                       std::string_view zero) {
  for (const auto &[name, expected] :
       {std::pair<std::string_view, std::string_view>{"scale_dtype", scale}, {"zp_dtype", zero}}) {
    const auto &actual = name == "scale_dtype" ? args.scale_dtype : args.zp_dtype;
    EXT_ENFORCE_INVALID(
        !actual || *actual == expected || *actual == "torch." + std::string(expected),
        "compressed-tensors: ", std::string(name), " must be ", std::string(expected), " or null.");
  }
}

struct IntegerGeometry {
  uint32_t bits;
  size_t block_size;
  Shape parameters;
};

IntegerGeometry GetIntegerGeometry(const CompressedTensorsQuantizationArgs &args,
                                   const Shape &shape) {
  EXT_ENFORCE_INVALID(args.type == "int" && args.num_bits >= 1 && args.num_bits <= 8,
                      "compressed-tensors: pack-quantized requires int, 1--8 bits.");
  const size_t rows = static_cast<size_t>(shape[0]), columns = static_cast<size_t>(shape[1]);
  size_t block_size = columns;
  Shape parameters{static_cast<int64_t>(rows), 1};
  if (args.strategy == "tensor") {
    block_size = Product(rows, columns);
    parameters = {1};
    EXT_ENFORCE_INVALID(!args.group_size || *args.group_size == -1,
                        "compressed-tensors: group_size conflicts with tensor strategy.");
  } else if (args.strategy == "channel") {
    EXT_ENFORCE_INVALID(!args.group_size || *args.group_size == -1,
                        "compressed-tensors: group_size conflicts with channel strategy.");
  } else if (args.strategy == "group") {
    EXT_ENFORCE_INVALID(args.group_size && *args.group_size > 0,
                        "compressed-tensors: group_size must be positive.");
    block_size = static_cast<size_t>(*args.group_size);
    EXT_ENFORCE_INVALID(columns % block_size == 0,
                        "compressed-tensors: K must be divisible by group_size.");
    parameters[1] = static_cast<int64_t>(columns / block_size);
  } else {
    EXT_THROW_INVALID("compressed-tensors: unsupported integer strategy.");
  }
  EXT_ENFORCE_INVALID(block_size <= std::numeric_limits<uint32_t>::max(),
                      "compressed-tensors: affine block is too large.");
  return {args.num_bits, block_size, std::move(parameters)};
}

QuantizationPlan IntegerPlan(const CompressedTensorsQuantizationArgs &args, const Shape &shape,
                             QuantizationFormat format = QuantizationFormat::kInt8,
                             int32_t reconstruction = TensorProto::DOUBLE) {
  const auto geometry = GetIntegerGeometry(args, shape);
  const size_t count = Product(static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1]));
  QuantizationPlan plan =
      MakeQuantizationPlan(format, count, static_cast<uint64_t>(geometry.block_size));
  plan.format = format;
  plan.reconstruction_dtype = reconstruction;
  plan.runs[0].layout.bits = geometry.bits;
  return plan;
}

EncodedValueProto ImportInteger(const CompressedTensorsWeights &checkpoint, size_t max_bytes) {
  const auto &args = checkpoint.quantization_args;
  RequireTensorNames(
      checkpoint,
      args.symmetric
          ? std::vector<std::string_view>{"weight_packed", "weight_shape", "weight_scale"}
          : std::vector<std::string_view>{"weight_packed", "weight_shape", "weight_scale",
                                          "weight_zero_point"});
  const Shape shape_tensor_shape{2};
  const Tensor &shape_tensor =
      RequireTensor(checkpoint, "weight_shape", TensorProto::INT64, &shape_tensor_shape);
  CheckTensorSize(shape_tensor, 2 * sizeof(int64_t), "weight_shape");
  int64_t dims[2];
  std::memcpy(dims, shape_tensor.bytes(), sizeof(dims));
  Shape shape{dims[0], dims[1]};
  EXT_ENFORCE_INVALID(shape[0] > 0 && shape[1] > 0,
                      "compressed-tensors: weights must be a nonempty matrix [N,K].");
  const size_t rows = static_cast<size_t>(shape[0]), columns = static_cast<size_t>(shape[1]);
  const size_t element_count = Product(rows, columns);
  EXT_ENFORCE_INVALID(element_count <= max_bytes / sizeof(float),
                      "compressed-tensors: logical FLOAT32 weights exceed max_bytes.");
  const auto geometry = GetIntegerGeometry(args, shape);
  ValidateArguments(args, "float32", "int8");
  const size_t packed_columns = (Product(columns, geometry.bits) + 31) / 32;
  const Shape packed_shape{shape[0], static_cast<int64_t>(packed_columns)};
  const Tensor &packed =
      RequireTensor(checkpoint, "weight_packed", TensorProto::INT32, &packed_shape);
  CheckTensorSize(packed, Product(Product(rows, packed_columns), sizeof(int32_t)), "weight_packed");
  const Tensor &scales =
      RequireTensor(checkpoint, "weight_scale", TensorProto::FLOAT, &geometry.parameters);
  const size_t parameter_count = TensorBytes(geometry.parameters, sizeof(float)) / sizeof(float);
  CheckTensorSize(scales, Product(parameter_count, sizeof(float)), "weight_scale");
  for (size_t i = 0; i < parameter_count; ++i) {
    float scale;
    std::memcpy(&scale, scales.bytes() + i * sizeof(float), sizeof(float));
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0,
                        "compressed-tensors: weight_scale must be finite and positive.");
  }
  const size_t code_bytes = (Product(geometry.block_size, geometry.bits) + 7) / 8;
  const size_t encoded_bytes = 1 + Product(parameter_count, 24 + code_bytes);
  EXT_ENFORCE_INVALID(encoded_bytes <= max_bytes,
                      "compressed-tensors: encoded affine payload exceeds max_bytes.");
  std::string zero_bytes;
  if (args.symmetric) {
    zero_bytes.assign(Product(parameter_count, sizeof(int8_t)), '\0');
  } else if (args.strategy == "tensor") {
    const Tensor &zero =
        RequireTensor(checkpoint, "weight_zero_point", TensorProto::INT8, &geometry.parameters);
    CheckTensorSize(zero, parameter_count, "weight_zero_point");
    zero_bytes.assign(reinterpret_cast<const char *>(zero.bytes()), zero.size_bytes());
  } else {
    const Shape zero_shape{static_cast<int64_t>((Product(rows, args.num_bits) + 31) / 32),
                           geometry.parameters[1]};
    const Tensor &zero =
        RequireTensor(checkpoint, "weight_zero_point", TensorProto::INT32, &zero_shape);
    const size_t zero_size = TensorBytes(zero_shape, sizeof(int32_t));
    CheckTensorSize(zero, zero_size, "weight_zero_point");
    zero_bytes = UnpackCompressedTensorInt32(
        Bytes(zero), rows, static_cast<size_t>(geometry.parameters[1]), geometry.bits, 0);
  }
  EXT_ENFORCE_INVALID(zero_bytes.size() == parameter_count,
                      "compressed-tensors: invalid zero point extent.");
  const auto *zeros = reinterpret_cast<const int8_t *>(zero_bytes.data());
  const int half = 1 << (geometry.bits - 1);
  for (size_t i = 0; i < parameter_count; ++i)
    EXT_ENFORCE_INVALID(zeros[i] >= -half && zeros[i] < half,
                        "compressed-tensors: zero points exceed the signed code range.");
  const std::string codes =
      UnpackCompressedTensorInt32(Bytes(packed), rows, columns, geometry.bits, 1);
  EncodedValueProto result;
  *result.mutable_struct_type() = MakeQuantizationType(IntegerPlan(args, shape));
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : shape)
    logical->mutable_shape()->add_dim()->set_dim_value(dim);
  std::string payload(1, '\0');
  size_t start = 0;
  for (size_t i = 0; i < parameter_count; ++i) {
    float scale_value;
    std::memcpy(&scale_value, scales.bytes() + i * sizeof(float), sizeof(float));
    const double scale = scale_value;
    const int32_t zero = zeros[i];
    const std::string_view block(codes.data() + start, geometry.block_size);
    ValidateCompressedTensorAffine(block, static_cast<float>(scale), zero);
    AppendLittleEndianDouble(payload, scale);
    AppendLittleEndianDouble(payload, static_cast<double>(zero));
    AppendLittleEndianDouble(payload, 0);
    payload.append(PackQuantizationCodes(block, geometry.bits));
    start += geometry.block_size;
  }
  auto *raw_data = result.mutable_raw_data();
  raw_data->resize(payload.size());
  std::memcpy(raw_data->data(), payload.data(), payload.size());
  return result;
}

Shape EncodedShape(const EncodedValueProto &value, size_t max_bytes) {
  EXT_ENFORCE_INVALID(value.data_location() != TensorProto::EXTERNAL &&
                          value.external_data_size() == 0,
                      "compressed-tensors: export requires inline storage.");
  EXT_ENFORCE_INVALID(!value.has_parameter_ref(),
                      "compressed-tensors: materialize shared quantization parameters first.");
  EXT_ENFORCE_INVALID(value.raw_data().size() <= max_bytes,
                      "compressed-tensors: encoded payload exceeds max_bytes.");
  EXT_ENFORCE_INVALID(value.logical_type().has_tensor_type() &&
                          value.logical_type().tensor_type().elem_type() == TensorProto::FLOAT,
                      "compressed-tensors: export requires logical FLOAT32 weights.");
  const auto &dims = value.logical_type().tensor_type().shape().dim();
  EXT_ENFORCE_INVALID(dims.size() == 2 && dims[0].has_dim_value() && dims[1].has_dim_value(),
                      "compressed-tensors: export requires a concrete matrix shape.");
  Shape shape{dims[0].dim_value(), dims[1].dim_value()};
  EXT_ENFORCE_INVALID(shape[0] > 0 && shape[1] > 0,
                      "compressed-tensors: weights must be a nonempty matrix [N,K].");
  EXT_ENFORCE_INVALID(Product(static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])) <=
                          max_bytes / sizeof(float),
                      "compressed-tensors: logical FLOAT32 weights exceed max_bytes.");
  return shape;
}

std::unordered_map<std::string, Tensor> ExportInteger(const EncodedValueProto &value,
                                                      const CompressedTensorsQuantizationArgs &args,
                                                      const Shape &shape, size_t max_bytes) {
  const auto geometry = GetIntegerGeometry(args, shape);
  ValidateArguments(args, "float32", "int8");
  const size_t parameter_count = TensorBytes(geometry.parameters, 1);
  const size_t count =
      Product(static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])) / parameter_count;
  const size_t expected_bytes =
      1 + Product(parameter_count, 24 + (Product(count, geometry.bits) + 7) / 8);
  const std::string &data = value.raw_data();
  EXT_ENFORCE_INVALID(data.size() == expected_bytes && data[0] == 0,
                      "compressed-tensors: invalid affine payload extent/header.");
  const auto &type = value.struct_type();
  EXT_ENFORCE_INVALID(type.has_name(),
                      "compressed-tensors: expected an inline portable affine schema.");
  const std::string type_name = type.name().value();
  EXT_ENFORCE_INVALID(type_name.starts_with(kV1Prefix) || type_name.starts_with(kV2Prefix),
                      "compressed-tensors: expected an inline portable affine schema.");
  const bool float_reconstruction = type_name.starts_with(kV2Prefix);
  std::string_view profile = type_name;
  profile.remove_prefix(float_reconstruction ? kV2Prefix.size() : kV1Prefix.size());
  const QuantizationFormat format = ParseQuantizationFormat(profile);
  EXT_ENFORCE_INVALID(format != QuantizationFormat::kQuarot &&
                          format != QuantizationFormat::kQuipSharp &&
                          format != QuantizationFormat::kSmoothquant,
                      "compressed-tensors: transform-based profiles are unsupported.");
  QuantizationPlan expected = IntegerPlan(
      args, shape, format, float_reconstruction ? TensorProto::FLOAT : TensorProto::DOUBLE);
  EXT_ENFORCE_INVALID(expected.runs[0].layout.method == QuantizationMethod::kAffine &&
                          expected.runs[0].layout.signed_codes,
                      "compressed-tensors: export requires signed affine codes.");
  EXT_ENFORCE_INVALID(type.SerializeAsString() ==
                          MakeQuantizationType(expected).SerializeAsString(),
                      "compressed-tensors: affine schema does not match bits/grouping or includes "
                      "unsupported transforms, unsigned codes, codebooks or outliers.");
  const size_t rows = static_cast<size_t>(shape[0]), columns = static_cast<size_t>(shape[1]);
  const size_t total = Product(rows, columns);
  std::string scales(parameter_count * sizeof(float), '\0');
  std::string zeros(parameter_count, '\0');
  std::string codes(total, '\0');
  size_t position = 1, start = 0;
  for (size_t i = 0; i < parameter_count; ++i) {
    const double scale = ReadLittleEndianDouble(data.data() + position);
    const double zero = ReadLittleEndianDouble(data.data() + position + sizeof(double));
    const double offset = ReadLittleEndianDouble(data.data() + position + 2 * sizeof(double));
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0 &&
                            scale <= std::numeric_limits<float>::max() &&
                            static_cast<double>(static_cast<float>(scale)) == scale,
                        "compressed-tensors: scale must be positive, finite and exactly FLOAT32; "
                        "scale conversion would not be lossless.");
    const int half = 1 << (geometry.bits - 1);
    EXT_ENFORCE_INVALID(std::isfinite(zero) && std::trunc(zero) == zero && zero >= -half &&
                            zero < half && offset == 0 && (!args.symmetric || zero == 0),
                        "compressed-tensors: invalid affine zero point, offset or symmetry.");
    const float float_scale = static_cast<float>(scale);
    std::memcpy(scales.data() + i * sizeof(float), &float_scale, sizeof(float));
    const int8_t signed_zero = static_cast<int8_t>(zero);
    zeros[i] = static_cast<char>(signed_zero);
    position += 3 * sizeof(double);
    const size_t packed_size = (geometry.block_size * geometry.bits + 7) / 8;
    const std::string_view packed(data.data() + position, packed_size);
    const size_t remainder = geometry.block_size * geometry.bits % 8;
    EXT_ENFORCE_INVALID(!remainder || (static_cast<uint8_t>(packed.back()) >> remainder) == 0,
                        "compressed-tensors: nonzero portable padding bits.");
    const std::string block = UnpackQuantizationCodes(packed, geometry.block_size, geometry.bits);
    ValidateCompressedTensorAffine(block, float_scale, signed_zero);
    std::memcpy(codes.data() + start, block.data(), block.size());
    position += packed_size;
    start += geometry.block_size;
  }
  const size_t packed_columns = (columns * geometry.bits + 31) / 32;
  std::unordered_map<std::string, Tensor> tensors;
  tensors.emplace("weight_packed",
                  MakeTensor("weight_packed", TensorProto::INT32,
                             {shape[0], static_cast<int64_t>(packed_columns)},
                             PackCompressedTensorInt32(codes, rows, columns, geometry.bits, 1)));
  tensors.emplace("weight_shape", MakeInt64Tensor("weight_shape", shape));
  tensors.emplace("weight_scale",
                  MakeTensor("weight_scale", TensorProto::FLOAT, geometry.parameters, scales));
  if (!args.symmetric) {
    if (args.strategy == "tensor") {
      tensors.emplace("weight_zero_point", MakeTensor("weight_zero_point", TensorProto::INT8,
                                                      geometry.parameters, zeros));
    } else {
      const size_t groups = static_cast<size_t>(geometry.parameters[1]);
      tensors.emplace("weight_zero_point",
                      MakeTensor("weight_zero_point", TensorProto::INT32,
                                 {(shape[0] * static_cast<int64_t>(geometry.bits) + 31) / 32,
                                  static_cast<int64_t>(groups)},
                                 PackCompressedTensorInt32(zeros, rows, groups, geometry.bits, 0)));
    }
  }
  for (const auto &[name, tensor] : tensors) {
    (void)name;
    EXT_ENFORCE_INVALID(tensor.size_bytes() <= max_bytes,
                        "compressed-tensors: exported tensors exceed max_bytes.");
  }
  return tensors;
}

std::pair<size_t, size_t> MxSpec(const std::string &format,
                                 const CompressedTensorsQuantizationArgs &args) {
  size_t bits = 0, group = 0;
  if (format == "mxfp4-pack-quantized") {
    bits = 4;
    group = 32;
  } else if (format == "mxfp6-pack-quantized") {
    bits = 6;
    group = 32;
  } else if (format == "mxfp8-quantized") {
    bits = 8;
    group = 32;
  } else if (format == "nvfp4-pack-quantized") {
    bits = 4;
    group = 16;
  } else if (format == "float-quantized") {
    EXT_THROW_INVALID("compressed-tensors: ordinary FP8 is unsupported.");
  } else {
    EXT_THROW_INVALID("compressed-tensors: unsupported compression format ", format, ".");
  }
  EXT_ENFORCE_INVALID(args.type == "float" && args.num_bits == bits && args.symmetric,
                      "compressed-tensors: MX/NV layouts require symmetric float weights.");
  EXT_ENFORCE_INVALID(args.strategy ==
                              (format == "nvfp4-pack-quantized" ? "tensor_group" : "group") &&
                          args.group_size && *args.group_size == static_cast<int64_t>(group),
                      "compressed-tensors: MX/NV layouts require the specified axis-1 group size.");
  ValidateArguments(args, format == "nvfp4-pack-quantized" ? "float8_e4m3fn" : "uint8",
                    format == "nvfp4-pack-quantized" ? "float8_e4m3fn" : "uint8");
  return {bits, group};
}

EncodedValueProto ImportMx(const CompressedTensorsWeights &checkpoint, size_t max_bytes) {
  const auto [bits, group] = MxSpec(checkpoint.compression_format, checkpoint.quantization_args);
  const bool nvfp4 = checkpoint.compression_format == "nvfp4-pack-quantized";
  const bool mxfp8 = bits == 8;
  const std::string_view weight_name = mxfp8 ? "weight" : "weight_packed";
  std::vector<std::string_view> required{weight_name, "weight_scale"};
  if (nvfp4)
    required.push_back("weight_global_scale");
  RequireTensorNames(checkpoint, required);
  const auto found = checkpoint.tensors.find(std::string(weight_name));
  EXT_ENFORCE_INVALID(found != checkpoint.tensors.end(),
                      "compressed-tensors: missing weight data.");
  const Tensor &packed = found->second;
  EXT_ENFORCE_INVALID(packed.shape.size() == 2,
                      "compressed-tensors: packed weights must be a matrix.");
  const int32_t packed_type = mxfp8 ? TensorProto::FLOAT8E4M3FN : TensorProto::UINT8;
  CheckMatrix(packed, packed_type, weight_name, max_bytes);
  size_t columns = static_cast<size_t>(packed.shape[1]);
  if (bits == 4) {
    EXT_ENFORCE_INVALID(columns <= std::numeric_limits<size_t>::max() / 2,
                        "compressed-tensors: packed columns overflow.");
    columns *= 2;
  } else if (bits == 6) {
    EXT_ENFORCE_INVALID(columns % 3 == 0 && columns / 3 <= std::numeric_limits<size_t>::max() / 4,
                        "compressed-tensors: invalid MXFP6 E2M3 packed extent.");
    columns = columns / 3 * 4;
  }
  EXT_ENFORCE_INVALID(columns % group == 0,
                      "compressed-tensors: logical columns do not align to scale groups.");
  const size_t rows = static_cast<size_t>(packed.shape[0]);
  EXT_ENFORCE_INVALID(Product(rows, columns) <= max_bytes / sizeof(float),
                      "compressed-tensors: logical FLOAT32 weights exceed max_bytes.");
  const Shape scales_shape{static_cast<int64_t>(rows), static_cast<int64_t>(columns / group)};
  const int32_t scale_type = nvfp4 ? TensorProto::FLOAT8E4M3FN : TensorProto::UINT8;
  const Tensor &scales = RequireTensor(checkpoint, "weight_scale", scale_type, &scales_shape);
  if (nvfp4) {
    const Shape global_shape{1};
    const Tensor &global =
        RequireTensor(checkpoint, "weight_global_scale", TensorProto::FLOAT, &global_shape);
    CheckTensorSize(global, sizeof(float), "weight_global_scale");
    float global_scale;
    std::memcpy(&global_scale, global.bytes(), sizeof(global_scale));
    EXT_ENFORCE_INVALID(std::isfinite(global_scale) && global_scale > 0,
                        "compressed-tensors: global scale must be finite and positive.");
    auto weights = Nvfp4Weights{packed, scales, global_scale};
    return ImportNvfp4Weights(weights);
  }
  if (mxfp8)
    return ImportMxfp8Weights(Mxfp8Weights{packed, scales});
  if (bits == 6)
    return ImportMxfp6E2m3Weights(Mxfp6E2m3Weights{packed, scales});
  return ImportMxfp4Weights(Mxfp4Weights{packed, scales});
}

} // namespace

EncodedValueProto ImportCompressedTensorsWeights(const CompressedTensorsWeights &checkpoint,
                                                 size_t max_bytes) {
  CheckMetadata(checkpoint, max_bytes);
  if (checkpoint.compression_format == "pack-quantized")
    return ImportInteger(checkpoint, max_bytes);
  return ImportMx(checkpoint, max_bytes);
}

CompressedTensorsWeights
ExportCompressedTensorsWeights(const EncodedValueProto &value, std::string compression_format,
                               CompressedTensorsQuantizationArgs quantization_args,
                               std::string upstream_revision, size_t max_bytes) {
  CompressedTensorsWeights result;
  result.compression_format = std::move(compression_format);
  result.quantization_args = std::move(quantization_args);
  result.upstream_revision = std::move(upstream_revision);
  CheckMetadata(result, max_bytes);
  const Shape shape = EncodedShape(value, max_bytes);
  if (result.compression_format == "pack-quantized") {
    result.tensors = ExportInteger(value, result.quantization_args, shape, max_bytes);
  } else {
    const auto [bits, group] = MxSpec(result.compression_format, result.quantization_args);
    (void)group;
    const bool nvfp4 = result.compression_format == "nvfp4-pack-quantized";
    if (bits == 8) {
      const auto weights = ExportMxfp8Weights(value);
      result.tensors.emplace("weight", weights.weight);
      result.tensors.emplace("weight_scale", weights.weight_scale);
    } else if (bits == 6) {
      const auto weights = ExportMxfp6E2m3Weights(value);
      result.tensors.emplace("weight_packed", weights.weight_packed);
      result.tensors.emplace("weight_scale", weights.weight_scale);
    } else if (nvfp4) {
      const auto weights = ExportNvfp4Weights(value);
      result.tensors.emplace("weight_packed", weights.weight_packed);
      result.tensors.emplace("weight_scale", weights.weight_scale);
      std::string global(sizeof(float), '\0');
      std::memcpy(global.data(), &weights.weight_global_scale, sizeof(float));
      result.tensors.emplace("weight_global_scale",
                             MakeTensor("weight_global_scale", TensorProto::FLOAT, {1}, global));
    } else {
      const auto weights = ExportMxfp4Weights(value);
      result.tensors.emplace("weight_packed", weights.weight_packed);
      result.tensors.emplace("weight_scale", weights.weight_scale);
    }
  }
  size_t total = 0;
  for (const auto &[name, tensor] : result.tensors) {
    (void)name;
    EXT_ENFORCE_INVALID(tensor.size_bytes() <= max_bytes - total,
                        "compressed-tensors: exported tensors exceed max_bytes.");
    total += tensor.size_bytes();
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
