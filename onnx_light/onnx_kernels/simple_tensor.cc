// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/simple_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

size_t ElementSize(int32_t dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return sizeof(float);
  case DataType::DOUBLE:
    return sizeof(double);
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT64:
    return sizeof(int64_t);
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::BOOL:
  case DataType::FLOAT8E4M3FN:
  case DataType::FLOAT8E4M3FNUZ:
  case DataType::FLOAT8E5M2:
  case DataType::FLOAT8E5M2FNUZ:
    return 1;
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
    return 2;
  case DataType::UINT32:
    return 4;
  case DataType::UINT64:
    return 8;
  default:
    throw std::invalid_argument("Tensor::ElementSize: unsupported data_type.");
  }
}

int64_t Tensor::element_count() const {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

size_t Tensor::element_size() const { return ElementSize(data_type); }

size_t PackedByteSize(int32_t dtype, int64_t element_count) {
  EXT_ENFORCE_INVALID(element_count >= 0, "PackedByteSize: element_count must be non-negative.");
  switch (static_cast<DataType>(dtype)) {
  case DataType::INT4:
  case DataType::UINT4:
    // Two 4-bit elements packed per byte (low nibble first).
    return static_cast<size_t>((element_count + 1) / 2);
  case DataType::INT2:
  case DataType::UINT2:
    // Four 2-bit elements packed per byte (least significant pair first).
    return static_cast<size_t>((element_count + 3) / 4);
  default:
    return static_cast<size_t>(element_count) * ElementSize(dtype);
  }
}

Tensor Tensor::FromFloat(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<float> &values) {
  return Tensor::From<float>(name, shape, values);
}

Tensor Tensor::FromDouble(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<double> &values) {
  return Tensor::From<double>(name, shape, values);
}

Tensor Tensor::FromInt32(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int32_t> &values) {
  return Tensor::From<int32_t>(name, shape, values);
}

Tensor Tensor::FromInt64(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int64_t> &values) {
  return Tensor::From<int64_t>(name, shape, values);
}

Tensor Tensor::FromInt8(const std::string &name, const std::vector<int64_t> &shape,
                        const std::vector<int8_t> &values) {
  return Tensor::From<int8_t>(name, shape, values);
}

Tensor Tensor::FromUint8(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<uint8_t> &values) {
  return Tensor::From<uint8_t>(name, shape, values);
}

Tensor Tensor::FromInt16(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int16_t> &values) {
  return Tensor::From<int16_t>(name, shape, values);
}

Tensor Tensor::FromUint16(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<uint16_t> &values) {
  return Tensor::From<uint16_t>(name, shape, values);
}

Tensor Tensor::FromUint32(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<uint32_t> &values) {
  return Tensor::From<uint32_t>(name, shape, values);
}

Tensor Tensor::FromUint64(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<uint64_t> &values) {
  return Tensor::From<uint64_t>(name, shape, values);
}

Tensor Tensor::FromBool(const std::string &name, const std::vector<int64_t> &shape,
                        const std::vector<uint8_t> &values) {
  int64_t expected = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "Tensor shape dimensions must be non-negative.");
    expected *= d;
  }
  EXT_ENFORCE_INVALID(static_cast<int64_t>(values.size()) == expected,
                      "Tensor values size does not match the product of shape.");
  std::vector<uint8_t> bytes(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[i] = values[i] ? uint8_t{1} : uint8_t{0};
  }
  return Tensor(name, static_cast<int32_t>(DataType::BOOL), shape, std::move(bytes));
}

Tensor Tensor::FromStrings(const std::string &name, const std::vector<int64_t> &shape,
                           const std::vector<std::string> &values) {
  int64_t expected = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "Tensor shape dimensions must be non-negative.");
    expected *= d;
  }
  EXT_ENFORCE_INVALID(static_cast<int64_t>(values.size()) == expected,
                      "Tensor values size does not match the product of shape.");
  return Tensor::MakeString(name, shape, values);
}

const float *Tensor::AsFloat() const { return As<float>(); }
float *Tensor::AsFloat() { return As<float>(); }
const double *Tensor::AsDouble() const { return As<double>(); }
double *Tensor::AsDouble() { return As<double>(); }
const int32_t *Tensor::AsInt32() const { return As<int32_t>(); }
int32_t *Tensor::AsInt32() { return As<int32_t>(); }
const int64_t *Tensor::AsInt64() const { return As<int64_t>(); }
int64_t *Tensor::AsInt64() { return As<int64_t>(); }
const int8_t *Tensor::AsInt8() const { return As<int8_t>(); }
int8_t *Tensor::AsInt8() { return As<int8_t>(); }
const uint8_t *Tensor::AsUint8() const { return As<uint8_t>(); }
uint8_t *Tensor::AsUint8() { return As<uint8_t>(); }
const int16_t *Tensor::AsInt16() const { return As<int16_t>(); }
int16_t *Tensor::AsInt16() { return As<int16_t>(); }
const uint16_t *Tensor::AsUint16() const { return As<uint16_t>(); }
uint16_t *Tensor::AsUint16() { return As<uint16_t>(); }
const uint32_t *Tensor::AsUint32() const { return As<uint32_t>(); }
uint32_t *Tensor::AsUint32() { return As<uint32_t>(); }
const uint64_t *Tensor::AsUint64() const { return As<uint64_t>(); }
uint64_t *Tensor::AsUint64() { return As<uint64_t>(); }

const uint8_t *Tensor::AsBool() const {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(DataType::BOOL),
                      "Tensor data_type does not match the requested view type.");
  return data.data();
}

uint8_t *Tensor::AsBool() {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(DataType::BOOL),
                      "Tensor data_type does not match the requested view type.");
  return data.data();
}

const std::vector<std::string> &Tensor::AsStrings() const {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(DataType::STRING),
                      "Tensor data_type does not match the requested view type.");
  return string_data;
}

std::vector<std::string> &Tensor::AsStrings() {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(DataType::STRING),
                      "Tensor data_type does not match the requested view type.");
  return string_data;
}

void FillValueInfo(const Tensor &tensor, ValueInfoProto &vi) {
  vi.set_name(tensor.name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(tensor.data_type);
  TensorShapeProto *sh = tt->add_shape();
  for (int64_t d : tensor.shape) {
    sh->add_dim()->set_dim_value(d);
  }
}

Tensor TensorFromProto(const TensorProto &tp) {
  // Tensor name.
  const std::string name = tp.name().as_string();

  // Tensor shape (dims are stored as uint64 in TensorProto).
  std::vector<int64_t> shape;
  shape.reserve(tp.dims().size());
  for (size_t i = 0; i < tp.dims().size(); ++i) {
    shape.push_back(static_cast<int64_t>(tp.dims()[i]));
  }

  const int32_t dtype = static_cast<int32_t>(tp.data_type());

  // STRING tensors live in string_data, not in raw/typed bytes.
  if (dtype == static_cast<int32_t>(DataType::STRING)) {
    std::vector<std::string> strings;
    strings.reserve(tp.string_data().size());
    for (size_t i = 0; i < tp.string_data().size(); ++i) {
      strings.emplace_back(tp.string_data()[i].as_string());
    }
    return Tensor::FromStrings(name, shape, strings);
  }

  // Raw data path: bytes are already in little-endian layout.
  if (tp.is_raw_data()) {
    const auto &rd = tp.raw_data();
    std::vector<uint8_t> bytes(rd.data(), rd.data() + rd.size());
    return Tensor(name, dtype, std::move(shape), std::move(bytes));
  }

  // Typed-field path: convert each field's values into a raw byte vector.
  std::vector<uint8_t> bytes;

  switch (tp.data_type()) {
  case TensorProto::DataType::FLOAT: {
    const auto &fd = tp.float_data().values();
    bytes.resize(fd.size() * sizeof(float));
    if (!fd.empty()) {
      std::memcpy(bytes.data(), fd.data(), bytes.size());
    }
    break;
  }
  case TensorProto::DataType::DOUBLE: {
    const auto &dd = tp.double_data().values();
    bytes.resize(dd.size() * sizeof(double));
    if (!dd.empty()) {
      std::memcpy(bytes.data(), dd.data(), bytes.size());
    }
    break;
  }
  case TensorProto::DataType::INT64: {
    const auto &i64 = tp.int64_data().values();
    bytes.resize(i64.size() * sizeof(int64_t));
    if (!i64.empty()) {
      std::memcpy(bytes.data(), i64.data(), bytes.size());
    }
    break;
  }
  case TensorProto::DataType::UINT64: {
    const auto &u64 = tp.uint64_data().values();
    bytes.resize(u64.size() * sizeof(uint64_t));
    if (!u64.empty()) {
      std::memcpy(bytes.data(), u64.data(), bytes.size());
    }
    break;
  }
  case TensorProto::DataType::UINT32: {
    // uint64_data stores uint32 values as uint64; truncate each to 4 bytes.
    const auto &u64 = tp.uint64_data().values();
    bytes.resize(u64.size() * sizeof(uint32_t));
    for (size_t i = 0; i < u64.size(); ++i) {
      const uint32_t v = static_cast<uint32_t>(u64[i]);
      std::memcpy(bytes.data() + i * sizeof(uint32_t), &v, sizeof(uint32_t));
    }
    break;
  }
  case TensorProto::DataType::INT32: {
    const auto &i32 = tp.int32_data().values();
    bytes.resize(i32.size() * sizeof(int32_t));
    if (!i32.empty()) {
      std::memcpy(bytes.data(), i32.data(), bytes.size());
    }
    break;
  }
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::UINT16:
  case TensorProto::DataType::FLOAT16:
  case TensorProto::DataType::BFLOAT16: {
    // int32_data stores one 16-bit element per int32; take the low 2 bytes.
    const auto &i32 = tp.int32_data().values();
    bytes.resize(i32.size() * sizeof(uint16_t));
    for (size_t i = 0; i < i32.size(); ++i) {
      const uint16_t v = static_cast<uint16_t>(static_cast<uint32_t>(i32[i]));
      std::memcpy(bytes.data() + i * sizeof(uint16_t), &v, sizeof(uint16_t));
    }
    break;
  }
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::BOOL:
  case TensorProto::DataType::FLOAT8E4M3FN:
  case TensorProto::DataType::FLOAT8E4M3FNUZ:
  case TensorProto::DataType::FLOAT8E5M2:
  case TensorProto::DataType::FLOAT8E5M2FNUZ:
  case TensorProto::DataType::FLOAT8E8M0: {
    // int32_data stores one 8-bit element per int32; take the low byte.
    const auto &i32 = tp.int32_data().values();
    bytes.resize(i32.size());
    for (size_t i = 0; i < i32.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(static_cast<uint32_t>(i32[i]));
    }
    break;
  }
  case TensorProto::DataType::INT4:
  case TensorProto::DataType::UINT4:
  case TensorProto::DataType::FLOAT4E2M1:
  case TensorProto::DataType::INT2:
  case TensorProto::DataType::UINT2: {
    // Sub-byte packed types: each int32 stores one packed byte (two 4-bit or
    // four 2-bit elements). Take the low byte of each int32.
    const auto &i32 = tp.int32_data().values();
    bytes.resize(i32.size());
    for (size_t i = 0; i < i32.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(static_cast<uint32_t>(i32[i]));
    }
    break;
  }
  default:
    throw std::invalid_argument("TensorFromProto: unsupported data_type " + std::to_string(dtype));
  }

  return Tensor(name, dtype, std::move(shape), std::move(bytes));
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
