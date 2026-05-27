// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/simple_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

size_t ElementSize(int32_t dtype) {
  switch (dtype) {
  case TensorProto::DataType::FLOAT:
    return sizeof(float);
  case TensorProto::DataType::DOUBLE:
    return sizeof(double);
  case TensorProto::DataType::INT32:
    return sizeof(int32_t);
  case TensorProto::DataType::INT64:
    return sizeof(int64_t);
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::BOOL:
    return 1;
  case TensorProto::DataType::UINT16:
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::FLOAT16:
  case TensorProto::DataType::BFLOAT16:
    return 2;
  case TensorProto::DataType::UINT32:
    return 4;
  case TensorProto::DataType::UINT64:
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
  return Tensor(name, static_cast<int32_t>(TensorProto::DataType::BOOL), shape, std::move(bytes));
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

const uint8_t *Tensor::AsBool() const {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(TensorProto::DataType::BOOL),
                      "Tensor data_type does not match the requested view type.");
  return data.data();
}

uint8_t *Tensor::AsBool() {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(TensorProto::DataType::BOOL),
                      "Tensor data_type does not match the requested view type.");
  return data.data();
}

const std::vector<std::string> &Tensor::AsStrings() const {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                      "Tensor data_type does not match the requested view type.");
  return string_data;
}

std::vector<std::string> &Tensor::AsStrings() {
  EXT_ENFORCE_INVALID(data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
