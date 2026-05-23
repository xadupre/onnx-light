// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/simple_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

template <typename T>
Tensor MakeTyped(int32_t dtype, const std::string &name, const std::vector<int64_t> &shape,
                 const std::vector<T> &values) {
  int64_t expected = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      throw std::invalid_argument("Tensor shape dimensions must be non-negative.");
    }
    expected *= d;
  }
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument("Tensor values size does not match the product of shape.");
  }
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return Tensor(name, dtype, shape, std::move(bytes));
}

template <typename T> const T *TypedView(const Tensor &t, int32_t expected_dtype) {
  if (t.data_type != expected_dtype) {
    throw std::invalid_argument("Tensor data_type does not match the requested view type.");
  }
  return reinterpret_cast<const T *>(t.data.data());
}

template <typename T> T *TypedView(Tensor &t, int32_t expected_dtype) {
  if (t.data_type != expected_dtype) {
    throw std::invalid_argument("Tensor data_type does not match the requested view type.");
  }
  return reinterpret_cast<T *>(t.data.data());
}

} // namespace

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
  return MakeTyped<float>(TensorProto::DataType::FLOAT, name, shape, values);
}

Tensor Tensor::FromDouble(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<double> &values) {
  return MakeTyped<double>(TensorProto::DataType::DOUBLE, name, shape, values);
}

Tensor Tensor::FromInt32(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int32_t> &values) {
  return MakeTyped<int32_t>(TensorProto::DataType::INT32, name, shape, values);
}

Tensor Tensor::FromInt64(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<int64_t> &values) {
  return MakeTyped<int64_t>(TensorProto::DataType::INT64, name, shape, values);
}

const float *Tensor::AsFloat() const {
  return TypedView<float>(*this, TensorProto::DataType::FLOAT);
}
float *Tensor::AsFloat() { return TypedView<float>(*this, TensorProto::DataType::FLOAT); }
const double *Tensor::AsDouble() const {
  return TypedView<double>(*this, TensorProto::DataType::DOUBLE);
}
double *Tensor::AsDouble() { return TypedView<double>(*this, TensorProto::DataType::DOUBLE); }
const int32_t *Tensor::AsInt32() const {
  return TypedView<int32_t>(*this, TensorProto::DataType::INT32);
}
int32_t *Tensor::AsInt32() { return TypedView<int32_t>(*this, TensorProto::DataType::INT32); }
const int64_t *Tensor::AsInt64() const {
  return TypedView<int64_t>(*this, TensorProto::DataType::INT64);
}
int64_t *Tensor::AsInt64() { return TypedView<int64_t>(*this, TensorProto::DataType::INT64); }

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
