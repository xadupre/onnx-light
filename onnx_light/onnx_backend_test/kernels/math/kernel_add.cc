// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kAddName = "kernel::Add";

template <typename T>
Tensor AddAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<T, T>(kAddName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return a + b; });
}

template <typename T>
void AddInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output) {
  detail::BinaryElementwise<T, T>(kAddName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return a + b; });
}

constexpr const char *kSupportedAddTypesMsg =
    " only supports FLOAT, INT8, INT16, UINT8, UINT16, UINT32 and UINT64 inputs.";
} // namespace

Tensor Add::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case TensorProto::DataType::FLOAT:
    return AddAlloc<float>("FLOAT", TensorProto::DataType::FLOAT, x, y);
  case TensorProto::DataType::INT8:
    return AddAlloc<int8_t>("INT8", TensorProto::DataType::INT8, x, y);
  case TensorProto::DataType::INT16:
    return AddAlloc<int16_t>("INT16", TensorProto::DataType::INT16, x, y);
  case TensorProto::DataType::UINT8:
    return AddAlloc<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y);
  case TensorProto::DataType::UINT16:
    return AddAlloc<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y);
  case TensorProto::DataType::UINT32:
    return AddAlloc<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y);
  case TensorProto::DataType::UINT64:
    return AddAlloc<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kAddName) + kSupportedAddTypesMsg);
  }
}

void Add::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case TensorProto::DataType::FLOAT:
    return AddInPlace<float>("FLOAT", TensorProto::DataType::FLOAT, x, y, output);
  case TensorProto::DataType::INT8:
    return AddInPlace<int8_t>("INT8", TensorProto::DataType::INT8, x, y, output);
  case TensorProto::DataType::INT16:
    return AddInPlace<int16_t>("INT16", TensorProto::DataType::INT16, x, y, output);
  case TensorProto::DataType::UINT8:
    return AddInPlace<uint8_t>("UINT8", TensorProto::DataType::UINT8, x, y, output);
  case TensorProto::DataType::UINT16:
    return AddInPlace<uint16_t>("UINT16", TensorProto::DataType::UINT16, x, y, output);
  case TensorProto::DataType::UINT32:
    return AddInPlace<uint32_t>("UINT32", TensorProto::DataType::UINT32, x, y, output);
  case TensorProto::DataType::UINT64:
    return AddInPlace<uint64_t>("UINT64", TensorProto::DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kAddName) + kSupportedAddTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
