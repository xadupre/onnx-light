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
constexpr const char *kSubName = "kernel::Sub";

// Templated dispatch helpers. The label is the upstream ONNX
// :class:`DataType` enumerator name so the resulting
// "kernel::Sub only supports <DTYPE> inputs." message is self-explanatory.
template <typename T>
Tensor SubAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<T, T>(kSubName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return a - b; });
}

template <typename T>
void SubInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output) {
  detail::BinaryElementwise<T, T>(kSubName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return a - b; });
}

constexpr const char *kUnsupportedMsg =
    " only supports FLOAT, INT8, INT16, UINT8, UINT16, UINT32 and UINT64 inputs.";
} // namespace

Tensor Sub::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return SubAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::INT8:
    return SubAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return SubAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::UINT8:
    return SubAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return SubAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return SubAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return SubAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kSubName) + kUnsupportedMsg);
  }
}

void Sub::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return SubInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::INT8:
    return SubInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return SubInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::UINT8:
    return SubInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return SubInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return SubInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return SubInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kSubName) + kUnsupportedMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
