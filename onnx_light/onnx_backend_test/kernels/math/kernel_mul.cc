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
constexpr const char *kMulName = "kernel::Mul";

template <typename T>
Tensor MulAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAlloc<T, T>(kMulName, dtype_name, dtype, x, y,
                                              [](T a, T b) -> T { return a * b; });
}

template <typename T>
void MulInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output) {
  detail::BinaryElementwise<T, T>(kMulName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return a * b; });
}

constexpr const char *kSupportedMulTypesMsg =
    " only supports FLOAT, INT8, INT16, UINT8, UINT16, UINT32 and UINT64 inputs.";
} // namespace

Tensor Mul::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return MulAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::INT8:
    return MulAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return MulAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::UINT8:
    return MulAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return MulAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return MulAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return MulAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kMulName) + kSupportedMulTypesMsg);
  }
}

void Mul::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return MulInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::INT8:
    return MulInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return MulInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::UINT8:
    return MulInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return MulInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return MulInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return MulInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kMulName) + kSupportedMulTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
