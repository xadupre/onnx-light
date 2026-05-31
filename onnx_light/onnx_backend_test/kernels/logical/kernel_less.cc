// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kLessName = "kernel::Less";
constexpr const char *kBoolName = "BOOL";

template <typename TIn>
Tensor LessAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kLessName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a < b ? 1 : 0; });
}

template <typename TIn>
void LessInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                 Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kLessName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a < b ? 1 : 0; });
}
} // namespace

Tensor Less::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::INT8:
    return LessAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return LessAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::UINT8:
    return LessAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return LessAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return LessAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return LessAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kLessName) +
                                " only supports FLOAT, INT8, INT16, UINT8, UINT16, UINT32 "
                                "and UINT64 inputs.");
  }
}

void Less::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::INT8:
    return LessInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return LessInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::UINT8:
    return LessInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return LessInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return LessInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return LessInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kLessName) +
                                " only supports FLOAT, INT8, INT16, UINT8, UINT16, UINT32 "
                                "and UINT64 inputs.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
