// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/elementwise_helpers.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
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
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";
} // namespace

Tensor Add::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return AddAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::DOUBLE:
    return AddAlloc<double>("DOUBLE", DataType::DOUBLE, x, y);
  case DataType::INT8:
    return AddAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return AddAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::INT32:
    return AddAlloc<int32_t>("INT32", DataType::INT32, x, y);
  case DataType::INT64:
    return AddAlloc<int64_t>("INT64", DataType::INT64, x, y);
  case DataType::UINT8:
    return AddAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return AddAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return AddAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return AddAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(kAddName, "FLOAT16", DataType::FLOAT16, x, y,
                                              Float16BitsToFloat, FloatToFloat16Bits,
                                              [](float a, float b) { return a + b; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(kAddName, "BFLOAT16", DataType::BFLOAT16, x, y,
                                              Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                              [](float a, float b) { return a + b; });
  default:
    throw std::invalid_argument(std::string(kAddName) + kSupportedAddTypesMsg);
  }
}

void Add::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return AddInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::DOUBLE:
    return AddInPlace<double>("DOUBLE", DataType::DOUBLE, x, y, output);
  case DataType::INT8:
    return AddInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return AddInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::INT32:
    return AddInPlace<int32_t>("INT32", DataType::INT32, x, y, output);
  case DataType::INT64:
    return AddInPlace<int64_t>("INT64", DataType::INT64, x, y, output);
  case DataType::UINT8:
    return AddInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return AddInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return AddInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return AddInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(kAddName, "FLOAT16", DataType::FLOAT16, x, y, output,
                                         Float16BitsToFloat, FloatToFloat16Bits,
                                         [](float a, float b) { return a + b; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(kAddName, "BFLOAT16", DataType::BFLOAT16, x, y, output,
                                         Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                         [](float a, float b) { return a + b; });
  default:
    throw std::invalid_argument(std::string(kAddName) + kSupportedAddTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
