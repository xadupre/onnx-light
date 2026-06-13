// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/elementwise_helpers.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kLessOrEqualName = "kernel::LessOrEqual";
constexpr const char *kBoolName = "BOOL";

// Helper used to label the input dtype branch in error messages and to
// route dispatch through a single templated entry point. The label is the
// upstream ONNX :class:`DataType` enumerator name so the resulting
// "kernel::LessOrEqual only supports <DTYPE> inputs." message is
// self-explanatory.
template <typename TIn>
Tensor LessOrEqualAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x,
                        const Tensor &y) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kLessOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a <= b ? 1 : 0; });
}

template <typename TIn>
void LessOrEqualInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x,
                        const Tensor &y, Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kLessOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a <= b ? 1 : 0; });
}
} // namespace

Tensor LessOrEqual::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessOrEqualAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessOrEqualName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessOrEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; });
  case DataType::INT8:
    return LessOrEqualAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return LessOrEqualAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::UINT8:
    return LessOrEqualAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return LessOrEqualAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return LessOrEqualAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return LessOrEqualAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  default:
    throw std::invalid_argument(std::string(kLessOrEqualName) +
                                " only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, UINT8, "
                                "UINT16, UINT32 and UINT64 inputs.");
  }
}

void LessOrEqual::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessOrEqualInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kLessOrEqualName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kLessOrEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; });
  case DataType::INT8:
    return LessOrEqualInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return LessOrEqualInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::UINT8:
    return LessOrEqualInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return LessOrEqualInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return LessOrEqualInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return LessOrEqualInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    throw std::invalid_argument(std::string(kLessOrEqualName) +
                                " only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, UINT8, "
                                "UINT16, UINT32 and UINT64 inputs.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
