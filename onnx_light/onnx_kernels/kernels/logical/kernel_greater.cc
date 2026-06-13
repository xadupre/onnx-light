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
constexpr const char *kGreaterName = "kernel::Greater";
constexpr const char *kBoolName = "BOOL";

// Helper used to label the input dtype branch in error messages and to
// route dispatch through a single templated entry point. The label is the
// upstream ONNX :class:`DataType` enumerator name so the
// resulting "kernel::Greater only supports <DTYPE> inputs." message is
// self-explanatory.
template <typename TIn>
Tensor GreaterAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kGreaterName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a > b ? 1 : 0; });
}

template <typename TIn>
void GreaterInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                    Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kGreaterName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a > b ? 1 : 0; });
}
} // namespace

Tensor Greater::operator()(const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterAlloc<float>("FLOAT", DataType::FLOAT, x, y);
  case DataType::INT8:
    return GreaterAlloc<int8_t>("INT8", DataType::INT8, x, y);
  case DataType::INT16:
    return GreaterAlloc<int16_t>("INT16", DataType::INT16, x, y);
  case DataType::UINT8:
    return GreaterAlloc<uint8_t>("UINT8", DataType::UINT8, x, y);
  case DataType::UINT16:
    return GreaterAlloc<uint16_t>("UINT16", DataType::UINT16, x, y);
  case DataType::UINT32:
    return GreaterAlloc<uint32_t>("UINT32", DataType::UINT32, x, y);
  case DataType::UINT64:
    return GreaterAlloc<uint64_t>("UINT64", DataType::UINT64, x, y);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(kGreaterName, "FLOAT16", DataType::FLOAT16, x,
                                                     y, Float16BitsToFloat,
                                                     [](float a, float b) { return a > b; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(kGreaterName, "BFLOAT16", DataType::BFLOAT16,
                                                     x, y, Bfloat16BitsToFloat,
                                                     [](float a, float b) { return a > b; });
  default:
    EXT_THROW_INVALID(kGreaterName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, UINT8, "
                      "UINT16, UINT32 and UINT64 inputs.");
  }
}

void Greater::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::INT8:
    return GreaterInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return GreaterInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::UINT8:
    return GreaterInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return GreaterInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return GreaterInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return GreaterInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(kGreaterName, "FLOAT16", DataType::FLOAT16, x, y,
                                                output, Float16BitsToFloat,
                                                [](float a, float b) { return a > b; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(kGreaterName, "BFLOAT16", DataType::BFLOAT16, x, y,
                                                output, Bfloat16BitsToFloat,
                                                [](float a, float b) { return a > b; });
  default:
    EXT_THROW_INVALID(kGreaterName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, UINT8, "
                      "UINT16, UINT32 and UINT64 inputs.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
