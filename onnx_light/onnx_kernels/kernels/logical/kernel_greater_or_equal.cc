// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/elementwise_helpers.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include "onnx_kernels/runtime_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kGreaterOrEqualName = "kernel::GreaterOrEqual";
constexpr const char *kBoolName = "BOOL";

// Helper used to label the input dtype branch in error messages and to
// route dispatch through a single templated entry point. The label is the
// upstream ONNX :class:`DataType` enumerator name so the resulting
// "kernel::GreaterOrEqual only supports <DTYPE> inputs." message is
// self-explanatory.
template <typename TIn>
Tensor GreaterOrEqualAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x,
                           const Tensor &y, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kGreaterOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a >= b ? 1 : 0; }, allocator);
}

template <typename TIn>
void GreaterOrEqualInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x,
                           const Tensor &y, Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kGreaterOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a >= b ? 1 : 0; });
}
} // namespace

Tensor GreaterOrEqual::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterOrEqualAlloc<float>("FLOAT", DataType::FLOAT, x, y, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kGreaterOrEqualName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a >= b ? 1 : 0; },
          rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kGreaterOrEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a >= b ? 1 : 0; },
          rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return GreaterOrEqualAlloc<int8_t>("INT8", DataType::INT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return GreaterOrEqualAlloc<int16_t>("INT16", DataType::INT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return GreaterOrEqualAlloc<int32_t>("INT32", DataType::INT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return GreaterOrEqualAlloc<int64_t>("INT64", DataType::INT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return GreaterOrEqualAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return GreaterOrEqualAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return GreaterOrEqualAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return GreaterOrEqualAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kGreaterOrEqualName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void GreaterOrEqual::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterOrEqualInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kGreaterOrEqualName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a >= b ? 1 : 0; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kGreaterOrEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a >= b ? 1 : 0; });
  case DataType::INT8:
    return GreaterOrEqualInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return GreaterOrEqualInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::INT32:
    return GreaterOrEqualInPlace<int32_t>("INT32", DataType::INT32, x, y, output);
  case DataType::INT64:
    return GreaterOrEqualInPlace<int64_t>("INT64", DataType::INT64, x, y, output);
  case DataType::UINT8:
    return GreaterOrEqualInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return GreaterOrEqualInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return GreaterOrEqualInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return GreaterOrEqualInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    EXT_THROW_INVALID(kGreaterOrEqualName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
