// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
                        const Tensor &y, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kLessOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a <= b ? 1 : 0; }, allocator);
}

template <typename TIn>
void LessOrEqualInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x,
                        const Tensor &y, Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kLessOrEqualName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a <= b ? 1 : 0; });
}
} // namespace

Tensor LessOrEqual::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessOrEqualAlloc<float>("FLOAT", DataType::FLOAT, x, y, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessOrEqualName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; }, rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessOrEqualName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a <= b ? 1 : 0; }, rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return LessOrEqualAlloc<int8_t>("INT8", DataType::INT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return LessOrEqualAlloc<int16_t>("INT16", DataType::INT16, x, y,
                                     rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return LessOrEqualAlloc<int32_t>("INT32", DataType::INT32, x, y,
                                     rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return LessOrEqualAlloc<int64_t>("INT64", DataType::INT64, x, y,
                                     rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return LessOrEqualAlloc<uint8_t>("UINT8", DataType::UINT8, x, y,
                                     rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return LessOrEqualAlloc<uint16_t>("UINT16", DataType::UINT16, x, y,
                                      rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return LessOrEqualAlloc<uint32_t>("UINT32", DataType::UINT32, x, y,
                                      rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return LessOrEqualAlloc<uint64_t>("UINT64", DataType::UINT64, x, y,
                                      rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kLessOrEqualName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
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
  case DataType::INT32:
    return LessOrEqualInPlace<int32_t>("INT32", DataType::INT32, x, y, output);
  case DataType::INT64:
    return LessOrEqualInPlace<int64_t>("INT64", DataType::INT64, x, y, output);
  case DataType::UINT8:
    return LessOrEqualInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return LessOrEqualInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return LessOrEqualInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return LessOrEqualInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    EXT_THROW_INVALID(kLessOrEqualName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void LessOrEqual::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
