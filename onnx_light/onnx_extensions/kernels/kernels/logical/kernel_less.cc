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
constexpr const char *kLessName = "kernel::Less";
constexpr const char *kBoolName = "BOOL";

template <typename TIn>
Tensor LessAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                 RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kLessName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a < b ? 1 : 0; }, allocator);
}

template <typename TIn>
void LessInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                 Tensor &output) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kLessName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a < b ? 1 : 0; });
}
} // namespace

Tensor Less::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessAlloc<float>("FLOAT", DataType::FLOAT, x, y, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a < b ? 1 : 0; }, rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kLessName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a < b ? 1 : 0; }, rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return LessAlloc<int8_t>("INT8", DataType::INT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return LessAlloc<int16_t>("INT16", DataType::INT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return LessAlloc<int32_t>("INT32", DataType::INT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return LessAlloc<int64_t>("INT64", DataType::INT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return LessAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return LessAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return LessAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return LessAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kLessName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void Less::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return LessInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kLessName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a < b ? 1 : 0; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kLessName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a < b ? 1 : 0; });
  case DataType::INT8:
    return LessInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return LessInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::INT32:
    return LessInPlace<int32_t>("INT32", DataType::INT32, x, y, output);
  case DataType::INT64:
    return LessInPlace<int64_t>("INT64", DataType::INT64, x, y, output);
  case DataType::UINT8:
    return LessInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return LessInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return LessInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return LessInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  default:
    EXT_THROW_INVALID(kLessName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void Less::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
