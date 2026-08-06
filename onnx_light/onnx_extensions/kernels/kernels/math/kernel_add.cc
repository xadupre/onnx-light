// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kAddName = "kernel::Add";

template <typename T>
Tensor AddAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kAddName, dtype_name, dtype, x, y, [](T a, T b) -> T { return a + b; }, allocator);
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

Tensor Add::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return AddAlloc<float>("FLOAT", DataType::FLOAT, x, y, rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return AddAlloc<double>("DOUBLE", DataType::DOUBLE, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return AddAlloc<int8_t>("INT8", DataType::INT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return AddAlloc<int16_t>("INT16", DataType::INT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return AddAlloc<int32_t>("INT32", DataType::INT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return AddAlloc<int64_t>("INT64", DataType::INT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return AddAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return AddAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return AddAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return AddAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kAddName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat, FloatToFloat16Bits,
        [](float a, float b) { return a + b; }, rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kAddName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [](float a, float b) { return a + b; }, rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kAddName, ": unsupported data type ", x.data_type, kSupportedAddTypesMsg);
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
    EXT_THROW_INVALID(kAddName, ": unsupported data type ", x.data_type, kSupportedAddTypesMsg);
  }
}

void Add::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
