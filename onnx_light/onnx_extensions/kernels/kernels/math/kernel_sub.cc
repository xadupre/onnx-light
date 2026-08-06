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
constexpr const char *kSubName = "kernel::Sub";

// Templated dispatch helpers. The label is the upstream ONNX
// :class:`DataType` enumerator name so the resulting
// "kernel::Sub only supports <DTYPE> inputs." message is self-explanatory.
template <typename T>
Tensor SubAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kSubName, dtype_name, dtype, x, y, [](T a, T b) -> T { return a - b; }, allocator);
}

template <typename T>
void SubInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output) {
  detail::BinaryElementwise<T, T>(kSubName, dtype_name, dtype, x, y, output,
                                  [](T a, T b) -> T { return a - b; });
}

constexpr const char *kSupportedSubTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";
} // namespace

Tensor Sub::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return SubAlloc<float>("FLOAT", DataType::FLOAT, x, y, rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return SubAlloc<double>("DOUBLE", DataType::DOUBLE, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT8:
    return SubAlloc<int8_t>("INT8", DataType::INT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return SubAlloc<int16_t>("INT16", DataType::INT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return SubAlloc<int32_t>("INT32", DataType::INT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return SubAlloc<int64_t>("INT64", DataType::INT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return SubAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return SubAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return SubAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return SubAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kSubName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat, FloatToFloat16Bits,
        [](float a, float b) { return a - b; }, rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kSubName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [](float a, float b) { return a - b; }, rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kSubName, ": unsupported data type ", x.data_type, kSupportedSubTypesMsg);
  }
}

void Sub::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return SubInPlace<float>("FLOAT", DataType::FLOAT, x, y, output);
  case DataType::DOUBLE:
    return SubInPlace<double>("DOUBLE", DataType::DOUBLE, x, y, output);
  case DataType::INT8:
    return SubInPlace<int8_t>("INT8", DataType::INT8, x, y, output);
  case DataType::INT16:
    return SubInPlace<int16_t>("INT16", DataType::INT16, x, y, output);
  case DataType::INT32:
    return SubInPlace<int32_t>("INT32", DataType::INT32, x, y, output);
  case DataType::INT64:
    return SubInPlace<int64_t>("INT64", DataType::INT64, x, y, output);
  case DataType::UINT8:
    return SubInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output);
  case DataType::UINT16:
    return SubInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output);
  case DataType::UINT32:
    return SubInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output);
  case DataType::UINT64:
    return SubInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(kSubName, "FLOAT16", DataType::FLOAT16, x, y, output,
                                         Float16BitsToFloat, FloatToFloat16Bits,
                                         [](float a, float b) { return a - b; });
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(kSubName, "BFLOAT16", DataType::BFLOAT16, x, y, output,
                                         Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                         [](float a, float b) { return a - b; });
  default:
    EXT_THROW_INVALID(kSubName, ": unsupported data type ", x.data_type, kSupportedSubTypesMsg);
  }
}

void Sub::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
