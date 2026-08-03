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

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kPReluName = "kernel::PRelu";

// Branches on the sign of ``x`` rather than evaluating
// ``max(0, x) + slope * min(0, x)``: the latter form produces ``NaN``
// for ``+inf`` / ``-inf`` inputs because the arithmetic on the masked
// branch still evaluates ``slope * inf`` before the sum cancels it.
// See microsoft/onnxruntime#28732 for the regression that motivated
// this reference kernel.
template <typename T> inline T PReluOp(T x, T slope) {
  return x < static_cast<T>(0) ? static_cast<T>(slope * x) : x;
}

template <typename T>
Tensor PReluAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &slope,
                  RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kPReluName, dtype_name, dtype, x, slope, [](T a, T b) -> T { return PReluOp<T>(a, b); },
      allocator);
}

template <typename T>
void PReluInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &slope,
                  Tensor &output) {
  detail::BinaryElementwise<T, T>(kPReluName, dtype_name, dtype, x, slope, output,
                                  [](T a, T b) -> T { return PReluOp<T>(a, b); });
}

constexpr const char *kSupportedPReluTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT32, INT64, UINT32 and UINT64 inputs.";
} // namespace

Tensor PRelu::operator()(const Tensor &x, const Tensor &slope, RuntimeContext *rt) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PReluAlloc<float>("FLOAT", DataType::FLOAT, x, slope, rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return PReluAlloc<double>("DOUBLE", DataType::DOUBLE, x, slope, rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return PReluAlloc<int32_t>("INT32", DataType::INT32, x, slope, rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return PReluAlloc<int64_t>("INT64", DataType::INT64, x, slope, rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return PReluAlloc<uint32_t>("UINT32", DataType::UINT32, x, slope,
                                rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return PReluAlloc<uint64_t>("UINT64", DataType::UINT64, x, slope,
                                rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kPReluName, "FLOAT16", DataType::FLOAT16, x, slope, Float16BitsToFloat, FloatToFloat16Bits,
        [](float a, float b) { return PReluOp<float>(a, b); }, rt ? rt->allocator() : nullptr);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kPReluName, "BFLOAT16", DataType::BFLOAT16, x, slope, Bfloat16BitsToFloat,
        FloatToBfloat16Bits, [](float a, float b) { return PReluOp<float>(a, b); },
        rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kPReluName, ": unsupported data type ", x.data_type, kSupportedPReluTypesMsg);
  }
}

void PRelu::operator()(const Tensor &x, const Tensor &slope, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PReluInPlace<float>("FLOAT", DataType::FLOAT, x, slope, output);
  case DataType::DOUBLE:
    return PReluInPlace<double>("DOUBLE", DataType::DOUBLE, x, slope, output);
  case DataType::INT32:
    return PReluInPlace<int32_t>("INT32", DataType::INT32, x, slope, output);
  case DataType::INT64:
    return PReluInPlace<int64_t>("INT64", DataType::INT64, x, slope, output);
  case DataType::UINT32:
    return PReluInPlace<uint32_t>("UINT32", DataType::UINT32, x, slope, output);
  case DataType::UINT64:
    return PReluInPlace<uint64_t>("UINT64", DataType::UINT64, x, slope, output);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(kPReluName, "FLOAT16", DataType::FLOAT16, x, slope, output,
                                         Float16BitsToFloat, FloatToFloat16Bits,
                                         [](float a, float b) { return PReluOp<float>(a, b); });
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(kPReluName, "BFLOAT16", DataType::BFLOAT16, x, slope,
                                         output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                         [](float a, float b) { return PReluOp<float>(a, b); });
  default:
    EXT_THROW_INVALID(kPReluName, ": unsupported data type ", x.data_type, kSupportedPReluTypesMsg);
  }
}

void PRelu::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
