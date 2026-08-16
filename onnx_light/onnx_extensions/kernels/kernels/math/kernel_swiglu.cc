// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kName = "kernel::SwiGLU";
constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

// SwiGLU gate: Swish_alpha(a) * b where Swish_alpha(a) = a * sigmoid(alpha * a).
template <typename T> inline T SwiGLUOp(T a, T b, T alpha) {
  const T s = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-alpha * a));
  return static_cast<T>(a * s * b);
}

// SwiGLU forbids broadcasting: A and B must have identical shapes. Enforce this
// before delegating to the element-wise helpers (which would otherwise allow
// numpy-style broadcasting).
void RequireEqualShapes(const Tensor &a, const Tensor &b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, kName,
                      ": inputs A and B must have identical shapes (broadcasting is not applied).");
}

template <typename T>
Tensor SwiGLUAlloc(const char *dtype_name, int32_t dtype, const Tensor &a, const Tensor &b, T alpha,
                   int64_t grain, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kName, dtype_name, dtype, a, b, [alpha](T x, T y) -> T { return SwiGLUOp<T>(x, y, alpha); },
      allocator, grain);
}

template <typename T>
void SwiGLUInPlace(const char *dtype_name, int32_t dtype, const Tensor &a, const Tensor &b, T alpha,
                   Tensor &output, int64_t grain) {
  detail::BinaryElementwise<T, T>(
      kName, dtype_name, dtype, a, b, output,
      [alpha](T x, T y) -> T { return SwiGLUOp<T>(x, y, alpha); }, grain);
}

constexpr const char *kSupportedTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 tensors.";
} // namespace

SwiGLU::SwiGLU(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "SwiGLU", kSupportedElementTypes, kParallelForGrainSize) {}

void SwiGLU::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("SwiGLU", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor SwiGLU::operator()(const Tensor &a, const Tensor &b, float alpha, RuntimeContext *rt) const {
  RequireEqualShapes(a, b);
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (a.data_type) {
  case DataType::FLOAT:
    return SwiGLUAlloc<float>("FLOAT", DataType::FLOAT, a, b, alpha, grain,
                              rt ? rt->allocator() : nullptr);
  case DataType::DOUBLE:
    return SwiGLUAlloc<double>("DOUBLE", DataType::DOUBLE, a, b, static_cast<double>(alpha), grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kName, "FLOAT16", DataType::FLOAT16, a, b, Float16BitsToFloat, FloatToFloat16Bits,
        [alpha](float x, float y) { return SwiGLUOp<float>(x, y, alpha); },
        rt ? rt->allocator() : nullptr, grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kName, "BFLOAT16", DataType::BFLOAT16, a, b, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [alpha](float x, float y) { return SwiGLUOp<float>(x, y, alpha); },
        rt ? rt->allocator() : nullptr, grain);
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", a.data_type, kSupportedTypesMsg);
  }
}

void SwiGLU::operator()(const Tensor &a, const Tensor &b, float alpha, Tensor &output) const {
  RequireEqualShapes(a, b);
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (a.data_type) {
  case DataType::FLOAT:
    return SwiGLUInPlace<float>("FLOAT", DataType::FLOAT, a, b, alpha, output, grain);
  case DataType::DOUBLE:
    return SwiGLUInPlace<double>("DOUBLE", DataType::DOUBLE, a, b, static_cast<double>(alpha),
                                 output, grain);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(
        kName, "FLOAT16", DataType::FLOAT16, a, b, output, Float16BitsToFloat, FloatToFloat16Bits,
        [alpha](float x, float y) { return SwiGLUOp<float>(x, y, alpha); }, grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(
        kName, "BFLOAT16", DataType::BFLOAT16, a, b, output, Bfloat16BitsToFloat,
        FloatToBfloat16Bits, [alpha](float x, float y) { return SwiGLUOp<float>(x, y, alpha); },
        grain);
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", a.data_type, kSupportedTypesMsg);
  }
}

void SwiGLU::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const Tensor &a = GetInput(node, 0, rt.tensors());
  const Tensor &b = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(a, b, alpha, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
