// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kMulName = "kernel::Mul";
constexpr uint32_t kTuningAbi = 1;
constexpr std::array<int32_t, 12> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
    static_cast<int32_t>(DataType::UINT8),   static_cast<int32_t>(DataType::UINT16),
    static_cast<int32_t>(DataType::UINT32),  static_cast<int32_t>(DataType::UINT64),
};

template <typename T>
Tensor MulAlloc(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                int64_t parallel_minimum_elements, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAlloc<T, T>(
      kMulName, dtype_name, dtype, x, y, [](T a, T b) -> T { return a * b; }, allocator,
      parallel_minimum_elements);
}

template <typename T>
void MulInPlace(const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
                Tensor &output, int64_t parallel_minimum_elements) {
  detail::BinaryElementwise<T, T>(
      kMulName, dtype_name, dtype, x, y, output, [](T a, T b) -> T { return a * b; },
      parallel_minimum_elements);
}

constexpr const char *kSupportedMulTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";
} // namespace

Mul::Mul(const KernelContext &ctx) : KernelBase(ctx), tuning_(kParallelForGrainSize) {}

void Mul::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Mul", kSupportedElementTypes, kParallelForGrainSize,
                                        kTuningAbi);
}

KernelTuningKey Mul::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Mul", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Mul::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Mul", parameters, tuning_, kTuningAbi);
}

Tensor Mul::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  if (rt != nullptr) {
    const Shape out_shape = detail::BroadcastShape("kernel::Mul", x.shape, y.shape);
    const int64_t out_count = out_shape.product();
    Tensor output = rt->MakeOutputTensor(0, x.data_type, out_shape,
                                         static_cast<size_t>(out_count) * ElementSize(x.data_type));
    (*this)(x, y, output);
    return output;
  }
  switch (x.data_type) {
  case DataType::FLOAT:
    return MulAlloc<float>("FLOAT", DataType::FLOAT, x, y, tuning_.parallel_minimum_elements,
                           nullptr);
  case DataType::DOUBLE:
    return MulAlloc<double>("DOUBLE", DataType::DOUBLE, x, y, tuning_.parallel_minimum_elements,
                            nullptr);
  case DataType::INT8:
    return MulAlloc<int8_t>("INT8", DataType::INT8, x, y, tuning_.parallel_minimum_elements,
                            nullptr);
  case DataType::INT16:
    return MulAlloc<int16_t>("INT16", DataType::INT16, x, y, tuning_.parallel_minimum_elements,
                             nullptr);
  case DataType::INT32:
    return MulAlloc<int32_t>("INT32", DataType::INT32, x, y, tuning_.parallel_minimum_elements,
                             nullptr);
  case DataType::INT64:
    return MulAlloc<int64_t>("INT64", DataType::INT64, x, y, tuning_.parallel_minimum_elements,
                             nullptr);
  case DataType::UINT8:
    return MulAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, tuning_.parallel_minimum_elements,
                             nullptr);
  case DataType::UINT16:
    return MulAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, tuning_.parallel_minimum_elements,
                              nullptr);
  case DataType::UINT32:
    return MulAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, tuning_.parallel_minimum_elements,
                              nullptr);
  case DataType::UINT64:
    return MulAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, tuning_.parallel_minimum_elements,
                              nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kMulName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat, FloatToFloat16Bits,
        [](float a, float b) { return a * b; }, nullptr, tuning_.parallel_minimum_elements);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwiseAlloc(
        kMulName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [](float a, float b) { return a * b; }, nullptr, tuning_.parallel_minimum_elements);
  default:
    EXT_THROW_INVALID(kMulName, ": unsupported data type ", x.data_type, kSupportedMulTypesMsg);
  }
}

void Mul::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  switch (x.data_type) {
  case DataType::FLOAT:
    return MulInPlace<float>("FLOAT", DataType::FLOAT, x, y, output,
                             tuning_.parallel_minimum_elements);
  case DataType::DOUBLE:
    return MulInPlace<double>("DOUBLE", DataType::DOUBLE, x, y, output,
                              tuning_.parallel_minimum_elements);
  case DataType::INT8:
    return MulInPlace<int8_t>("INT8", DataType::INT8, x, y, output,
                              tuning_.parallel_minimum_elements);
  case DataType::INT16:
    return MulInPlace<int16_t>("INT16", DataType::INT16, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::INT32:
    return MulInPlace<int32_t>("INT32", DataType::INT32, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::INT64:
    return MulInPlace<int64_t>("INT64", DataType::INT64, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::UINT8:
    return MulInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output,
                               tuning_.parallel_minimum_elements);
  case DataType::UINT16:
    return MulInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::UINT32:
    return MulInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::UINT64:
    return MulInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output,
                                tuning_.parallel_minimum_elements);
  case DataType::FLOAT16:
    return detail::BinaryHalfElementwise(
        kMulName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        FloatToFloat16Bits, [](float a, float b) { return a * b; },
        tuning_.parallel_minimum_elements);
  case DataType::BFLOAT16:
    return detail::BinaryHalfElementwise(
        kMulName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        FloatToBfloat16Bits, [](float a, float b) { return a * b; },
        tuning_.parallel_minimum_elements);
  default:
    EXT_THROW_INVALID(kMulName, ": unsupported data type ", x.data_type, kSupportedMulTypesMsg);
  }
}

void Mul::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
