// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kGreaterName = "kernel::Greater";
constexpr const char *kBoolName = "BOOL";
constexpr std::array<int32_t, 11> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),    static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16), static_cast<int32_t>(DataType::INT8),
    static_cast<int32_t>(DataType::INT16),    static_cast<int32_t>(DataType::INT32),
    static_cast<int32_t>(DataType::INT64),    static_cast<int32_t>(DataType::UINT8),
    static_cast<int32_t>(DataType::UINT16),   static_cast<int32_t>(DataType::UINT32),
    static_cast<int32_t>(DataType::UINT64),
};

// Helper used to label the input dtype branch in error messages and to
// route dispatch through a single templated entry point. The label is the
// upstream ONNX :class:`DataType` enumerator name so the
// resulting "kernel::Greater only supports <DTYPE> inputs." message is
// self-explanatory.
template <typename TIn>
Tensor GreaterAlloc(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                    int64_t grain, RawBufferAllocator *allocator = nullptr) {
  return detail::BinaryElementwiseAllocInOut<TIn, uint8_t>(
      kGreaterName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y,
      [](TIn a, TIn b) -> uint8_t { return a > b ? 1 : 0; }, allocator, grain);
}

template <typename TIn>
void GreaterInPlace(const char *in_dtype_name, int32_t in_dtype, const Tensor &x, const Tensor &y,
                    Tensor &output, int64_t grain) {
  detail::BinaryElementwiseInOut<TIn, uint8_t>(
      kGreaterName, in_dtype_name, in_dtype, kBoolName, DataType::BOOL, x, y, output,
      [](TIn a, TIn b) -> uint8_t { return a > b ? 1 : 0; }, grain);
}
} // namespace

Greater::Greater(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Greater", kSupportedElementTypes, kParallelForGrainSize) {}

void Greater::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Greater", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Greater::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterAlloc<float>("FLOAT", DataType::FLOAT, x, y, grain,
                               rt ? rt->allocator() : nullptr);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kGreaterName, "FLOAT16", DataType::FLOAT16, x, y, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a > b ? 1 : 0; }, rt ? rt->allocator() : nullptr,
        grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwiseAlloc(
        kGreaterName, "BFLOAT16", DataType::BFLOAT16, x, y, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a > b ? 1 : 0; }, rt ? rt->allocator() : nullptr,
        grain);
  case DataType::INT8:
    return GreaterAlloc<int8_t>("INT8", DataType::INT8, x, y, grain,
                                rt ? rt->allocator() : nullptr);
  case DataType::INT16:
    return GreaterAlloc<int16_t>("INT16", DataType::INT16, x, y, grain,
                                 rt ? rt->allocator() : nullptr);
  case DataType::INT32:
    return GreaterAlloc<int32_t>("INT32", DataType::INT32, x, y, grain,
                                 rt ? rt->allocator() : nullptr);
  case DataType::INT64:
    return GreaterAlloc<int64_t>("INT64", DataType::INT64, x, y, grain,
                                 rt ? rt->allocator() : nullptr);
  case DataType::UINT8:
    return GreaterAlloc<uint8_t>("UINT8", DataType::UINT8, x, y, grain,
                                 rt ? rt->allocator() : nullptr);
  case DataType::UINT16:
    return GreaterAlloc<uint16_t>("UINT16", DataType::UINT16, x, y, grain,
                                  rt ? rt->allocator() : nullptr);
  case DataType::UINT32:
    return GreaterAlloc<uint32_t>("UINT32", DataType::UINT32, x, y, grain,
                                  rt ? rt->allocator() : nullptr);
  case DataType::UINT64:
    return GreaterAlloc<uint64_t>("UINT64", DataType::UINT64, x, y, grain,
                                  rt ? rt->allocator() : nullptr);
  default:
    EXT_THROW_INVALID(kGreaterName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void Greater::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (x.data_type) {
  case DataType::FLOAT:
    return GreaterInPlace<float>("FLOAT", DataType::FLOAT, x, y, output, grain);
  case DataType::FLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kGreaterName, "FLOAT16", DataType::FLOAT16, x, y, output, Float16BitsToFloat,
        [](float a, float b) -> uint8_t { return a > b ? 1 : 0; }, grain);
  case DataType::BFLOAT16:
    return detail::BinaryHalfCompareElementwise(
        kGreaterName, "BFLOAT16", DataType::BFLOAT16, x, y, output, Bfloat16BitsToFloat,
        [](float a, float b) -> uint8_t { return a > b ? 1 : 0; }, grain);
  case DataType::INT8:
    return GreaterInPlace<int8_t>("INT8", DataType::INT8, x, y, output, grain);
  case DataType::INT16:
    return GreaterInPlace<int16_t>("INT16", DataType::INT16, x, y, output, grain);
  case DataType::INT32:
    return GreaterInPlace<int32_t>("INT32", DataType::INT32, x, y, output, grain);
  case DataType::INT64:
    return GreaterInPlace<int64_t>("INT64", DataType::INT64, x, y, output, grain);
  case DataType::UINT8:
    return GreaterInPlace<uint8_t>("UINT8", DataType::UINT8, x, y, output, grain);
  case DataType::UINT16:
    return GreaterInPlace<uint16_t>("UINT16", DataType::UINT16, x, y, output, grain);
  case DataType::UINT32:
    return GreaterInPlace<uint32_t>("UINT32", DataType::UINT32, x, y, output, grain);
  case DataType::UINT64:
    return GreaterInPlace<uint64_t>("UINT64", DataType::UINT64, x, y, output, grain);
  default:
    EXT_THROW_INVALID(kGreaterName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
}

void Greater::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
