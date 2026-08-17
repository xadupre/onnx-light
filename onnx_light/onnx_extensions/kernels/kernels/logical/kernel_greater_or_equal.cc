// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kGreaterOrEqualName = "kernel::GreaterOrEqual";
constexpr std::array<int32_t, 11> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),    static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16), static_cast<int32_t>(DataType::INT8),
    static_cast<int32_t>(DataType::INT16),    static_cast<int32_t>(DataType::INT32),
    static_cast<int32_t>(DataType::INT64),    static_cast<int32_t>(DataType::UINT8),
    static_cast<int32_t>(DataType::UINT16),   static_cast<int32_t>(DataType::UINT32),
    static_cast<int32_t>(DataType::UINT64),
};
constexpr auto kGreaterOrEqualOp = [](auto a, auto b) -> uint8_t { return a >= b ? 1 : 0; };
} // namespace

GreaterOrEqual::GreaterOrEqual(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GreaterOrEqual", kSupportedElementTypes, kParallelForGrainSize) {}

void GreaterOrEqual::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("GreaterOrEqual", kSupportedElementTypes,
                                        kParallelForGrainSize);
}

Tensor GreaterOrEqual::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  if (rt != nullptr) {
    const Shape out_shape = detail::BroadcastShape("kernel::GreaterOrEqual", x.shape, y.shape);
    const int64_t out_count = out_shape.product();
    Tensor output = rt->MakeOutputTensor(
        0, DataType::BOOL, out_shape, static_cast<size_t>(out_count) * ElementSize(DataType::BOOL));
    (*this)(x, y, output);
    return output;
  }
  return detail::BinaryComparisonAlloc(kGreaterOrEqualName, x, y, kGreaterOrEqualOp, nullptr,
                                       tuning().parallel_minimum_elements);
}

void GreaterOrEqual::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryComparison(kGreaterOrEqualName, x, y, output, kGreaterOrEqualOp,
                           tuning().parallel_minimum_elements);
}

void GreaterOrEqual::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
