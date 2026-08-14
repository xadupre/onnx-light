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
constexpr const char *kLessName = "kernel::Less";
constexpr std::array<int32_t, 11> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),    static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16), static_cast<int32_t>(DataType::INT8),
    static_cast<int32_t>(DataType::INT16),    static_cast<int32_t>(DataType::INT32),
    static_cast<int32_t>(DataType::INT64),    static_cast<int32_t>(DataType::UINT8),
    static_cast<int32_t>(DataType::UINT16),   static_cast<int32_t>(DataType::UINT32),
    static_cast<int32_t>(DataType::UINT64),
};
constexpr auto kLessOp = [](auto a, auto b) -> uint8_t { return a < b ? 1 : 0; };
} // namespace

Less::Less(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Less", kSupportedElementTypes, kParallelForGrainSize) {}

void Less::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Less", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Less::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return detail::BinaryComparisonAlloc(kLessName, x, y, kLessOp, rt ? rt->allocator() : nullptr,
                                       tuning().parallel_minimum_elements);
}

void Less::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryComparison(kLessName, x, y, output, kLessOp, tuning().parallel_minimum_elements);
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
