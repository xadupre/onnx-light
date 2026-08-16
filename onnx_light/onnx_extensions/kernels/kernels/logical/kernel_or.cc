// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kOrName = "kernel::Or";
constexpr const char *kBoolName = "BOOL";
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::BOOL)};
constexpr auto kOrOp = [](uint8_t a, uint8_t b) -> uint8_t { return (a != 0 || b != 0) ? 1 : 0; };
} // namespace

Or::Or(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Or", kSupportedElementTypes, kParallelForGrainSize) {}

void Or::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Or", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Or::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y,
                                                          kOrOp, rt ? rt->allocator() : nullptr,
                                                          tuning().parallel_minimum_elements);
}

void Or::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y, output,
                                              kOrOp, tuning().parallel_minimum_elements);
}

void Or::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
