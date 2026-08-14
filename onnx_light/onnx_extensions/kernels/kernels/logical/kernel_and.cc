// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kAndName = "kernel::And";
constexpr const char *kBoolName = "BOOL";
constexpr uint32_t kTuningAbi = 1;
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::BOOL)};
constexpr auto kAndOp = [](uint8_t a, uint8_t b) -> uint8_t { return (a != 0 && b != 0) ? 1 : 0; };
} // namespace

And::And(const KernelContext &ctx) : KernelBase(ctx), tuning_(kParallelForGrainSize) {}

void And::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("And", kSupportedElementTypes, kParallelForGrainSize,
                                        kTuningAbi);
}

KernelTuningKey And::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("And", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void And::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("And", parameters, tuning_, kTuningAbi);
}

Tensor And::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kAndName, kBoolName, DataType::BOOL, x, y,
                                                          kAndOp, rt ? rt->allocator() : nullptr,
                                                          tuning_.parallel_minimum_elements);
}

void And::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kAndName, kBoolName, DataType::BOOL, x, y, output,
                                              kAndOp, tuning_.parallel_minimum_elements);
}

void And::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
