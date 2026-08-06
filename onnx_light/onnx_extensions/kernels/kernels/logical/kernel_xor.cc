// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kXorName = "kernel::Xor";
constexpr const char *kBoolName = "BOOL";
constexpr auto kXorOp = [](uint8_t a, uint8_t b) -> uint8_t {
  return ((a != 0) != (b != 0)) ? 1 : 0;
};
} // namespace

Tensor Xor::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kXorName, kBoolName, DataType::BOOL, x, y,
                                                          kXorOp, rt ? rt->allocator() : nullptr);
}

void Xor::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kXorName, kBoolName, DataType::BOOL, x, y, output,
                                              kXorOp);
}

void Xor::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
