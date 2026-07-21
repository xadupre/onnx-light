// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/onnx_kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kOrName = "kernel::Or";
constexpr const char *kBoolName = "BOOL";
constexpr auto kOrOp = [](uint8_t a, uint8_t b) -> uint8_t { return (a != 0 || b != 0) ? 1 : 0; };
} // namespace

Tensor Or::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y,
                                                          kOrOp, rt ? rt->allocator() : nullptr);
}

void Or::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y, output,
                                              kOrOp);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
