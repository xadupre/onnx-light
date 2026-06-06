// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/elementwise_helpers.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kXorName = "kernel::Xor";
constexpr const char *kBoolName = "BOOL";
constexpr auto kXorOp = [](uint8_t a, uint8_t b) -> uint8_t {
  return ((a != 0) != (b != 0)) ? 1 : 0;
};
} // namespace

Tensor Xor::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kXorName, kBoolName, DataType::BOOL, x, y,
                                                          kXorOp);
}

void Xor::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kXorName, kBoolName, DataType::BOOL, x, y, output,
                                              kXorOp);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
