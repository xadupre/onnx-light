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
constexpr const char *kAndName = "kernel::And";
constexpr const char *kBoolName = "BOOL";
constexpr auto kAndOp = [](uint8_t a, uint8_t b) -> uint8_t { return (a != 0 && b != 0) ? 1 : 0; };
} // namespace

Tensor And::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kAndName, kBoolName, DataType::BOOL, x, y,
                                                          kAndOp);
}

void And::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kAndName, kBoolName, DataType::BOOL, x, y, output,
                                              kAndOp);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
