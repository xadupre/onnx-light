// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kOrName = "kernel::Or";
constexpr const char *kBoolName = "BOOL";
constexpr auto kOrOp = [](uint8_t a, uint8_t b) -> uint8_t { return (a != 0 || b != 0) ? 1 : 0; };
} // namespace

Tensor Or::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y,
                                                          kOrOp);
}

void Or::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<uint8_t, uint8_t>(kOrName, kBoolName, DataType::BOOL, x, y, output,
                                              kOrOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
