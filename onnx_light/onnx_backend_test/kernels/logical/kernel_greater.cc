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
constexpr const char *kGreaterName = "kernel::Greater";
constexpr const char *kFloatName = "FLOAT";
constexpr const char *kBoolName = "BOOL";
constexpr auto kGreaterOp = [](float a, float b) -> uint8_t { return a > b ? 1 : 0; };
} // namespace

Tensor Greater::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAllocInOut<float, uint8_t>(
      kGreaterName, kFloatName, TensorProto::DataType::FLOAT, kBoolName,
      TensorProto::DataType::BOOL, x, y, kGreaterOp);
}

void Greater::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwiseInOut<float, uint8_t>(
      kGreaterName, kFloatName, TensorProto::DataType::FLOAT, kBoolName,
      TensorProto::DataType::BOOL, x, y, output, kGreaterOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
