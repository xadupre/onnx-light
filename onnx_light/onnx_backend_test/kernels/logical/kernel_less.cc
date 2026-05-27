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
constexpr const char *kLessName = "kernel::Less";
constexpr const char *kFloatName = "FLOAT";
constexpr const char *kBoolName = "BOOL";
constexpr auto kLessOp = [](float a, float b) -> uint8_t { return a < b ? 1 : 0; };
} // namespace

Tensor Less::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAllocInOut<float, uint8_t>(
      kLessName, kFloatName, TensorProto::DataType::FLOAT, kBoolName, TensorProto::DataType::BOOL,
      x, y, kLessOp);
}

void Less::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwiseInOut<float, uint8_t>(
      kLessName, kFloatName, TensorProto::DataType::FLOAT, kBoolName, TensorProto::DataType::BOOL,
      x, y, output, kLessOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
