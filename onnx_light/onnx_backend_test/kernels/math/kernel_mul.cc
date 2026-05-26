// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kMulName = "kernel::Mul";
constexpr const char *kFloatName = "FLOAT";
constexpr auto kMulOp = [](float a, float b) { return a * b; };
} // namespace

Tensor Mul::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<float, float>(kMulName, kFloatName,
                                                      TensorProto::DataType::FLOAT, x, y, kMulOp);
}

void Mul::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<float, float>(kMulName, kFloatName, TensorProto::DataType::FLOAT, x, y,
                                          output, kMulOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
