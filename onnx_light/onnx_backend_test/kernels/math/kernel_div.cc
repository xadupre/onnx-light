// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kDivName = "kernel::Div";
constexpr const char *kFloatName = "FLOAT";
constexpr auto kDivOp = [](float a, float b) { return a / b; };
} // namespace

Tensor Div::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<float, float>(kDivName, kFloatName,
                                                      TensorProto::DataType::FLOAT, x, y, kDivOp);
}

void Div::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<float, float>(kDivName, kFloatName, TensorProto::DataType::FLOAT, x, y,
                                          output, kDivOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
