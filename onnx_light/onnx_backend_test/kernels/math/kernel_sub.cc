// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kSubName = "kernel::Sub";
constexpr const char *kFloatName = "FLOAT";
constexpr auto kSubOp = [](float a, float b) { return a - b; };
} // namespace

Tensor Sub::operator()(const Tensor &x, const Tensor &y) const {
  return detail::BinaryElementwiseAlloc<float, float>(kSubName, kFloatName,
                                                      TensorProto::DataType::FLOAT, x, y, kSubOp);
}

void Sub::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  detail::BinaryElementwise<float, float>(kSubName, kFloatName, TensorProto::DataType::FLOAT, x, y,
                                          output, kSubOp);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
