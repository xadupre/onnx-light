// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_common.h"
#include "onnx_gradient/gradient/math/include_math_grads.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

bool GradNeg(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].empty() && !inputs[0].empty()) {
    std::string neg = NewGradName("neg_grad", counter);
    func.add_node("Neg", {output_grad}, {neg});
    AccumulateGrad(neg, grad_accum[inputs[0]], counter, func);
  }
  return true;
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
