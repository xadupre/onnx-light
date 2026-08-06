// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/math/include_math_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradMul(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func) {
  const auto &inputs = node.input();
  const std::string &A =
      (inputs.size() >= 1 && !inputs[0].empty()) ? inputs[0] : utils::String::empty_string();
  const std::string &B =
      (inputs.size() >= 2 && !inputs[1].empty()) ? inputs[1] : utils::String::empty_string();
  if (!A.empty()) {
    std::string dA = NewGradName("dA", counter);
    func.add_node("Mul", {output_grad, B}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);
  }
  if (!B.empty()) {
    std::string dB = NewGradName("dB", counter);
    func.add_node("Mul", {output_grad, A}, {dB});
    AccumulateGrad(dB, grad_accum[B], counter, func);
  }
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
