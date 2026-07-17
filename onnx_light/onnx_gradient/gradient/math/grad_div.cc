// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_common.h"
#include "onnx_gradient/gradient/math/include_math_grads.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

bool GradDiv(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func) {
  const auto &inputs = node.input();
  std::string A = (inputs.size() >= 1 && !inputs[0].empty()) ? std::string(inputs[0]) : "";
  std::string B = (inputs.size() >= 2 && !inputs[1].empty()) ? std::string(inputs[1]) : "";
  if (!A.empty()) {
    std::string dA = NewGradName("dA", counter);
    func.add_node("Div", {output_grad, B}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);
  }
  if (!B.empty()) {
    std::string B2 = NewGradName("B2", counter);
    func.add_node("Mul", {B, B}, {B2});
    std::string A_div_B2 = NewGradName("A_div_B2", counter);
    func.add_node("Div", {A, B2}, {A_div_B2});
    std::string neg_grad = NewGradName("neg_grad", counter);
    func.add_node("Neg", {output_grad}, {neg_grad});
    std::string dB = NewGradName("dB", counter);
    func.add_node("Mul", {neg_grad, A_div_B2}, {dB});
    AccumulateGrad(dB, grad_accum[B], counter, func);
  }
  return true;
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
