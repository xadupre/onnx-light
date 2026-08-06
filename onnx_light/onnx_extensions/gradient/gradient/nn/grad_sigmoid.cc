// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradSigmoid(const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 1 || inputs[0].empty())
    return true;
  const std::string &A = inputs[0];
  if (node.output().empty() || node.output()[0].empty())
    return true;
  const std::string C = node.output()[0];

  // dA = dC * (C - C*C)
  std::string C2 = NewGradName("C2", counter);
  func.add_node("Mul", {C, C}, {C2});
  std::string C_minus_C2 = NewGradName("C_minus_C2", counter);
  func.add_node("Sub", {C, C2}, {C_minus_C2});
  std::string dA = NewGradName("dA", counter);
  func.add_node("Mul", {output_grad, C_minus_C2}, {dA});
  AccumulateGrad(dA, grad_accum[A], counter, func);
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
