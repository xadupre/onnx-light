// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/nn/include_nn_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradRelu(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].empty()) {
    const std::string &A = inputs[0];
    std::string sgn = NewGradName("relu_sign", counter);
    func.add_node("Sign", {A}, {sgn});
    std::string mask = NewGradName("relu_mask", counter);
    func.add_node("Relu", {sgn}, {mask});
    std::string dA = NewGradName("dA", counter);
    func.add_node("Mul", {output_grad, mask}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);
  }
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
