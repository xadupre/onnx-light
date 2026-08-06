// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/reduction/include_reduction_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradReduceMean(const NodeProto &node, const std::string &output_grad,
                    std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                    FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].empty()) {
    const std::string &A = inputs[0];
    std::string shape_A = NewGradName("shape_A", counter);
    func.add_node("Shape", {A}, {shape_A});
    std::string expanded = NewGradName("expanded", counter);
    func.add_node("Expand", {output_grad, shape_A}, {expanded});
    std::string size_X = NewGradName("size_X", counter);
    func.add_node("Size", {A}, {size_X});
    std::string size_X_cast = NewGradName("size_X_cast", counter);
    func.add_node("CastLike", {size_X, expanded}, {size_X_cast});
    std::string dA = NewGradName("dA", counter);
    func.add_node("Div", {expanded, size_X_cast}, {dA});
    AccumulateGrad(dA, grad_accum[A], counter, func);
  }
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
