// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/math/include_math_grads.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradGemm(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func) {
  const auto &inputs = node.input();
  const std::string &A =
      (inputs.size() >= 1 && !inputs[0].empty()) ? inputs[0] : utils::String::empty_string();
  const std::string &B =
      (inputs.size() >= 2 && !inputs[1].empty()) ? inputs[1] : utils::String::empty_string();
  if (A.empty() || B.empty())
    return false;

  // dA = dC @ B^T
  std::string B_T = NewGradName("B_T", counter);
  func.add_node("Transpose", {B}, {B_T});
  std::string dA = NewGradName("dA", counter);
  func.add_node("MatMul", {output_grad, B_T}, {dA});
  AccumulateGrad(dA, grad_accum[A], counter, func);

  // dB = A^T @ dC
  std::string A_T = NewGradName("A_T", counter);
  func.add_node("Transpose", {A}, {A_T});
  std::string dB = NewGradName("dB", counter);
  func.add_node("MatMul", {A_T, output_grad}, {dB});
  AccumulateGrad(dB, grad_accum[B], counter, func);

  // Gradient for optional bias: sum over the batch axis.
  if (inputs.size() >= 3 && !inputs[2].empty()) {
    std::string bias = inputs[2];
    std::string dBias = NewGradName("dC_init", counter);
    NodeProto &rs = func.add_node("ReduceSum", {output_grad}, {dBias});
    AddAttribute(rs, "axes", std::vector<int64_t>{0}); // 0 = batch axis
    AddAttribute(rs, "keepdims", int64_t{0});
    AccumulateGrad(dBias, grad_accum[bias], counter, func);
  }

  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
