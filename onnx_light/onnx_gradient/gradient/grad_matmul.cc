// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_matmul.h"
#include "onnx_gradient/gradient/grad_common.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

bool GradMatMul(const NodeProto &node, const std::string &output_grad,
                std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                FunctionProto &func) {
  const auto &inputs = node.input();
  std::string A = (inputs.size() >= 1 && !inputs[0].null()) ? inputs[0].as_string() : "";
  std::string B = (inputs.size() >= 2 && !inputs[1].null()) ? inputs[1].as_string() : "";
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

  return true;
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
