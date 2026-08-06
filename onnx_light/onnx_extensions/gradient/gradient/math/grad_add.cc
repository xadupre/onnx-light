// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/math/include_math_grads.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradAdd(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].empty())
    AccumulateGrad(output_grad, grad_accum[inputs[0]], counter, func);
  if (inputs.size() >= 2 && !inputs[1].empty())
    AccumulateGrad(output_grad, grad_accum[inputs[1]], counter, func);
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
