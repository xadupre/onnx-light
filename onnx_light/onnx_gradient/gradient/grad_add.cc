// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_add.h"
#include "onnx_gradient/gradient/grad_common.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

bool GradAdd(const NodeProto &node, const std::string &output_grad,
             std::unordered_map<std::string, std::string> &grad_accum, int &counter,
             FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].null() && !inputs[0].empty())
    AccumulateGrad(output_grad, grad_accum[inputs[0].as_string()], counter, func);
  if (inputs.size() >= 2 && !inputs[1].null() && !inputs[1].empty())
    AccumulateGrad(output_grad, grad_accum[inputs[1].as_string()], counter, func);
  return true;
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
