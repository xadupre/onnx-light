// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient/grad_common.h"
#include "onnx_gradient/gradient/tensor/include_tensor_grads.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

bool GradIdentity(const NodeProto &node, const std::string &output_grad,
                  std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                  FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() >= 1 && !inputs[0].null() && !inputs[0].empty())
    AccumulateGrad(output_grad, grad_accum[inputs[0].as_string()], counter, func);
  return true;
}

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
