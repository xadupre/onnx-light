// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_extensions/gradient/gradient/tensor/include_tensor_grads.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

bool GradTranspose(const NodeProto &node, const std::string &output_grad,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func) {
  const auto &inputs = node.input();
  if (inputs.size() < 1 || inputs[0].empty())
    return true;
  const std::string &A = inputs[0];
  std::string dA = NewGradName("dA", counter);
  const AttributeProto *perm_attr = FindAttribute(node, "perm");
  if (perm_attr != nullptr) {
    const auto &perm = perm_attr->ref_ints();
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t pi = 0; pi < perm.size(); ++pi) {
      inv_perm[static_cast<size_t>(perm[pi])] = static_cast<int64_t>(pi);
    }
    NodeProto &tp = func.add_node("Transpose", {output_grad}, {dA});
    AddAttribute(tp, "perm", inv_perm);
  } else {
    // No perm: reverse all dimensions (default ONNX behaviour).
    func.add_node("Transpose", {output_grad}, {dA});
  }
  AccumulateGrad(dA, grad_accum[A], counter, func);
  return true;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
