// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/gradient/grad_dispatcher.h"
#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::core::gradient {

const GradRegistry &DefaultGradRegistry() {
  static const GradRegistry kRegistry = [] {
    GradRegistry r;
    return r;
  }();
  return kRegistry;
}

void RegisterGradientFunction(const std::string &domain, const std::string &op_type, GradFn fn,
                              GradRegistry &registry) {
  registry[{domain, op_type}] = std::move(fn);
}

void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func, const GradRegistry &registry) {
  // Find the output gradient: take the first output that has a gradient.
  std::string output_grad;
  for (const auto &out : node.output()) {
    auto it = grad_table.find(out);
    if (it != grad_table.end()) {
      output_grad = it->second;
      break;
    }
  }
  if (output_grad.empty())
    return; // no gradient flows through this node

  const std::string domain = node.domain();
  const std::string op_type = node.op_type();
  auto it = registry.find({domain, op_type});
  EXT_ENFORCE(it != registry.end(), "onnx_gradient: no gradient function registered for domain='",
              domain, "' op_type='", op_type, "'");
  it->second(node, output_grad, grad_accum, counter, func);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::gradient
