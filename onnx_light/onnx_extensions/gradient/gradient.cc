// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/gradient/gradient.h"
#include "onnx_core/gradient/gradient.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

FunctionProto GradientOfNodes(std::span<const NodeProto> nodes, std::span<const std::string> inputs,
                              std::span<const TensorProto> initializers,
                              std::span<const std::string> xs, const std::string &y,
                              std::span<const std::string> zs, const GradRegistry &registry) {
  return core::gradient::GradientOfNodes(nodes, inputs, initializers, xs, y, zs, registry);
}

FunctionProto GradientOfFunction(const FunctionProto &function, std::span<const std::string> xs,
                                 const std::string &y, std::span<const std::string> zs,
                                 const GradRegistry &registry) {
  return core::gradient::GradientOfFunction(function, xs, y, zs, registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
