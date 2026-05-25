// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/math/shape_math.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Checks the node belongs to the default ONNX domain (empty string or
// "ai.onnx"). Throws std::invalid_argument otherwise. Domain-specific
// dispatch can be added here when other domains gain support.
void CheckOnnxDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  if (!domain.empty() && domain != kOnnxDomain) {
    throw std::invalid_argument("ComputeShapeNode: unsupported domain '" + domain + "' for op '" +
                                node.op_type().as_string() + "'.");
  }
}

// Verifies the node declares at least `expected` inputs.
void RequireInputs(const NodeProto &node, int expected) {
  if (node.input_size() < expected) {
    throw std::invalid_argument("ComputeShapeNode: op '" + node.op_type().as_string() +
                                "' expects at least " + std::to_string(expected) +
                                " input(s), got " + std::to_string(node.input_size()) + ".");
  }
}

} // namespace

void ComputeShapeNode(ShapesContext &ctx, const NodeProto &node) {
  CheckOnnxDomain(node);
  const std::string op_type = node.op_type().as_string();

  if (op_type == "Abs") {
    RequireInputs(node, 1);
    math::ComputeShapeAbs(ctx, node, node.input(0).as_string());
    return;
  }
  if (op_type == "Add") {
    RequireInputs(node, 2);
    math::ComputeShapeAdd(ctx, node, node.input(0).as_string(), node.input(1).as_string());
    return;
  }
  if (op_type == "And") {
    RequireInputs(node, 2);
    logical::ComputeShapeAnd(ctx, node, node.input(0).as_string(), node.input(1).as_string());
    return;
  }

  throw std::invalid_argument("ComputeShapeNode: unsupported op_type '" + op_type + "'.");
}

void ComputeShapes(ShapesContext &ctx, const utils::RepeatedProtoField<NodeProto> &nodes) {
  for (int i = 0; i < nodes.size(); ++i) {
    ComputeShapeNode(ctx, nodes[i]);
  }
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
