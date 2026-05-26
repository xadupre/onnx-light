// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx_optim/shapes/generator/shape_generator.h"
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
  if (!node.domain().empty() && node.domain() != kOnnxDomain) {
    throw std::invalid_argument("ComputeShapeNode: unsupported domain '" +
                                node.domain().as_string() + "' for op '" +
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

// Signature of every per-operator ComputeShape* trampoline registered
// in kDispatchTable: it reads the node's inputs from ``ctx`` and
// inserts the resulting output descriptors back into ``ctx``.
using ComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;

// Returns the op_type -> ComputeShape* dispatch table. Constructed on
// first use and shared across calls. Adding a new operator only
// requires inserting one new entry here.
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  static const std::unordered_map<std::string, ComputeShapeFn> table = {
      {"Abs",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAbs(ctx, node, node.input(0).as_string().c_str());
       }},
      {"Acos",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcos(ctx, node, node.input(0).as_string().c_str());
       }},
      {"Acosh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcosh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"Add",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeAdd(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"And",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeAnd(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"Constant", [](ShapesContext &ctx,
                      const NodeProto &node) { generator::ComputeShapeConstant(ctx, node); }},
  };
  return table;
}

} // namespace

void CheckInputsAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.input_size(); ++i) {
    const std::string name = node.input(i).as_string();
    if (name.empty()) {
      continue;
    }
    if (!ctx.Has(name)) {
      throw std::invalid_argument("CheckInputsAvailable: input '" + name + "' of op '" +
                                  node.op_type().as_string() + "' is missing from ShapesContext.");
    }
  }
}

void CheckOutputsNotAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    if (ctx.Has(name)) {
      throw std::invalid_argument("CheckOutputsNotAvailable: output '" + name + "' of op '" +
                                  node.op_type().as_string() +
                                  "' is already present in ShapesContext.");
    }
  }
}

void ComputeShapeNode(ShapesContext &ctx, const NodeProto &node) {
  CheckOnnxDomain(node);
  CheckInputsAvailable(ctx, node);
  CheckOutputsNotAvailable(ctx, node);
  const std::string op_type = node.op_type().as_string();
  const auto &table = DispatchTable();
  auto it = table.find(op_type);
  if (it == table.end()) {
    throw std::invalid_argument("ComputeShapeNode: unsupported op_type '" + op_type + "'.");
  }
  it->second(ctx, node);
}

void ComputeShapes(ShapesContext &ctx, const utils::RepeatedProtoField<NodeProto> &nodes) {
  for (int i = 0; i < nodes.size(); ++i) {
    ComputeShapeNode(ctx, nodes[i]);
  }
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
