// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx_optim/shapes/controlflow/shape_controlflow.h"
#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/nn/shape_nn.h"
#include "onnx_optim/shapes/optional/shape_optional.h"
#include "onnx_optim/shapes/reduction/shape_reduction.h"
#include "onnx_optim/shapes/sequence/shape_sequence.h"
#include "onnx_optim/shapes/text/shape_text.h"
#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Checks the node belongs to a supported domain: the default ONNX
// domain (empty string or "ai.onnx") or the traditional ML domain
// ("ai.onnx.ml"). Throws std::invalid_argument otherwise.
// Domain-specific dispatch can be added here when other domains gain
// support.
void CheckOnnxDomain(const NodeProto &node) {
  EXT_ENFORCE_INVALID(node.domain().empty() || node.domain() == kOnnxDomain ||
                          node.domain() == traditionalml::kOnnxMlDomain,
                      "ComputeShapeNode: unsupported domain '" + node.domain().as_string() +
                          "' for op '" + node.op_type().as_string() + "'.");
}

// Normalises the empty default ONNX domain to ``kOnnxDomain`` so that
// dispatch-table lookups always use a canonical key.
std::string NormaliseDispatchDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  return domain.empty() ? std::string(kOnnxDomain) : domain;
}

// Verifies the node declares at least `expected` inputs.
void RequireInputs(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(node.input_size() >= expected,
                      "ComputeShapeNode: op '" + node.op_type().as_string() +
                          "' expects at least " + std::to_string(expected) + " input(s), got " +
                          std::to_string(node.input_size()) + ".");
}

// Signature of every per-operator ComputeShape* trampoline registered
// in kDispatchTable: it reads the node's inputs from ``ctx`` and
// inserts the resulting output descriptors back into ``ctx``.
using ComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;

// Returns the (normalised_domain, op_type) -> ComputeShape* dispatch
// table. Constructed on first use and shared across calls. Adding a
// new operator only requires inserting one new entry here. The
// dispatch key is the pair ``(domain, op_type)`` encoded as
// ``"<domain>:<op_type>"``; the empty default ONNX domain must be
// normalised to ``kOnnxDomain`` before lookup.
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  static const std::unordered_map<std::string, ComputeShapeFn> table = {
      {"ai.onnx:Abs",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAbs(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acos",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcos(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acosh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcosh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Add",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeAdd(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"ai.onnx:And",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeAnd(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"ai.onnx:AveragePool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeAveragePool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Constant",
       [](ShapesContext &ctx, const NodeProto &node) {
         generator::ComputeShapeConstant(ctx, node);
       }},
      {"ai.onnx:If",
       [](ShapesContext &ctx, const NodeProto &node) { controlflow::ComputeShapeIf(ctx, node); }},
      {"ai.onnx:Optional",
       [](ShapesContext &ctx, const NodeProto &node) {
         optional::ComputeShapeOptional(ctx, node);
       }},
      {"ai.onnx:ReduceSum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceSum(ctx, node, data_name.c_str(),
                                          node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:SequenceConstruct",
       [](ShapesContext &ctx, const NodeProto &node) {
         sequence::ComputeShapeSequenceConstruct(ctx, node);
       }},
      {"ai.onnx:StringConcat",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         text::ComputeShapeStringConcat(ctx, node, node.input(0).as_string().c_str(),
                                        node.input(1).as_string().c_str());
       }},
      {"ai.onnx.ml:LabelEncoder",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeLabelEncoder(ctx, node, node.input(0).as_string().c_str());
       }},
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
    EXT_ENFORCE_INVALID(ctx.Has(name), "CheckInputsAvailable: input '" + name + "' of op '" +
                                           node.op_type().as_string() +
                                           "' is missing from ShapesContext.");
  }
}

void CheckOutputsNotAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(!ctx.Has(name) && !ctx.HasSequence(name),
                        "CheckOutputsNotAvailable: output '" + name + "' of op '" +
                            node.op_type().as_string() + "' is already present in ShapesContext.");
  }
}

void ComputeShapeNode(ShapesContext &ctx, const NodeProto &node) {
  CheckOnnxDomain(node);
  CheckInputsAvailable(ctx, node);
  CheckOutputsNotAvailable(ctx, node);
  const std::string op_type = node.op_type().as_string();
  const std::string key = NormaliseDispatchDomain(node) + ":" + op_type;
  const auto &table = DispatchTable();
  auto it = table.find(key);
  EXT_ENFORCE_INVALID(it != table.end(), "ComputeShapeNode: unsupported op_type '" + op_type +
                                             "' in domain '" + NormaliseDispatchDomain(node) +
                                             "'.");
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
