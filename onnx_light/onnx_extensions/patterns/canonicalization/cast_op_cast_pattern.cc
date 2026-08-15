// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/cast_op_cast_pattern.h"

#include <algorithm>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern_helpers.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::symbolic::TensorType;

bool IsSupportedOperation(const NodeProto &node) {
  const std::string &op_type = node.op_type().value();
  const bool binary = op_type == "Add" || op_type == "Div" || op_type == "Mul" || op_type == "Sub";
  const bool unary =
      op_type == "MulSigmoid" || op_type == "Neg" || op_type == "Sigmoid" || op_type == "Softmax";
  return ((binary && NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain &&
           node.input_size() == 2) ||
          (unary && node.input_size() == 1)) &&
         node.output_size() == 1;
}

bool ContainsNode(const std::vector<const NodeProto *> &nodes, const NodeProto *node) {
  return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
}

} // namespace

std::set<std::string> CastOpCastPattern::FastOpType() const {
  return {"Add", "Div", "Mul", "MulSigmoid", "Neg", "Sigmoid", "Softmax", "Sub"};
}

core::builder::MatchResult CastOpCastPattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsSupportedOperation(candidate)) {
    return NoMatch(candidate, "candidate is not a supported unary or binary operation");
  }
  if (candidate.name().value().find("CastOpCastPattern--") != std::string::npos) {
    return NoMatch(candidate, "candidate was already produced by this pattern");
  }
  for (const auto &input : candidate.input()) {
    if (graph.IsUsedMoreThanOnce(input.value())) {
      return NoMatch(candidate, "an operation input has another use");
    }
  }

  const std::vector<const NodeProto *> &consumers = graph.NextNodes(candidate.output()[0].value());
  if (consumers.size() != 1 || !detail::IsDefaultCast(*consumers[0])) {
    return NoMatch(candidate, "the operation output must feed one unary Cast node");
  }
  const NodeProto *output_cast = consumers[0];
  TensorType output_type;
  if (!detail::CastTarget(*output_cast, output_type) ||
      !graph.HasType(candidate.output()[0].value()) ||
      !graph.HasType(output_cast->output()[0].value())) {
    return NoMatch(candidate, "the computation or output Cast type is unknown");
  }
  const TensorType computation_type = graph.GetType(candidate.output()[0].value());
  if (!detail::IsSupportedFloat(computation_type) || !detail::IsSupportedFloat(output_type) ||
      graph.GetType(output_cast->output()[0].value()) != output_type) {
    return NoMatch(candidate, "the computation and output types must be floating-point");
  }

  std::vector<const NodeProto *> input_casts;
  for (const auto &input : candidate.input()) {
    const NodeProto *producer = graph.NodeBefore(input.value());
    if (producer == nullptr || !detail::IsDefaultCast(*producer)) {
      continue;
    }
    if (!graph.HasType(producer->input()[0].value())) {
      return NoMatch(candidate, "an input Cast source type is unknown");
    }
    TensorType input_target;
    if (!detail::CastTarget(*producer, input_target) || input_target != computation_type ||
        graph.GetType(producer->input()[0].value()) != output_type) {
      return NoMatch(candidate,
                     "input Cast source and target types do not match the output and computation");
    }
    if (!ContainsNode(input_casts, producer)) {
      input_casts.push_back(producer);
    }
  }
  if (input_casts.empty()) {
    return NoMatch(candidate, "at least one operation input must be produced by a Cast");
  }

  input_casts.push_back(&candidate);
  input_casts.push_back(output_cast);
  return core::builder::MatchResult{this, std::move(input_casts), &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastOpCastPattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() < 3 || nodes[nodes.size() - 2] == nullptr || nodes.back() == nullptr) {
    throw core::builder::BuilderError(
        "CastOpCastPattern::Apply expects input Cast nodes, one operation, and one output Cast.");
  }
  const NodeProto &operation = *nodes[nodes.size() - 2];
  const core::builder::MatchResult verified = Match(graph, operation);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw core::builder::BuilderError(
        "CastOpCastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &output_cast = *nodes.back();
  TensorType output_type;
  detail::CastTarget(output_cast, output_type);

  const std::string base_name = "CastOpCastPattern--" + operation.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  std::vector<std::string> inputs;
  inputs.reserve(operation.input_size());
  for (const auto &input : operation.input()) {
    const NodeProto *producer = graph.NodeBefore(input.value());
    if (producer != nullptr && ContainsNode(nodes, producer)) {
      inputs.push_back(producer->input()[0].value());
      continue;
    }
    const std::string cast_output = graph.Builder().UniqueName("CastOpCastPattern");
    NodeProto cast =
        MakeNode("Cast", {input.value()}, {cast_output}, "", (base_name + "--Cast").c_str());
    AddAttribute<int64_t>(cast, "to",
                          static_cast<int64_t>(core::symbolic::TensorTypeToDataType(output_type)));
    replacements.push_back(std::move(cast));
    inputs.push_back(cast_output);
  }

  NodeProto replacement =
      MakeNode(operation.op_type().value().c_str(), inputs, {output_cast.output()[0].value()},
               operation.domain().value().c_str(), base_name.c_str());
  for (const AttributeProto &attribute : operation.attribute()) {
    replacement.mutable_attribute()->push_back(attribute);
  }
  if (operation.has_doc_string()) {
    replacement.set_doc_string(operation.doc_string().value());
  }
  replacements.push_back(std::move(replacement));

  if (graph.IsUsedMoreThanOnce(operation.output()[0].value())) {
    NodeProto restore = MakeNode("Cast", {output_cast.output()[0].value()},
                                 {operation.output()[0].value()}, "", base_name.c_str());
    AddAttribute<int64_t>(restore, "to",
                          static_cast<int64_t>(core::symbolic::TensorTypeToDataType(
                              graph.GetType(operation.output()[0].value()))));
    replacements.push_back(std::move(restore));
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
