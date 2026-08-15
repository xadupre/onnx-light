// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/cast_cast_binary_pattern.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern_helpers.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::symbolic::TensorType;

bool IsBinaryArithmetic(const NodeProto &node) {
  const std::string &op_type = node.op_type().value();
  return (op_type == "Add" || op_type == "Div" || op_type == "Mul" || op_type == "Sub") &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 2 &&
         node.output_size() == 1;
}

} // namespace

std::set<std::string> CastCastBinaryPattern::FastOpType() const {
  return {"Add", "Div", "Mul", "Sub"};
}

core::builder::MatchResult CastCastBinaryPattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsBinaryArithmetic(candidate)) {
    return NoMatch(candidate, "candidate is not a supported default-domain binary operation");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.input()[1].value())) {
    return NoMatch(candidate, "a Cast output has another use");
  }

  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr || !detail::IsDefaultCast(*left) ||
      !detail::IsDefaultCast(*right)) {
    return NoMatch(candidate, "both binary inputs must be produced by unary Cast nodes");
  }
  if (left == right) {
    return NoMatch(candidate, "the binary inputs must be produced by distinct Cast nodes");
  }
  if (!graph.HasType(left->input()[0].value()) || !graph.HasType(right->input()[0].value()) ||
      !graph.HasType(candidate.input()[0].value()) ||
      !graph.HasType(candidate.input()[1].value())) {
    return NoMatch(candidate, "a source or Cast output element type is unknown");
  }

  TensorType left_target;
  TensorType right_target;
  if (!detail::CastTarget(*left, left_target) || !detail::CastTarget(*right, right_target)) {
    return NoMatch(candidate, "a Cast 'to' attribute is missing or invalid");
  }
  const TensorType left_source = graph.GetType(left->input()[0].value());
  const TensorType right_source = graph.GetType(right->input()[0].value());
  if (!detail::IsSupportedFloat(left_source) || !detail::IsSupportedFloat(right_source) ||
      !detail::IsSupportedFloat(left_target) || !detail::IsSupportedFloat(right_target)) {
    return NoMatch(candidate, "the Cast source and target types must be floating-point");
  }
  if (left_source != right_source || left_target != right_target ||
      graph.GetType(candidate.input()[0].value()) != left_target ||
      graph.GetType(candidate.input()[1].value()) != right_target) {
    return NoMatch(candidate, "the two Cast operations must have matching source and target types");
  }
  if (detail::FloatWidth(left_target) > detail::FloatWidth(left_source)) {
    return NoMatch(candidate, "moving the binary operation would lower its precision");
  }
  return core::builder::MatchResult{this, {left, right, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastCastBinaryPattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw core::builder::BuilderError(
        "CastCastBinaryPattern::Apply expects two Cast nodes and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw core::builder::BuilderError(
        "CastCastBinaryPattern::Apply received an unsafe or inconsistent match.");
  }

  const NodeProto &left = *nodes[0];
  const NodeProto &right = *nodes[1];
  const NodeProto &binary = *nodes[2];
  TensorType target_type;
  detail::CastTarget(left, target_type);

  const std::string intermediate = graph.Builder().UniqueName("CastCastBinaryPattern");
  const std::string node_name = "CastCastBinaryPattern--" + binary.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode(binary.op_type().value().c_str(),
                                  {left.input()[0].value(), right.input()[0].value()},
                                  {intermediate}, "", node_name.c_str()));
  NodeProto cast =
      MakeNode("Cast", {intermediate}, {binary.output()[0].value()}, "", node_name.c_str());
  AddAttribute<int64_t>(cast, "to",
                        static_cast<int64_t>(core::symbolic::TensorTypeToDataType(target_type)));
  if (binary.has_doc_string()) {
    cast.set_doc_string(binary.doc_string().value());
  }
  replacements.push_back(std::move(cast));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
