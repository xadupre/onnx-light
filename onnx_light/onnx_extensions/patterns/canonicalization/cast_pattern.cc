// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"

#include <algorithm>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;
using core::symbolic::TensorType;

bool IsSupportedFloat(TensorType type) {
  return type == TensorType::kFloat16 || type == TensorType::kBfloat16 ||
         type == TensorType::kFloat || type == TensorType::kDouble;
}

int FloatWidth(TensorType type) {
  switch (type) {
  case TensorType::kFloat16:
  case TensorType::kBfloat16:
    return 16;
  case TensorType::kFloat:
    return 32;
  case TensorType::kDouble:
    return 64;
  default:
    return 0;
  }
}

bool IsDefaultCast(const NodeProto &node) {
  return node.op_type().value() == "Cast" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 1 &&
         node.output_size() == 1;
}

bool CastTarget(const NodeProto &node, TensorType &type) {
  const AttributeProto *attribute = FindAttribute(node, "to");
  if (attribute == nullptr || attribute->type() != AttributeProto::AttributeType::INT) {
    return false;
  }
  type = core::symbolic::DataTypeToTensorType(static_cast<TensorProto::DataType>(attribute->i()));
  return type != TensorType::kUndefined;
}

bool IsFloatingRoundTripType(TensorType type) {
  return type == TensorType::kFloat16 || type == TensorType::kBfloat16 ||
         type == TensorType::kFloat;
}

bool IsBinaryArithmetic(const NodeProto &node) {
  const std::string &op_type = node.op_type().value();
  return (op_type == "Add" || op_type == "Div" || op_type == "Mul" || op_type == "Sub") &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 2 &&
         node.output_size() == 1;
}

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

std::set<std::string> CastPattern::FastOpType() const { return {"Cast"}; }

core::builder::MatchResult CastPattern::Match(core::builder::GraphGraph &graph,
                                              const NodeProto &candidate) const {
  if (!IsDefaultCast(candidate)) {
    return NoMatch(candidate, "candidate is not a unary default-domain Cast");
  }
  if (!graph.HasType(candidate.input()[0].value())) {
    return NoMatch(candidate, "input element type is unknown");
  }
  core::symbolic::TensorType target_type;
  if (!CastTarget(candidate, target_type)) {
    return NoMatch(candidate, "the Cast 'to' attribute is missing or invalid");
  }
  if (target_type != graph.GetType(candidate.input()[0].value())) {
    return NoMatch(candidate, "the input and target element types differ");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastPattern::Apply(core::builder::GraphGraph &graph,
                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr || !IsDefaultCast(*nodes[0])) {
    throw core::builder::BuilderError("CastPattern::Apply expects one default-domain Cast node.");
  }
  const NodeProto &cast = *nodes[0];
  core::symbolic::TensorType target_type;
  if (!graph.HasType(cast.input()[0].value()) || !CastTarget(cast, target_type) ||
      target_type != graph.GetType(cast.input()[0].value())) {
    throw core::builder::BuilderError("CastPattern::Apply received a non-redundant Cast.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  const std::string name = "CastPattern--" + cast.name().value();
  NodeProto replacement =
      MakeNode("Identity", {cast.input()[0].value()}, {cast.output()[0].value()}, "", name.c_str());
  if (cast.has_doc_string()) {
    replacement.set_doc_string(cast.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> CastCastPattern::FastOpType() const { return {"Cast"}; }

TensorType CastCastPattern::OneCastType(TensorType input_type, TensorType middle_type,
                                        TensorType final_type) {
  if (input_type == final_type) {
    if (middle_type == input_type) {
      return input_type;
    }
    if (IsFloatingRoundTripType(input_type) &&
        (middle_type == TensorType::kFloat || middle_type == TensorType::kDouble)) {
      return input_type;
    }
  } else if (final_type == middle_type) {
    return middle_type;
  } else if (input_type == middle_type) {
    return final_type;
  }
  return TensorType::kUndefined;
}

core::builder::MatchResult CastCastPattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!IsDefaultCast(candidate)) {
    return NoMatch(candidate, "candidate is not a unary default-domain Cast");
  }
  const NodeProto *inner = graph.NodeBefore(candidate.input()[0].value());
  if (inner == nullptr) {
    return NoMatch(candidate, "the Cast input is not produced by another node");
  }
  if (!IsDefaultCast(*inner)) {
    return NoMatch(candidate, "the preceding node is not a unary default-domain Cast");
  }
  if (!graph.HasType(inner->input()[0].value())) {
    return NoMatch(candidate, "the first Cast input element type is unknown");
  }

  TensorType middle_type;
  TensorType final_type;
  if (!CastTarget(*inner, middle_type)) {
    return NoMatch(candidate, "the first Cast 'to' attribute is missing or invalid");
  }
  if (!CastTarget(candidate, final_type)) {
    return NoMatch(candidate, "the second Cast 'to' attribute is missing or invalid");
  }
  if (OneCastType(graph.GetType(inner->input()[0].value()), middle_type, final_type) ==
      TensorType::kUndefined) {
    return NoMatch(candidate, "combining the Cast operations would change the result");
  }
  return core::builder::MatchResult{
      this,
      {inner, &candidate},
      graph.IsUsedMoreThanOnce(inner->output()[0].value()) ? nullptr : &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastCastPattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      !IsDefaultCast(*nodes[0]) || !IsDefaultCast(*nodes[1])) {
    throw BuilderError("CastCastPattern::Apply expects two consecutive Cast nodes.");
  }
  const NodeProto &inner = *nodes[0];
  const NodeProto &outer = *nodes[1];
  TensorType middle_type;
  TensorType final_type;
  if (!graph.HasType(inner.input()[0].value()) || !CastTarget(inner, middle_type) ||
      !CastTarget(outer, final_type)) {
    throw BuilderError("CastCastPattern::Apply received an invalid Cast match.");
  }
  const TensorType input_type = graph.GetType(inner.input()[0].value());
  const TensorType replacement_type = OneCastType(input_type, middle_type, final_type);
  if (replacement_type == TensorType::kUndefined) {
    throw BuilderError("CastCastPattern::Apply received an unsafe Cast match.");
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(inner.output()[0].value())) {
    replacements.add() = inner;
  }

  NodeProto replacement =
      MakeNode(replacement_type == input_type ? "Identity" : "Cast", {inner.input()[0].value()},
               {outer.output()[0].value()}, "", "CastCastPattern");
  if (replacement_type != input_type) {
    AddAttribute<int64_t>(
        replacement, "to",
        static_cast<int64_t>(core::symbolic::TensorTypeToDataType(replacement_type)));
  }
  replacements.add() = std::move(replacement);
  return replacements;
}

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
  if (left == nullptr || right == nullptr || !IsDefaultCast(*left) || !IsDefaultCast(*right)) {
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
  if (!CastTarget(*left, left_target) || !CastTarget(*right, right_target)) {
    return NoMatch(candidate, "a Cast 'to' attribute is missing or invalid");
  }
  const TensorType left_source = graph.GetType(left->input()[0].value());
  const TensorType right_source = graph.GetType(right->input()[0].value());
  if (!IsSupportedFloat(left_source) || !IsSupportedFloat(right_source) ||
      !IsSupportedFloat(left_target) || !IsSupportedFloat(right_target)) {
    return NoMatch(candidate, "the Cast source and target types must be floating-point");
  }
  if (left_source != right_source || left_target != right_target ||
      graph.GetType(candidate.input()[0].value()) != left_target ||
      graph.GetType(candidate.input()[1].value()) != right_target) {
    return NoMatch(candidate, "the two Cast operations must have matching source and target types");
  }
  if (FloatWidth(left_target) > FloatWidth(left_source)) {
    return NoMatch(candidate, "moving the binary operation would lower its precision");
  }
  return core::builder::MatchResult{this, {left, right, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastCastBinaryPattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("CastCastBinaryPattern::Apply expects two Cast nodes and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("CastCastBinaryPattern::Apply received an unsafe or inconsistent match.");
  }

  const NodeProto &left = *nodes[0];
  const NodeProto &right = *nodes[1];
  const NodeProto &binary = *nodes[2];
  TensorType target_type;
  CastTarget(left, target_type);

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
  if (consumers.size() != 1 || !IsDefaultCast(*consumers[0])) {
    return NoMatch(candidate, "the operation output must feed one unary Cast node");
  }
  const NodeProto *output_cast = consumers[0];
  TensorType output_type;
  if (!CastTarget(*output_cast, output_type) || !graph.HasType(candidate.output()[0].value()) ||
      !graph.HasType(output_cast->output()[0].value())) {
    return NoMatch(candidate, "the computation or output Cast type is unknown");
  }
  const TensorType computation_type = graph.GetType(candidate.output()[0].value());
  if (!IsSupportedFloat(computation_type) || !IsSupportedFloat(output_type) ||
      graph.GetType(output_cast->output()[0].value()) != output_type) {
    return NoMatch(candidate, "the computation and output types must be floating-point");
  }

  std::vector<const NodeProto *> input_casts;
  for (const auto &input : candidate.input()) {
    const NodeProto *producer = graph.NodeBefore(input.value());
    if (producer == nullptr || !IsDefaultCast(*producer)) {
      continue;
    }
    if (!graph.HasType(producer->input()[0].value())) {
      return NoMatch(candidate, "an input Cast source type is unknown");
    }
    TensorType input_target;
    if (!CastTarget(*producer, input_target) || input_target != computation_type ||
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
    throw BuilderError(
        "CastOpCastPattern::Apply expects input Cast nodes, one operation, and one output Cast.");
  }
  const NodeProto &operation = *nodes[nodes.size() - 2];
  const core::builder::MatchResult verified = Match(graph, operation);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("CastOpCastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &output_cast = *nodes.back();
  TensorType output_type;
  CastTarget(output_cast, output_type);

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
