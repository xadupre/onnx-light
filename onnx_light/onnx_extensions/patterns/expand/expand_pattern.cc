// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/expand/expand_pattern.h"

#include <set>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultOp(const NodeProto &node, const char *op_type) {
  return node.op_type().value() == op_type &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

// Element-wise binary and comparison operators that broadcast their inputs.
const std::set<std::string> &BinaryOpTypes() {
  static const std::set<std::string> types = {"Add",
                                              "And",
                                              "Div",
                                              "Mul",
                                              "Mod",
                                              "Or",
                                              "Sub",
                                              "Xor",
                                              "Equal",
                                              "Greater",
                                              "GreaterOrEqual",
                                              "Less",
                                              "LessOrEqual"};
  return types;
}

// Unary-like operators that preserve the shape of their first input.
const std::set<std::string> &UnaryLikeOpTypes() {
  static const std::set<std::string> types = {"Abs",
                                              "Acos",
                                              "Acosh",
                                              "Asin",
                                              "Asinh",
                                              "Atan",
                                              "Atanh",
                                              "BitShift",
                                              "Cast",
                                              "CastLike",
                                              "Ceil",
                                              "Celu",
                                              "Clip",
                                              "Cos",
                                              "Cosh",
                                              "DequantizeLinear",
                                              "DynamicQuantizeLinear",
                                              "Elu",
                                              "Erf",
                                              "Exp",
                                              "IsInf",
                                              "Log",
                                              "LogSoftmax",
                                              "Neg",
                                              "Not",
                                              "PRelu",
                                              "QuantizeLinear",
                                              "Reciprocal",
                                              "Relu",
                                              "Round",
                                              "Selu",
                                              "Sigmoid",
                                              "Sign",
                                              "Sin",
                                              "Sinh",
                                              "Softmax",
                                              "SoftmaxCrossEntropyLoss",
                                              "Softplus",
                                              "Softsign",
                                              "Sqrt",
                                              "Tan",
                                              "Tanh",
                                              "ThresholdRelu"};
  return types;
}

// Operators handled by ExpandSwapPattern regardless of their domain.
const std::set<std::string> &SwapOtherOpTypes() {
  static const std::set<std::string> types = {"NegXplus1", "ReplaceZero", "Pow"};
  return types;
}

// Reads the fully-static shape of ``name`` into ``dims``, returning ``false``
// when the shape is unknown or contains a symbolic dimension.
bool ReadStaticShape(core::builder::GraphGraph &graph, const std::string &name,
                     std::vector<int64_t> &dims) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  dims.clear();
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt()) {
      return false;
    }
    dims.push_back(shape[i].AsInt());
  }
  return true;
}

// Reads the constant integer tensor ``name`` into ``values``, returning
// ``false`` when it is not a constant INT32/INT64 tensor.
bool ReadConstantShape(core::builder::GraphGraph &graph, const std::string &name,
                       std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  const auto dtype = static_cast<TensorProto::DataType>(tensor->data_type());
  if (dtype != TensorProto::DataType::INT32 && dtype != TensorProto::DataType::INT64) {
    return false;
  }
  return ReadIntegerValues(*tensor, values);
}

} // namespace

std::set<std::string> ExpandPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandPattern::Match(core::builder::GraphGraph &graph,
                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  std::vector<int64_t> shape;
  if (!ReadStaticShape(graph, candidate.input()[0].value(), shape)) {
    return NoMatch(candidate, "the Expand input has no fully-static shape");
  }
  std::vector<int64_t> new_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), new_shape)) {
    return NoMatch(candidate, "the Expand target shape is not a constant integer tensor");
  }
  if (shape != new_shape) {
    return NoMatch(candidate, "the Expand changes the input shape");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandPattern::Apply(core::builder::GraphGraph &graph,
                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ExpandPattern::Apply expects one Expand node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const std::string name = "ExpandPattern--" + expand.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Identity", {expand.input()[0].value()},
                                  {expand.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> ExpandBroadcastPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandBroadcastPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  std::vector<int64_t> shape;
  if (!ReadStaticShape(graph, candidate.input()[0].value(), shape)) {
    return NoMatch(candidate, "the Expand input has no fully-static shape");
  }
  std::vector<int64_t> new_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), new_shape)) {
    return NoMatch(candidate, "the Expand target shape is not a constant integer tensor");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the Expand output is not consumed by exactly one node");
  }
  const NodeProto &next_node = *next_nodes[0];
  if (NormaliseDomain(next_node.domain().value()) != kDefaultOnnxDomain ||
      BinaryOpTypes().find(next_node.op_type().value()) == BinaryOpTypes().end() ||
      next_node.input_size() != 2) {
    return NoMatch(candidate, "the consumer is not a default-domain element-wise binary operator");
  }

  const std::string &other = next_node.input()[0].value() == candidate.output()[0].value()
                                 ? next_node.input()[1].value()
                                 : next_node.input()[0].value();
  std::vector<int64_t> other_shape;
  if (!ReadStaticShape(graph, other, other_shape)) {
    return NoMatch(candidate, "the other operand has no fully-static shape");
  }
  if (new_shape != other_shape) {
    return NoMatch(candidate, "Expand does not target the shape of the other operand");
  }
  if (shape.size() != other_shape.size()) {
    return NoMatch(candidate, "the operands have different ranks");
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != other_shape[i] && shape[i] != 1 && other_shape[i] != 1) {
      return NoMatch(candidate, "the operands do not broadcast together");
    }
  }

  return core::builder::MatchResult{this, {&candidate, &next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandBroadcastPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ExpandBroadcastPattern::Apply expects one Expand and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandBroadcastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &next_node = *nodes[1];

  std::vector<std::string> inputs;
  if (next_node.input()[0].value() == expand.output()[0].value()) {
    inputs = {expand.input()[0].value(), next_node.input()[1].value()};
  } else {
    inputs = {next_node.input()[0].value(), expand.input()[0].value()};
  }
  const std::string name = "ExpandBroadcastPattern--" + expand.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode(next_node.op_type().value().c_str(), inputs,
                                  {next_node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> ExpandSwapPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandSwapPattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  if (!graph.HasShape(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Expand input has no known shape");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the Expand output is not consumed by exactly one node");
  }
  const NodeProto &next_node = *next_nodes[0];
  const std::string &op_type = next_node.op_type().value();
  const bool is_other = SwapOtherOpTypes().find(op_type) != SwapOtherOpTypes().end();
  const bool is_unary = UnaryLikeOpTypes().find(op_type) != UnaryLikeOpTypes().end() &&
                        NormaliseDomain(next_node.domain().value()) == kDefaultOnnxDomain;
  if (!is_other && !is_unary) {
    return NoMatch(candidate, "the consumer is not a unary-like operator");
  }
  if (next_node.output_size() != 1) {
    return NoMatch(candidate, "the consumer does not have a single output");
  }

  return core::builder::MatchResult{this, {&candidate, &next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandSwapPattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ExpandSwapPattern::Apply expects one Expand and one unary-like node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandSwapPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &next_node = *nodes[1];

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ExpandSwapPattern--" + expand.name().value();
  const std::string new_name = builder.UniqueName("ExpandSwapPattern_" + expand.input()[0].value());

  std::vector<std::string> unary_inputs{expand.input()[0].value()};
  for (int i = 1; i < next_node.input_size(); ++i) {
    unary_inputs.push_back(next_node.input()[i].value());
  }

  NodeProto unary = MakeNode(next_node.op_type().value().c_str(), unary_inputs, {new_name},
                             next_node.domain().value().c_str(), name.c_str());
  for (const AttributeProto &attribute : next_node.attribute()) {
    unary.mutable_attribute()->push_back(attribute);
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(unary));
  replacements.push_back(MakeNode("Expand", {new_name, expand.input()[1].value()},
                                  {next_node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
