// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/expand/where_pattern.h"

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

bool ReadIntegerTensor(const TensorProto &tensor, std::vector<int64_t> &values) {
  const auto dtype = static_cast<TensorProto::DataType>(tensor.data_type());
  if (dtype != TensorProto::DataType::INT32 && dtype != TensorProto::DataType::INT64) {
    return false;
  }
  return ReadIntegerValues(tensor, values);
}

bool ReadUnsqueezeAxes(core::builder::GraphGraph &graph, const NodeProto &node,
                       std::vector<int64_t> &axes) {
  if (node.input_size() != 2) {
    return false;
  }
  const std::string &axes_name = node.input()[1].value();
  if (!graph.IsConstant(axes_name)) {
    return false;
  }
  const TensorProto *axes_tensor = graph.GetComputedConstant(axes_name);
  if (axes_tensor == nullptr) {
    return false;
  }
  return ReadIntegerTensor(*axes_tensor, axes);
}

bool ExtractWhereAddTerms(const NodeProto &then_add, const NodeProto &else_add,
                          std::string &then_term, std::string &else_term,
                          std::string &common_term) {
  if (!IsDefaultOp(then_add, "Add") || !IsDefaultOp(else_add, "Add") ||
      then_add.input_size() != 2 || else_add.input_size() != 2) {
    return false;
  }

  const std::string t0 = then_add.input()[0].value();
  const std::string t1 = then_add.input()[1].value();
  const std::string e0 = else_add.input()[0].value();
  const std::string e1 = else_add.input()[1].value();

  if (t0 == e0) {
    common_term = t0;
    then_term = t1;
    else_term = e1;
    return true;
  }
  if (t0 == e1) {
    common_term = t0;
    then_term = t1;
    else_term = e0;
    return true;
  }
  if (t1 == e0) {
    common_term = t1;
    then_term = t0;
    else_term = e1;
    return true;
  }
  if (t1 == e1) {
    common_term = t1;
    then_term = t0;
    else_term = e0;
    return true;
  }
  return false;
}

} // namespace

std::set<std::string> NotWherePattern::FastOpType() const { return {"Where"}; }

core::builder::MatchResult NotWherePattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Where") || candidate.input_size() != 3) {
    return NoMatch(candidate, "candidate is not a default-domain Where with three inputs");
  }
  const NodeProto *not_node = graph.NodeBefore(candidate.input()[0].value());
  if (not_node == nullptr || !IsDefaultOp(*not_node, "Not") || not_node->input_size() != 1 ||
      not_node->output_size() != 1) {
    return NoMatch(candidate, "the condition is not produced by a default-domain Not");
  }
  return core::builder::MatchResult{this, {not_node, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
NotWherePattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("NotWherePattern::Apply expects one Not and one Where node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("NotWherePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &not_node = *nodes[0];
  const NodeProto &where = *nodes[1];

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(not_node.output()[0].value())) {
    replacements.add() = not_node;
  }

  const std::string name = "NotWherePattern--" + where.name().value();
  replacements.push_back(MakeNode(
      "Where", {not_node.input()[0].value(), where.input()[2].value(), where.input()[1].value()},
      {where.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> UnsqueezeEqualPattern::FastOpType() const { return {"Equal"}; }

core::builder::MatchResult UnsqueezeEqualPattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Equal") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Equal with two inputs");
  }

  const NodeProto *left_unsqueeze = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right_unsqueeze = graph.NodeBefore(candidate.input()[1].value());
  if (left_unsqueeze == nullptr || right_unsqueeze == nullptr ||
      !IsDefaultOp(*left_unsqueeze, "Unsqueeze") || !IsDefaultOp(*right_unsqueeze, "Unsqueeze") ||
      left_unsqueeze->output_size() != 1 || right_unsqueeze->output_size() != 1) {
    return NoMatch(candidate, "Equal inputs are not produced by default-domain Unsqueeze nodes");
  }

  std::vector<int64_t> left_axes;
  std::vector<int64_t> right_axes;
  if (!ReadUnsqueezeAxes(graph, *left_unsqueeze, left_axes) ||
      !ReadUnsqueezeAxes(graph, *right_unsqueeze, right_axes)) {
    return NoMatch(candidate, "Unsqueeze axes are not constant integer tensors");
  }
  if (left_axes != right_axes) {
    return NoMatch(candidate, "the Unsqueeze axes differ");
  }
  if (graph.IsUsedMoreThanOnce(left_unsqueeze->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(right_unsqueeze->output()[0].value())) {
    return NoMatch(candidate, "an Unsqueeze output is shared");
  }

  return core::builder::MatchResult{
      this, {left_unsqueeze, right_unsqueeze, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
UnsqueezeEqualPattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "UnsqueezeEqualPattern::Apply expects two Unsqueeze nodes and one Equal node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("UnsqueezeEqualPattern::Apply received an unsafe or inconsistent match.");
  }

  const NodeProto &left_unsqueeze = *nodes[0];
  const NodeProto &right_unsqueeze = *nodes[1];
  const NodeProto &equal = *nodes[2];
  const std::string name = "UnsqueezeEqualPattern--" + equal.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakeNode("Equal", {left_unsqueeze.input()[0].value(), right_unsqueeze.input()[0].value()},
               {equal.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> WhereAddPattern::FastOpType() const { return {"Where"}; }

core::builder::MatchResult WhereAddPattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Where") || candidate.input_size() != 3 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Where with three inputs");
  }
  const NodeProto *then_add = graph.NodeBefore(candidate.input()[1].value());
  const NodeProto *else_add = graph.NodeBefore(candidate.input()[2].value());
  if (then_add == nullptr || else_add == nullptr) {
    return NoMatch(candidate, "Where branches are not both produced by nodes");
  }

  std::string then_term;
  std::string else_term;
  std::string common_term;
  if (!ExtractWhereAddTerms(*then_add, *else_add, then_term, else_term, common_term)) {
    return NoMatch(candidate, "Where branches are not two Add nodes sharing one input");
  }
  if (graph.IsUsedMoreThanOnce(then_add->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(else_add->output()[0].value())) {
    return NoMatch(candidate, "a branch Add output is shared");
  }

  return core::builder::MatchResult{this, {then_add, else_add, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
WhereAddPattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("WhereAddPattern::Apply expects two Add nodes and one Where node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("WhereAddPattern::Apply received an unsafe or inconsistent match.");
  }

  const NodeProto &then_add = *nodes[0];
  const NodeProto &else_add = *nodes[1];
  const NodeProto &where = *nodes[2];

  std::string then_term;
  std::string else_term;
  std::string common_term;
  if (!ExtractWhereAddTerms(then_add, else_add, then_term, else_term, common_term)) {
    throw BuilderError("WhereAddPattern::Apply could not extract Add branch terms.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "WhereAddPattern--" + where.name().value();
  const std::string inner_where_output = builder.UniqueName(name + "_where");

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Where", {where.input()[0].value(), then_term, else_term},
                                  {inner_where_output}, "", (name + "--where").c_str()));
  replacements.push_back(MakeNode("Add", {inner_where_output, common_term},
                                  {where.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
