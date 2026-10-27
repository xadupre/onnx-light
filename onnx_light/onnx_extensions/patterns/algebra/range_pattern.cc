// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/range_pattern.h"

#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;

bool HasOneElementShape(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  return shape.Rank() == 1 && shape[0].IsInt() && shape[0].AsInt() == 1;
}

} // namespace

std::set<std::string> SwapRangeAddScalarPattern::FastOpType() const { return {"Range"}; }

core::builder::MatchResult SwapRangeAddScalarPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Range") || candidate.input_size() < 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Range");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || !IsDefaultOp(*next_nodes[0], "Add") ||
      next_nodes[0]->input_size() != 2 || next_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "the Range is not followed by one default-domain Add");
  }
  if (!HasOneElementShape(graph, next_nodes[0]->input()[1].value())) {
    return NoMatch(candidate, "the Add second input does not have shape [1]");
  }
  return core::builder::MatchResult{this, {&candidate, next_nodes[0]}, next_nodes[0]};
}

utils::RepeatedProtoField<NodeProto>
SwapRangeAddScalarPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SwapRangeAddScalarPattern::Apply expects a Range and an Add node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SwapRangeAddScalarPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &range = *nodes[0];
  const NodeProto &add = *nodes[1];
  core::builder::GraphBuilder &builder = graph.Builder();

  const std::string squeezed =
      builder.UniqueName("SwapRangeAddScalarPattern--" + add.input()[1].value());
  const std::string new_end =
      builder.UniqueName("SwapRangeAddScalarPattern--" + range.input()[1].value());

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto squeeze = MakeNode("Squeeze", {add.input()[1].value()}, {squeezed}, "",
                               ("SwapRangeAddScalarPattern--" + add.name().value()).c_str());
  if (add.has_doc_string()) {
    squeeze.set_doc_string(add.doc_string().value());
  }
  replacements.push_back(std::move(squeeze));

  NodeProto end_add = MakeNode("Add", {range.input()[1].value(), squeezed}, {new_end}, "",
                               ("SwapRangeAddScalarPattern--" + range.name().value()).c_str());
  if (range.has_doc_string()) {
    end_add.set_doc_string(range.doc_string().value());
  }
  replacements.push_back(std::move(end_add));

  bool start_is_zero = false;
  if (graph.IsConstant(range.input()[0].value())) {
    const TensorProto *start = graph.GetComputedConstant(range.input()[0].value());
    double value = 0.0;
    start_is_zero = start != nullptr && ReadScalarAsDouble(*start, value) && value == 0.0;
  }

  std::vector<std::string> range_inputs;
  if (start_is_zero) {
    range_inputs = {squeezed, new_end};
  } else {
    const std::string new_start =
        builder.UniqueName("SwapRangeAddScalarPattern--" + range.input()[0].value());
    NodeProto start_add = MakeNode("Add", {range.input()[0].value(), squeezed}, {new_start}, "",
                                   ("SwapRangeAddScalarPattern--" + range.name().value()).c_str());
    if (range.has_doc_string()) {
      start_add.set_doc_string(range.doc_string().value());
    }
    replacements.push_back(std::move(start_add));
    range_inputs = {new_start, new_end};
  }
  for (int index = 2; index < range.input_size(); ++index) {
    range_inputs.push_back(range.input()[index].value());
  }
  NodeProto replacement = MakeNode("Range", range_inputs, {add.output()[0].value()}, "",
                                   ("SwapRangeAddScalarPattern--" + range.name().value()).c_str());
  if (range.has_doc_string()) {
    replacement.set_doc_string(range.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
