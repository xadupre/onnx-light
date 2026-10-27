// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/sequence_pattern.h"

#include <algorithm>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::GetAxis;
using collections::IsDefaultOp;
using collections::ReadScalarInt;
using core::builder::BuilderError;

// Collects the constant scalar SequenceAt indices consuming ``seq_output``.
// Returns ``false`` (and leaves ``indices`` unspecified) when any consumer is
// not a SequenceAt with a constant scalar index, or the indices do not cover
// ``0 .. n-1`` exactly once. On success ``consumers`` holds the consuming nodes
// in graph order and ``indices[i]`` the index read from ``consumers[i]``.
bool CollectSequenceAt(core::builder::GraphGraph &graph, const std::string &seq_output,
                       std::vector<const NodeProto *> &consumers, std::vector<int64_t> &indices) {
  consumers = graph.NextNodes(seq_output);
  if (consumers.empty()) {
    return false;
  }
  indices.clear();
  std::vector<bool> seen(consumers.size(), false);
  for (const NodeProto *consumer : consumers) {
    if (consumer == nullptr || !IsDefaultOp(*consumer, "SequenceAt") ||
        consumer->input_size() < 2) {
      return false;
    }
    const std::string &index_name = consumer->input()[1].value();
    if (!graph.IsConstantScalar(index_name)) {
      return false;
    }
    const TensorProto *value = graph.GetComputedConstant(index_name);
    int64_t index = 0;
    if (value == nullptr || !ReadScalarInt(*value, index)) {
      return false;
    }
    if (index < 0 || static_cast<std::size_t>(index) >= consumers.size() ||
        seen[static_cast<std::size_t>(index)]) {
      return false;
    }
    seen[static_cast<std::size_t>(index)] = true;
    indices.push_back(index);
  }
  return true;
}

} // namespace

std::set<std::string> SequenceConstructAtPattern::FastOpType() const {
  return {"SequenceConstruct"};
}

core::builder::MatchResult SequenceConstructAtPattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (!graph.IsOperator(candidate))
    return NoMatch(candidate, "candidate is a local function call or an overloaded operator");
  if (!IsDefaultOp(candidate, "SequenceConstruct")) {
    return NoMatch(candidate, "candidate is not a default-domain SequenceConstruct");
  }
  std::vector<const NodeProto *> consumers;
  std::vector<int64_t> indices;
  if (!CollectSequenceAt(graph, candidate.output()[0].value(), consumers, indices)) {
    return NoMatch(candidate, "the sequence is not fully consumed by constant SequenceAt nodes");
  }
  if (consumers.size() != static_cast<std::size_t>(candidate.input_size())) {
    return NoMatch(candidate, "the SequenceAt count does not match the SequenceConstruct inputs");
  }
  std::vector<const NodeProto *> nodes;
  nodes.reserve(consumers.size() + 1);
  nodes.push_back(&candidate);
  nodes.insert(nodes.end(), consumers.begin(), consumers.end());
  for (const NodeProto *node : nodes) {
    if (node != nullptr && !graph.IsOperator(*node)) {
      return NoMatch(candidate, "a matched node is a local function call or overload");
    }
  }
  return core::builder::MatchResult{this, std::move(nodes), &candidate};
}

utils::RepeatedProtoField<NodeProto>
SequenceConstructAtPattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr) {
    throw BuilderError("SequenceConstructAtPattern::Apply expects a SequenceConstruct node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SequenceConstructAtPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &sequence = *nodes[0];
  const std::string name = "SequenceConstructAtPattern--" + sequence.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const NodeProto &at = *nodes[i];
    const TensorProto *value = graph.GetComputedConstant(at.input()[1].value());
    int64_t index = 0;
    if (value == nullptr || !ReadScalarInt(*value, index) || index < 0 ||
        index >= sequence.input_size()) {
      throw BuilderError("SequenceConstructAtPattern::Apply found an out-of-range index.");
    }
    replacements.push_back(MakeNode("Identity", {sequence.input()[index].value()},
                                    {at.output()[0].value()}, "", name.c_str()));
  }
  return replacements;
}

std::set<std::string> SplitToSequenceSequenceAtPattern::FastOpType() const {
  return {"SplitToSequence"};
}

core::builder::MatchResult
SplitToSequenceSequenceAtPattern::Match(core::builder::GraphGraph &graph,
                                        const NodeProto &candidate) const {
  if (!graph.IsOperator(candidate))
    return NoMatch(candidate, "candidate is a local function call or an overloaded operator");
  if (!IsDefaultOp(candidate, "SplitToSequence")) {
    return NoMatch(candidate, "candidate is not a default-domain SplitToSequence");
  }
  std::vector<const NodeProto *> consumers;
  std::vector<int64_t> indices;
  if (!CollectSequenceAt(graph, candidate.output()[0].value(), consumers, indices)) {
    return NoMatch(candidate, "the sequence is not fully consumed by constant SequenceAt nodes");
  }
  if (candidate.input_size() > 1 && !candidate.input()[1].value().empty()) {
    const std::string &split_name = candidate.input()[1].value();
    if (!graph.HasShape(split_name)) {
      return NoMatch(candidate, "the split input shape is unknown");
    }
    if (graph.GetShape(split_name).Shape().Rank() == 0) {
      const TensorProto *split = graph.GetComputedConstant(split_name);
      int64_t chunk = 0;
      const std::string &data = candidate.input()[0].value();
      if (split == nullptr || split->dims_size() != 0 || !ReadScalarInt(*split, chunk) ||
          chunk <= 0 || !graph.HasShape(data)) {
        return NoMatch(candidate, "scalar split requires a positive constant and a known shape");
      }
      const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
      const int64_t rank = static_cast<int64_t>(shape.Rank());
      int64_t axis = GetAxis(candidate, 0);
      axis = axis < 0 ? axis + rank : axis;
      if (axis < 0 || axis >= rank || !shape[static_cast<std::size_t>(axis)].IsInt()) {
        return NoMatch(candidate, "scalar split requires a known axis dimension");
      }
      const int64_t dim = shape[static_cast<std::size_t>(axis)].AsInt();
      const int64_t expected = dim == 0 ? 1 : dim / chunk + (dim % chunk != 0);
      if (expected != static_cast<int64_t>(consumers.size())) {
        return NoMatch(candidate, "SequenceAt count does not match scalar split chunks");
      }
    }
  }
  std::vector<const NodeProto *> nodes;
  nodes.reserve(consumers.size() + 1);
  nodes.push_back(&candidate);
  nodes.insert(nodes.end(), consumers.begin(), consumers.end());
  for (const NodeProto *node : nodes) {
    if (node != nullptr && !graph.IsOperator(*node)) {
      return NoMatch(candidate, "a matched node is a local function call or overload");
    }
  }
  return core::builder::MatchResult{this, std::move(nodes), &candidate};
}

utils::RepeatedProtoField<NodeProto>
SplitToSequenceSequenceAtPattern::Apply(core::builder::GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr) {
    throw BuilderError("SplitToSequenceSequenceAtPattern::Apply expects a SplitToSequence node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SplitToSequenceSequenceAtPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &sequence = *nodes[0];
  const int64_t axis = GetAxis(sequence, 0);
  const int64_t keepdims = GetAttributeOr<int64_t>(sequence, "keepdims", 1);
  const std::size_t count = nodes.size() - 1;

  // Map each split index to the final SequenceAt output name.
  std::vector<std::string> final_outputs(count);
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const NodeProto &at = *nodes[i];
    const TensorProto *value = graph.GetComputedConstant(at.input()[1].value());
    int64_t index = 0;
    if (value == nullptr || !ReadScalarInt(*value, index) || index < 0 ||
        static_cast<std::size_t>(index) >= count) {
      throw BuilderError("SplitToSequenceSequenceAtPattern::Apply found an out-of-range index.");
    }
    final_outputs[static_cast<std::size_t>(index)] = at.output()[0].value();
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SplitToSequenceSequenceAtPattern--" + sequence.name().value();

  const bool has_split_input = sequence.input_size() > 1 && !sequence.input()[1].value().empty();
  const bool squeeze = !has_split_input && keepdims == 0;
  // The Split output names: the final outputs when keepdims keeps every axis,
  // fresh intermediate names when a Squeeze must reduce the split axis.
  std::vector<std::string> split_outputs(count);
  for (std::size_t i = 0; i < count; ++i) {
    split_outputs[i] = !squeeze ? final_outputs[i] : builder.UniqueName(name + "_split");
  }

  std::vector<std::string> split_inputs{sequence.input()[0].value()};
  if (has_split_input) {
    const std::string &split_name = sequence.input()[1].value();
    if (graph.GetShape(split_name).Shape().Rank() == 0) {
      int64_t chunk = 0;
      ReadScalarInt(*graph.GetComputedConstant(split_name), chunk);
      const core::symbolic::SymShape &shape = graph.GetShape(sequence.input()[0].value()).Shape();
      const int64_t rank = static_cast<int64_t>(shape.Rank());
      const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
      int64_t remaining = shape[static_cast<std::size_t>(resolved_axis)].AsInt();
      std::vector<int64_t> sizes;
      while (remaining > 0) {
        const int64_t take = std::min(remaining, chunk);
        sizes.push_back(take);
        remaining -= take;
      }
      if (sizes.empty()) {
        sizes.push_back(0);
      }
      const std::string sizes_name = collections::FreeInitializerName(builder, name + "_sizes");
      builder.MakeInitializer(MakeInitializerShape(sizes_name.c_str(), sizes));
      split_inputs.push_back(sizes_name);
    } else {
      split_inputs.push_back(split_name);
    }
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto split = MakeNode("Split", split_inputs, split_outputs, "", name.c_str());
  AddAttribute<int64_t>(split, "axis", axis);
  if (!has_split_input) {
    AddAttribute<int64_t>(split, "num_outputs", static_cast<int64_t>(count));
  }
  replacements.push_back(split);

  if (squeeze) {
    const std::string axes_name = collections::FreeInitializerName(builder, name + "_axes");
    builder.MakeInitializer(MakeInitializerShape(axes_name.c_str(), {axis}));
    for (std::size_t i = 0; i < count; ++i) {
      replacements.push_back(
          MakeNode("Squeeze", {split_outputs[i], axes_name}, {final_outputs[i]}, "", name.c_str()));
    }
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
