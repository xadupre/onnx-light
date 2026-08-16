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
  if (!IsDefaultOp(candidate, "SplitToSequence")) {
    return NoMatch(candidate, "candidate is not a default-domain SplitToSequence");
  }
  std::vector<const NodeProto *> consumers;
  std::vector<int64_t> indices;
  if (!CollectSequenceAt(graph, candidate.output()[0].value(), consumers, indices)) {
    return NoMatch(candidate, "the sequence is not fully consumed by constant SequenceAt nodes");
  }
  std::vector<const NodeProto *> nodes;
  nodes.reserve(consumers.size() + 1);
  nodes.push_back(&candidate);
  nodes.insert(nodes.end(), consumers.begin(), consumers.end());
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

  // The Split output names: the final outputs when keepdims keeps every axis,
  // fresh intermediate names when a Squeeze must reduce the split axis.
  std::vector<std::string> split_outputs(count);
  for (std::size_t i = 0; i < count; ++i) {
    split_outputs[i] = keepdims != 0 ? final_outputs[i] : builder.UniqueName(name + "_split");
  }

  std::vector<std::string> split_inputs{sequence.input()[0].value()};
  const bool has_split_input = sequence.input_size() > 1;
  if (has_split_input) {
    split_inputs.push_back(sequence.input()[1].value());
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto split = MakeNode("Split", split_inputs, split_outputs, "", name.c_str());
  AddAttribute<int64_t>(split, "axis", axis);
  if (!has_split_input) {
    AddAttribute<int64_t>(split, "num_outputs", static_cast<int64_t>(count));
  }
  replacements.push_back(split);

  if (keepdims == 0) {
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
