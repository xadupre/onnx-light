// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/split_pattern.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::FreeInitializerName;
using collections::GetAxis;
using collections::IsDefaultOp;
using core::builder::BuilderError;

// Returns the rank of ``name`` into ``rank`` when its shape is known.
bool TryGetRank(core::builder::GraphGraph &graph, const std::string &name, int64_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = static_cast<int64_t>(graph.GetShape(name).Shape().Rank());
  return true;
}

// Reads a single-element constant integer ``name`` into ``value`` and reports
// its tensor rank (0 for a scalar, 1 for a single-element vector).
bool ConstScalarInt(core::builder::GraphGraph &graph, const std::string &name, int64_t &value,
                    int &rank) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(*tensor, values) || values.size() != 1) {
    return false;
  }
  value = values.front();
  rank = tensor->dims_size();
  return true;
}

struct PartitionRange {
  const NodeProto *node;
  int64_t start;
  int64_t end;
  bool squeeze;
};

bool ReadPartitionRange(core::builder::GraphGraph &graph, const NodeProto &node, int64_t rank,
                        int64_t dim_size, int64_t expected_axis, PartitionRange &range) {
  if (IsDefaultOp(node, "Gather")) {
    int64_t axis = GetAxis(node, 0);
    axis = axis < 0 ? axis + rank : axis;
    int index_rank = 0;
    int64_t index = 0;
    if (axis != expected_axis ||
        !ConstScalarInt(graph, node.input()[1].value(), index, index_rank) || index_rank != 0) {
      return false;
    }
    index = index < 0 ? index + dim_size : index;
    if (index < 0 || index >= dim_size) {
      return false;
    }
    range = PartitionRange{&node, index, index + 1, true};
    return true;
  }
  if (!IsDefaultOp(node, "Slice") || node.input_size() < 4 || node.input_size() > 5) {
    return false;
  }
  int axis_rank = 0;
  int64_t axis = 0;
  int start_rank = 0;
  int end_rank = 0;
  int64_t start = 0;
  int64_t end = 0;
  if (!ConstScalarInt(graph, node.input()[3].value(), axis, axis_rank) ||
      !ConstScalarInt(graph, node.input()[1].value(), start, start_rank) ||
      !ConstScalarInt(graph, node.input()[2].value(), end, end_rank)) {
    return false;
  }
  axis = axis < 0 ? axis + rank : axis;
  if (axis != expected_axis) {
    return false;
  }
  if (node.input_size() == 5) {
    int step_rank = 0;
    int64_t step = 0;
    if (!ConstScalarInt(graph, node.input()[4].value(), step, step_rank) || step != 1) {
      return false;
    }
  }
  start = start < 0 ? start + dim_size : start;
  end = end < 0 ? end + dim_size : end;
  start = std::max<int64_t>(0, std::min(start, dim_size));
  end = std::max<int64_t>(0, std::min(end, dim_size));
  if (start >= end) {
    return false;
  }
  range = PartitionRange{&node, start, end, false};
  return true;
}

bool CollectGatherSlicePartition(core::builder::GraphGraph &graph, const NodeProto &candidate,
                                 std::vector<PartitionRange> &ranges, int64_t &axis) {
  if (!IsDefaultOp(candidate, "Gather") && !IsDefaultOp(candidate, "Slice")) {
    return false;
  }
  const std::string &data = candidate.input()[0].value();
  if (!graph.HasShape(data)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
  const int64_t rank = static_cast<int64_t>(shape.Rank());
  if (IsDefaultOp(candidate, "Gather")) {
    axis = GetAxis(candidate, 0);
  } else {
    int axis_rank = 0;
    if (candidate.input_size() < 4 ||
        !ConstScalarInt(graph, candidate.input()[3].value(), axis, axis_rank)) {
      return false;
    }
  }
  axis = axis < 0 ? axis + rank : axis;
  if (axis < 0 || axis >= rank || !shape[static_cast<std::size_t>(axis)].IsInt()) {
    return false;
  }
  const int64_t dim_size = shape[static_cast<std::size_t>(axis)].AsInt();
  bool has_gather = false;
  bool has_slice = false;
  for (const NodeProto *consumer : graph.NextNodes(data)) {
    if (consumer->input_size() == 0 || consumer->input()[0].value() != data) {
      continue;
    }
    if (!IsDefaultOp(*consumer, "Gather") && !IsDefaultOp(*consumer, "Slice")) {
      continue;
    }
    PartitionRange range{};
    if (!ReadPartitionRange(graph, *consumer, rank, dim_size, axis, range)) {
      return false;
    }
    has_gather = has_gather || range.squeeze;
    has_slice = has_slice || !range.squeeze;
    ranges.push_back(range);
  }
  if (!has_gather || !has_slice ||
      std::find_if(ranges.begin(), ranges.end(), [&candidate](const PartitionRange &range) {
        return range.node == &candidate;
      }) == ranges.end()) {
    return false;
  }
  std::sort(ranges.begin(), ranges.end(),
            [](const PartitionRange &a, const PartitionRange &b) { return a.start < b.start; });
  int64_t expected = 0;
  for (const PartitionRange &range : ranges) {
    if (range.start != expected) {
      return false;
    }
    expected = range.end;
  }
  return expected == dim_size;
}

} // namespace

std::set<std::string> SplitConcatPattern::FastOpType() const { return {"Split"}; }

core::builder::MatchResult SplitConcatPattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Split")) {
    return NoMatch(candidate, "candidate is not a default-domain Split");
  }
  const NodeProto *concat = nullptr;
  for (const auto &output : candidate.output()) {
    const std::vector<const NodeProto *> &consumers = graph.NextNodes(output.value());
    if (consumers.size() != 1) {
      return NoMatch(candidate, "a Split output is not consumed by exactly one node");
    }
    if (concat == nullptr) {
      concat = consumers[0];
    } else if (concat != consumers[0]) {
      return NoMatch(candidate, "the Split outputs feed more than one consumer");
    }
  }
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat")) {
    return NoMatch(candidate, "the single consumer is not a default-domain Concat");
  }

  int64_t axis_split = GetAxis(candidate, 0);
  int64_t axis_concat = GetAxis(*concat, 0);
  if ((axis_split < 0) != (axis_concat < 0)) {
    int64_t rank = 0;
    if (!TryGetRank(graph, candidate.input()[0].value(), rank)) {
      return NoMatch(candidate, "the input rank is required to compare mixed-sign axes");
    }
    if (axis_split < 0) {
      axis_split += rank;
    }
    if (axis_concat < 0) {
      axis_concat += rank;
    }
  }
  if (axis_split != axis_concat) {
    return NoMatch(candidate, "the Split and Concat axes differ");
  }

  if (candidate.output_size() != concat->input_size()) {
    return NoMatch(candidate, "the Concat does not re-join every Split output");
  }
  for (int i = 0; i < candidate.output_size(); ++i) {
    if (candidate.output()[i].value() != concat->input()[i].value()) {
      return NoMatch(candidate, "the Concat inputs are not the Split outputs in order");
    }
  }

  return core::builder::MatchResult{this, {&candidate, concat}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SplitConcatPattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SplitConcatPattern::Apply expects a Split and a Concat node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SplitConcatPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &split = *nodes[0];
  const NodeProto &concat = *nodes[1];

  const std::string name = "SplitConcatPattern--" + split.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Identity", {split.input()[0].value()},
                                  {concat.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> GathersSplitPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult GathersSplitPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gather")) {
    return NoMatch(candidate, "candidate is not a default-domain Gather");
  }
  const std::string &data = candidate.input()[0].value();
  if (!graph.HasShape(data)) {
    return NoMatch(candidate, "the gathered value has no known shape");
  }
  std::vector<const NodeProto *> gather_users;
  for (const NodeProto *user : graph.NextNodes(data)) {
    if (IsDefaultOp(*user, "Gather")) {
      gather_users.push_back(user);
    }
  }
  if (gather_users.size() <= 1) {
    return NoMatch(candidate, "there is at most one sibling Gather on this value");
  }

  int64_t axis = 0;
  bool axis_seen = false;
  int index_rank = -1;
  std::set<int64_t> indices;
  std::vector<const NodeProto *> kept;
  for (const NodeProto *user : gather_users) {
    if (user->input_size() != 2) {
      continue;
    }
    const int64_t user_axis = GetAxis(*user, 0);
    if (axis_seen && user_axis != axis) {
      return NoMatch(candidate, "sibling Gather nodes disagree on the axis");
    }
    axis = user_axis;
    axis_seen = true;

    int64_t value = 0;
    int rank = 0;
    if (!ConstScalarInt(graph, user->input()[1].value(), value, rank)) {
      continue;
    }
    if (indices.count(value) != 0) {
      return NoMatch(candidate, "two sibling Gather nodes share an index");
    }
    if (index_rank >= 0 && rank != index_rank) {
      return NoMatch(candidate, "sibling Gather indices differ in rank");
    }
    index_rank = rank;
    indices.insert(value);
    kept.push_back(user);
  }
  if (std::find(kept.begin(), kept.end(), &candidate) == kept.end()) {
    return NoMatch(candidate, "the candidate is not a constant-index Gather");
  }

  int64_t expected = 0;
  for (int64_t value : indices) {
    if (value != expected) {
      return NoMatch(candidate, "the sibling Gather indices are not 0..n-1");
    }
    ++expected;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
  const int64_t rank = static_cast<int64_t>(shape.Rank());
  const int64_t normalized = axis < 0 ? axis + rank : axis;
  if (normalized < 0 || normalized >= rank) {
    return NoMatch(candidate, "the Gather axis is out of range");
  }
  const auto &dim = shape[static_cast<std::size_t>(normalized)];
  if (!dim.IsInt() || dim.AsInt() != static_cast<int64_t>(indices.size())) {
    return NoMatch(candidate, "the axis dimension does not match the number of gathers");
  }
  return core::builder::MatchResult{this, kept, nullptr};
}

utils::RepeatedProtoField<NodeProto>
GathersSplitPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr) {
    throw BuilderError("GathersSplitPattern::Apply expects at least one Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GathersSplitPattern::Apply received an unsafe or inconsistent match.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const int64_t axis = GetAxis(*nodes[0], 0);
  const std::size_t count = nodes.size();

  int probe_rank = 0;
  int64_t probe_value = 0;
  ConstScalarInt(graph, nodes[0]->input()[1].value(), probe_value, probe_rank);
  const bool scalar_index = probe_rank == 0;

  const std::string name = "GathersSplitPattern--" + nodes[0]->name().value();
  std::string axes_init;
  if (scalar_index) {
    axes_init = FreeInitializerName(builder, name + "_axes");
    builder.MakeInitializer(MakeInitializerShape(axes_init.c_str(), {axis}));
  }

  std::vector<std::string> outputs(count);
  utils::RepeatedProtoField<NodeProto> post_nodes;
  for (const NodeProto *user : nodes) {
    int64_t value = 0;
    int rank = 0;
    if (!ConstScalarInt(graph, user->input()[1].value(), value, rank)) {
      throw BuilderError("GathersSplitPattern::Apply could not read a Gather index.");
    }
    const std::size_t slot = static_cast<std::size_t>(value);
    if (scalar_index) {
      const std::string intermediate = builder.UniqueName(name + "_split");
      const std::string squeeze_name = name + "-Squeeze-" + std::to_string(value);
      post_nodes.push_back(MakeNode("Squeeze", {intermediate, axes_init},
                                    {user->output()[0].value()}, "", squeeze_name.c_str()));
      outputs[slot] = intermediate;
    } else {
      outputs[slot] = user->output()[0].value();
    }
  }

  NodeProto split =
      MakeNode("Split", {nodes[0]->input()[0].value()}, outputs, "", (name + "-Split").c_str());
  AddAttribute<int64_t>(split, "axis", axis);
  AddAttribute<int64_t>(split, "num_outputs", static_cast<int64_t>(count));

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(split);
  for (const auto &post : post_nodes) {
    replacements.add() = post;
  }
  return replacements;
}

std::set<std::string> GatherSliceToSplitPattern::FastOpType() const { return {"Gather", "Slice"}; }

core::builder::MatchResult GatherSliceToSplitPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 13) {
    return NoMatch(candidate, "the default opset is older than 13");
  }
  std::vector<PartitionRange> ranges;
  int64_t axis = 0;
  if (!CollectGatherSlicePartition(graph, candidate, ranges, axis)) {
    return NoMatch(candidate, "the sibling Gather/Slice ranges do not exactly partition one axis");
  }
  std::vector<const NodeProto *> nodes;
  nodes.reserve(ranges.size());
  for (const PartitionRange &range : ranges) {
    nodes.push_back(range.node);
  }
  return core::builder::MatchResult{this, nodes, nullptr};
}

utils::RepeatedProtoField<NodeProto>
GatherSliceToSplitPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() < 2 || nodes[0] == nullptr) {
    throw BuilderError("GatherSliceToSplitPattern::Apply expects sibling Gather/Slice nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "GatherSliceToSplitPattern::Apply received an unsafe or inconsistent match.");
  }

  std::vector<PartitionRange> ranges;
  int64_t axis = 0;
  if (!CollectGatherSlicePartition(graph, *nodes[0], ranges, axis)) {
    throw BuilderError("GatherSliceToSplitPattern::Apply could not rebuild the partition.");
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "GatherSliceToSplitPattern--" + nodes[0]->name().value();
  const std::string splits = FreeInitializerName(builder, name + "_splits");
  std::vector<int64_t> sizes;
  sizes.reserve(ranges.size());
  for (const PartitionRange &range : ranges) {
    sizes.push_back(range.end - range.start);
  }
  builder.MakeInitializer(MakeInitializerShape(splits.c_str(), sizes));

  std::vector<std::string> outputs;
  utils::RepeatedProtoField<NodeProto> squeezes;
  std::string axes;
  outputs.reserve(ranges.size());
  for (const PartitionRange &range : ranges) {
    if (range.squeeze) {
      if (axes.empty()) {
        axes = FreeInitializerName(builder, name + "_axes");
        builder.MakeInitializer(MakeInitializerShape(axes.c_str(), {axis}));
      }
      const std::string intermediate = builder.UniqueName(name + "_split");
      outputs.push_back(intermediate);
      squeezes.push_back(MakeNode("Squeeze", {intermediate, axes},
                                  {range.node->output()[0].value()}, "",
                                  builder.UniqueName(name + "_squeeze").c_str()));
    } else {
      outputs.push_back(range.node->output()[0].value());
    }
  }
  NodeProto split = MakeNode("Split", {nodes[0]->input()[0].value(), splits}, outputs, "",
                             builder.UniqueName(name + "_split_node").c_str());
  AddAttribute<int64_t>(split, "axis", axis);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(split);
  for (const NodeProto &squeeze : squeezes) {
    replacements.push_back(squeeze);
  }
  return replacements;
}

std::set<std::string> SlicesSplitPattern::FastOpType() const { return {"Slice"}; }

core::builder::MatchResult SlicesSplitPattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Slice")) {
    return NoMatch(candidate, "candidate is not a default-domain Slice");
  }
  const std::string &data = candidate.input()[0].value();
  if (!graph.HasShape(data)) {
    return NoMatch(candidate, "the sliced value has no known shape");
  }
  std::vector<const NodeProto *> users;
  for (const NodeProto *user : graph.NextNodes(data)) {
    if (IsDefaultOp(*user, "Slice")) {
      users.push_back(user);
    }
  }
  if (users.size() <= 1) {
    return NoMatch(candidate, "there is at most one sibling Slice on this value");
  }

  for (const NodeProto *user : users) {
    if (user->input_size() == 4) {
      continue;
    }
    if (user->input_size() == 5) {
      int64_t step = 0;
      int rank = 0;
      if (!ConstScalarInt(graph, user->input()[4].value(), step, rank) || step != 1) {
        return NoMatch(candidate, "a sibling Slice uses a non-unit step");
      }
      continue;
    }
    return NoMatch(candidate, "a sibling Slice does not provide axes");
  }

  int64_t axis = 0;
  bool axis_seen = false;
  for (const NodeProto *user : users) {
    int64_t user_axis = 0;
    int rank = 0;
    if (!ConstScalarInt(graph, user->input()[3].value(), user_axis, rank)) {
      return NoMatch(candidate, "a sibling Slice has a non-constant axis");
    }
    if (axis_seen && user_axis != axis) {
      return NoMatch(candidate, "sibling Slice nodes disagree on the axis");
    }
    axis = user_axis;
    axis_seen = true;
  }

  const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
  const int64_t rank = static_cast<int64_t>(shape.Rank());
  const int64_t normalized = axis < 0 ? axis + rank : axis;
  if (normalized < 0 || normalized >= rank) {
    return NoMatch(candidate, "the Slice axis is out of range");
  }
  const auto &dim = shape[static_cast<std::size_t>(normalized)];
  if (!dim.IsInt()) {
    return NoMatch(candidate, "the sliced axis dimension is not static");
  }

  std::vector<int64_t> starts(users.size());
  std::vector<int64_t> ends(users.size());
  for (std::size_t i = 0; i < users.size(); ++i) {
    int start_rank = 0;
    int end_rank = 0;
    if (!ConstScalarInt(graph, users[i]->input()[1].value(), starts[i], start_rank) ||
        !ConstScalarInt(graph, users[i]->input()[2].value(), ends[i], end_rank)) {
      return NoMatch(candidate, "a sibling Slice has non-constant bounds");
    }
  }
  if (starts.front() != 0) {
    return NoMatch(candidate, "the first sibling Slice does not start at zero");
  }
  const int64_t sentinel = std::numeric_limits<int64_t>::max();
  if (ends.back() != dim.AsInt() && ends.back() != sentinel) {
    return NoMatch(candidate, "the last sibling Slice does not end at the axis dimension");
  }
  for (std::size_t i = 0; i + 1 < users.size(); ++i) {
    if (ends[i] != starts[i + 1]) {
      return NoMatch(candidate, "the sibling Slice ranges are not contiguous");
    }
  }
  if (std::find(users.begin(), users.end(), &candidate) == users.end()) {
    return NoMatch(candidate, "the candidate is not among the sibling slices");
  }
  return core::builder::MatchResult{this, users, nullptr};
}

utils::RepeatedProtoField<NodeProto>
SlicesSplitPattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr) {
    throw BuilderError("SlicesSplitPattern::Apply expects at least one Slice node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SlicesSplitPattern::Apply received an unsafe or inconsistent match.");
  }

  const std::string &data = nodes[0]->input()[0].value();
  int axis_rank = 0;
  int64_t axis = 0;
  ConstScalarInt(graph, nodes[0]->input()[3].value(), axis, axis_rank);
  const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
  const int64_t rank = static_cast<int64_t>(shape.Rank());
  const int64_t normalized = axis < 0 ? axis + rank : axis;
  const int64_t total = shape[static_cast<std::size_t>(normalized)].AsInt();
  const int64_t sentinel = std::numeric_limits<int64_t>::max();

  std::vector<int64_t> sizes;
  std::vector<std::string> outputs;
  sizes.reserve(nodes.size());
  outputs.reserve(nodes.size());
  for (const NodeProto *user : nodes) {
    int64_t start = 0;
    int64_t end = 0;
    int start_rank = 0;
    int end_rank = 0;
    ConstScalarInt(graph, user->input()[1].value(), start, start_rank);
    ConstScalarInt(graph, user->input()[2].value(), end, end_rank);
    if (end == sentinel) {
      end = total;
    }
    int64_t delta = 0;
    if ((end < 0 && start >= 0) || (end >= 0 && start < 0)) {
      delta = end < 0 ? (end + total - start) : (end - start - total);
    } else {
      delta = end - start;
    }
    if (delta < 0) {
      throw BuilderError("SlicesSplitPattern::Apply computed a negative split size.");
    }
    sizes.push_back(delta);
    outputs.push_back(user->output()[0].value());
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SlicesSplitPattern--" + nodes[0]->name().value();
  const std::string splits = FreeInitializerName(builder, name + "_splits");
  builder.MakeInitializer(MakeInitializerShape(splits.c_str(), sizes));

  NodeProto split = MakeNode("Split", {data, splits}, outputs, "", name.c_str());
  AddAttribute<int64_t>(split, "axis", axis);

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(split);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
