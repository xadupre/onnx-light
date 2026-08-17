// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

bool IsSqueezeOrUnsqueeze(const NodeProto &node) {
  return IsDefaultOp(node, "Squeeze") || IsDefaultOp(node, "Unsqueeze");
}

/// Returns a fresh initializer name derived from ``base`` that is not already
/// used by ``builder`` (without reserving it, so ``MakeInitializer`` can).
std::string FreeInitializerName(core::builder::GraphBuilder &builder, const std::string &base) {
  if (!builder.HasName(base)) {
    return base;
  }
  std::string candidate;
  for (int suffix = 0;; ++suffix) {
    candidate = base + "_" + std::to_string(suffix);
    if (!builder.HasName(candidate)) {
      return candidate;
    }
  }
}

/// Reads the integer payload of the constant ``name`` into ``values`` when it is
/// a one-dimensional (or scalar) INT32/INT64 tensor.
bool ReadConstantAxes(core::builder::GraphGraph &graph, const std::string &name,
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
  if (tensor->dims_size() > 1) {
    return false;
  }
  return ReadIntegerValues(*tensor, values);
}

/// Returns ``true`` when every entry of ``axes`` is non-negative.
bool AllNonNegative(const std::vector<int64_t> &axes) {
  return std::all_of(axes.begin(), axes.end(), [](int64_t a) { return a >= 0; });
}

/// Returns the rank of ``name`` when its shape is known.
bool GetRank(core::builder::GraphGraph &graph, const std::string &name, std::size_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = graph.GetShape(name).Shape().Rank();
  return true;
}

/// Composes ``axis_first`` applied before ``axis_second`` into the equivalent
/// single set of ``Unsqueeze`` axes over a tensor of rank ``rank``.
std::vector<int64_t> CombineUnsqueezeAxes(const std::vector<int64_t> &axis_first,
                                          const std::vector<int64_t> &axis_second,
                                          std::size_t rank) {
  if (axis_first.size() == 1 && axis_second.size() == 1) {
    const int64_t a1 = axis_first[0];
    const int64_t a2 = axis_second[0];
    if (a1 < a2) {
      return {a1, a2};
    }
    return {a2, a1 + 1};
  }

  std::vector<bool> existing(rank, false);
  for (int64_t a : axis_first) {
    const std::size_t pos = std::min(static_cast<std::size_t>(a), existing.size());
    existing.insert(existing.begin() + static_cast<std::ptrdiff_t>(pos), true);
  }
  for (int64_t a : axis_second) {
    const std::size_t pos = std::min(static_cast<std::size_t>(a), existing.size());
    existing.insert(existing.begin() + static_cast<std::ptrdiff_t>(pos), true);
  }
  std::vector<int64_t> combined;
  for (std::size_t i = 0; i < existing.size(); ++i) {
    if (existing[i]) {
      combined.push_back(static_cast<int64_t>(i));
    }
  }
  return combined;
}

} // namespace

std::set<std::string> UnsqueezeUnsqueezePattern::FastOpType() const { return {"Unsqueeze"}; }

core::builder::MatchResult UnsqueezeUnsqueezePattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Unsqueeze") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Unsqueeze with two inputs");
  }
  const NodeProto *next_node = nullptr;
  for (const NodeProto *consumer : graph.NextNodes(candidate.output()[0].value())) {
    if (consumer != nullptr && IsDefaultOp(*consumer, "Unsqueeze")) {
      next_node = consumer;
      break;
    }
  }
  if (next_node == nullptr) {
    return NoMatch(candidate, "the Unsqueeze output is not consumed by another Unsqueeze");
  }
  if (next_node->input_size() != 2 ||
      next_node->input()[0].value() != candidate.output()[0].value()) {
    return NoMatch(candidate, "the second Unsqueeze does not consume the first as its data input");
  }

  std::vector<int64_t> axis_first;
  std::vector<int64_t> axis_second;
  if (!ReadConstantAxes(graph, candidate.input()[1].value(), axis_first) ||
      !ReadConstantAxes(graph, next_node->input()[1].value(), axis_second)) {
    return NoMatch(candidate, "both Unsqueeze nodes must have one-dimensional constant axes");
  }
  if (!AllNonNegative(axis_first) || !AllNonNegative(axis_second)) {
    return NoMatch(candidate, "the Unsqueeze axes must be non-negative");
  }
  if (axis_first.size() > 1 || axis_second.size() > 1) {
    std::size_t rank = 0;
    if (!GetRank(graph, candidate.input()[0].value(), rank)) {
      return NoMatch(candidate, "the source rank is required to merge multi-axis Unsqueeze nodes");
    }
  }
  return core::builder::MatchResult{this, {&candidate, next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
UnsqueezeUnsqueezePattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("UnsqueezeUnsqueezePattern::Apply expects two consecutive Unsqueeze nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "UnsqueezeUnsqueezePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];

  std::vector<int64_t> axis_first;
  std::vector<int64_t> axis_second;
  ReadConstantAxes(graph, first.input()[1].value(), axis_first);
  ReadConstantAxes(graph, second.input()[1].value(), axis_second);

  std::size_t rank = 0;
  GetRank(graph, first.input()[0].value(), rank);
  const std::vector<int64_t> combined = CombineUnsqueezeAxes(axis_first, axis_second, rank);

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "UnsqueezeUnsqueezePattern--" + first.name().value();
  const std::string axes_init = builder.MakeInitializer(
      MakeInitializerShape(FreeInitializerName(builder, name + "_axes").c_str(), combined));

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(first.output()[0].value())) {
    replacements.add() = first;
  }
  replacements.push_back(MakeNode("Unsqueeze", {first.input()[0].value(), axes_init},
                                  {second.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> SqueezeUnsqueezePattern::FastOpType() const {
  return {"Squeeze", "Unsqueeze"};
}

namespace {

/// Result of comparing the axes of a ``Squeeze``/``Unsqueeze`` pair.
struct AxesDiff {
  bool matched = false;
  std::string op_type;
  bool has_axes = false;
  std::vector<int64_t> axes;
};

/// Collects the indices of the size-one dimensions of ``name`` when its shape is
/// fully known for those axes. Returns ``false`` when the shape is unknown.
bool SqueezeAxesFromShape(core::builder::GraphGraph &graph, const std::string &name,
                          std::vector<int64_t> &axes) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    const auto &dim = shape[i];
    if (dim.IsInt() && dim.AsInt() == 1) {
      axes.push_back(static_cast<int64_t>(i));
    }
  }
  return true;
}

/// Reads the effective axes of ``node`` (a ``Squeeze`` or ``Unsqueeze``) into
/// ``axes``. ``has_axes`` reports whether an explicit axes input is present.
bool ResolveAxes(core::builder::GraphGraph &graph, const NodeProto &node,
                 std::vector<int64_t> &axes) {
  if (node.input_size() == 2) {
    return ReadConstantAxes(graph, node.input()[1].value(), axes);
  }
  if (IsDefaultOp(node, "Squeeze")) {
    return SqueezeAxesFromShape(graph, node.input()[0].value(), axes);
  }
  return false;
}

/// Returns ``true`` when ``axes`` is the contiguous range ``[min, max]``.
bool IsContiguous(const std::vector<int64_t> &axes) {
  const auto minmax = std::minmax_element(axes.begin(), axes.end());
  const int64_t span = *minmax.second - *minmax.first + 1;
  if (span != static_cast<int64_t>(axes.size())) {
    return false;
  }
  std::set<int64_t> unique(axes.begin(), axes.end());
  return static_cast<int64_t>(unique.size()) == span;
}

AxesDiff DiffAxes(core::builder::GraphGraph &graph, const NodeProto &first,
                  const NodeProto &second) {
  AxesDiff diff;
  if (IsDefaultOp(first, "Unsqueeze") && second.input_size() == 1) {
    diff.matched = true;
    diff.op_type = "Squeeze";
    return diff;
  }

  std::vector<int64_t> axes_first;
  std::vector<int64_t> axes_second;
  if (!ResolveAxes(graph, first, axes_first) || !ResolveAxes(graph, second, axes_second)) {
    return diff;
  }

  const std::set<int64_t> set_first(axes_first.begin(), axes_first.end());
  const std::set<int64_t> set_second(axes_second.begin(), axes_second.end());

  if (set_first == set_second) {
    if (axes_first.size() > 1 && !IsContiguous(axes_first)) {
      return diff;
    }
    diff.matched = true;
    diff.op_type = "Identity";
    return diff;
  }

  if (IsDefaultOp(first, "Unsqueeze") &&
      std::includes(set_second.begin(), set_second.end(), set_first.begin(), set_first.end())) {
    std::vector<int64_t> keep;
    for (int64_t a : set_second) {
      if (set_first.find(a) == set_first.end()) {
        keep.push_back(a);
      }
    }
    std::sort(keep.begin(), keep.end());
    for (int64_t &value : keep) {
      const int64_t below = static_cast<int64_t>(std::count_if(
          axes_first.begin(), axes_first.end(), [value](int64_t a) { return a < value; }));
      value -= below;
    }
    diff.matched = true;
    diff.op_type = "Squeeze";
    diff.has_axes = true;
    diff.axes = std::move(keep);
    return diff;
  }

  return diff;
}

} // namespace

core::builder::MatchResult SqueezeUnsqueezePattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (!IsSqueezeOrUnsqueeze(candidate) || candidate.input_size() < 1 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Squeeze or Unsqueeze");
  }
  const NodeProto *node_before = graph.NodeBefore(candidate.input()[0].value());
  if (node_before == nullptr || !IsSqueezeOrUnsqueeze(*node_before) ||
      node_before->op_type().value() == candidate.op_type().value() ||
      node_before->output_size() != 1) {
    return NoMatch(candidate, "the input is not produced by the opposite Squeeze/Unsqueeze");
  }
  const AxesDiff diff = DiffAxes(graph, *node_before, candidate);
  if (!diff.matched) {
    return NoMatch(candidate, "the Squeeze/Unsqueeze axes do not simplify");
  }
  const NodeProto *insert_at =
      graph.IsUsedMoreThanOnce(candidate.input()[0].value()) ? node_before : &candidate;
  return core::builder::MatchResult{this, {node_before, &candidate}, insert_at};
}

utils::RepeatedProtoField<NodeProto>
SqueezeUnsqueezePattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SqueezeUnsqueezePattern::Apply expects one Squeeze and one Unsqueeze.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SqueezeUnsqueezePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];

  const AxesDiff diff = DiffAxes(graph, first, second);
  if (!diff.matched) {
    throw BuilderError("SqueezeUnsqueezePattern::Apply received a match that no longer holds.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SqueezeUnsqueezePattern--" + first.name().value();

  NodeProto new_node;
  if (diff.has_axes) {
    const std::string axes_init = builder.MakeInitializer(
        MakeInitializerShape(FreeInitializerName(builder, name + "_axes").c_str(), diff.axes));
    new_node = MakeNode(diff.op_type.c_str(), {first.input()[0].value(), axes_init},
                        {second.output()[0].value()}, "", name.c_str());
  } else {
    new_node = MakeNode(diff.op_type.c_str(), {first.input()[0].value()},
                        {second.output()[0].value()}, "", name.c_str());
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(second.input()[0].value())) {
    replacements.add() = first;
  }
  replacements.push_back(std::move(new_node));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
