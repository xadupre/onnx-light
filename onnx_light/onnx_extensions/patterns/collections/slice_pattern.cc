// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/slice_pattern.h"

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
using collections::IsDefaultOp;
using core::builder::BuilderError;

bool ReadConstantIntegers(core::builder::GraphGraph &graph, const NodeProto &node, int index,
                          std::vector<int64_t> &values) {
  if (index >= node.input_size() || node.input()[index].value().empty() ||
      !graph.IsConstant(node.input()[index].value())) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(node.input()[index].value());
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

// Reads the constant ``axes`` values (input index 3) of a Slice into ``out``.
bool ReadAxes(core::builder::GraphGraph &graph, const NodeProto &slice, std::vector<int64_t> &out) {
  if (slice.input_size() < 4) {
    return false;
  }
  const std::string &name = slice.input()[3].value();
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *axes = graph.GetComputedConstant(name);
  return axes != nullptr && ReadIntegerValues(*axes, out);
}

// Emits a Concat(axis=0) of ``lhs`` and ``rhs`` producing ``output``.
NodeProto ConcatAxis0(const std::string &lhs, const std::string &rhs, const std::string &output,
                      const std::string &name) {
  NodeProto concat = MakeNode("Concat", {lhs, rhs}, {output}, "", name.c_str());
  AddAttribute<int64_t>(concat, "axis", 0);
  return concat;
}

} // namespace

std::set<std::string> SliceEliminationPattern::FastOpType() const { return {"Slice"}; }

core::builder::MatchResult SliceEliminationPattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Slice") || candidate.input_size() < 1 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a unary-output default-domain Slice");
  }

  const int opset = graph.Builder().OpsetVersion("");
  if (opset == core::shapes::kUnknownOpsetVersion || opset < 1) {
    return NoMatch(candidate, "the default-domain opset is unknown or invalid");
  }

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;
  if (opset < 10) {
    if (!GetAttributeInts(candidate, "starts", starts) ||
        !GetAttributeInts(candidate, "ends", ends)) {
      return NoMatch(candidate, "the attribute-form Slice needs starts and ends");
    }
    if (GetAttributeInts(candidate, "axes", axes) && axes.size() != starts.size()) {
      return NoMatch(candidate, "the Slice axes size differs from starts");
    }
  } else {
    if (!ReadConstantIntegers(graph, candidate, 1, starts) ||
        !ReadConstantIntegers(graph, candidate, 2, ends)) {
      return NoMatch(candidate, "the Slice starts and ends must be constant integers");
    }
    if (candidate.input_size() > 3 && !candidate.input()[3].value().empty()) {
      if (!ReadConstantIntegers(graph, candidate, 3, axes) || axes.size() != starts.size()) {
        return NoMatch(candidate, "the Slice axes must be constant and match starts");
      }
      if (std::set<int64_t>(axes.begin(), axes.end()).size() != axes.size()) {
        return NoMatch(candidate, "the Slice axes must be unique");
      }
    }
    if (candidate.input_size() > 4 && !candidate.input()[4].value().empty()) {
      std::vector<int64_t> steps;
      if (!ReadConstantIntegers(graph, candidate, 4, steps) || steps.size() != starts.size()) {
        return NoMatch(candidate, "the Slice steps must be constant and match starts");
      }
      for (int64_t step : steps) {
        if (step != 1) {
          return NoMatch(candidate, "the Slice steps are not all one");
        }
      }
    }
  }

  if (starts.empty() || starts.size() != ends.size()) {
    return NoMatch(candidate, "the Slice starts and ends must have the same non-zero size");
  }
  const core::symbolic::SymShape *shape = nullptr;
  if (graph.HasShape(candidate.input()[0].value())) {
    shape = &graph.GetShape(candidate.input()[0].value()).Shape();
  }
  for (std::size_t i = 0; i < starts.size(); ++i) {
    if (starts[i] != 0) {
      return NoMatch(candidate, "the Slice does not select the full range");
    }
    if (ends[i] == std::numeric_limits<int64_t>::max()) {
      continue;
    }
    int64_t axis = axes.empty() ? static_cast<int64_t>(i) : axes[i];
    if (shape == nullptr) {
      return NoMatch(candidate, "a finite Slice end requires a known input shape");
    }
    const int64_t rank = static_cast<int64_t>(shape->Rank());
    axis = axis < 0 ? axis + rank : axis;
    if (axis < 0 || axis >= rank || !(*shape)[static_cast<std::size_t>(axis)].IsInt() ||
        ends[i] < (*shape)[static_cast<std::size_t>(axis)].AsInt()) {
      return NoMatch(candidate, "the Slice does not select the full range");
    }
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SliceEliminationPattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("SliceEliminationPattern::Apply expects one Slice node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SliceEliminationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &slice = *nodes[0];
  NodeProto replacement =
      MakeNode("Identity", {slice.input()[0].value()}, {slice.output()[0].value()}, "",
               ("SliceEliminationPattern--" + slice.name().value()).c_str());
  if (slice.has_doc_string()) {
    replacement.set_doc_string(slice.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> SliceSlicePattern::FastOpType() const { return {"Slice"}; }

core::builder::MatchResult SliceSlicePattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Slice")) {
    return NoMatch(candidate, "candidate is not a default-domain Slice");
  }
  const std::string &data = candidate.input()[0].value();
  const NodeProto *before = graph.NodeBefore(data);
  if (before == nullptr || !IsDefaultOp(*before, "Slice")) {
    return NoMatch(candidate, "the sliced value is not produced by a Slice");
  }
  if (graph.IsUsedMoreThanOnce(data)) {
    return NoMatch(candidate, "the intermediate slice is consumed more than once");
  }
  std::vector<int64_t> axes1;
  std::vector<int64_t> axes2;
  if (!ReadAxes(graph, *before, axes1) || !ReadAxes(graph, candidate, axes2)) {
    return NoMatch(candidate, "both slices must provide a constant axes input");
  }
  const std::set<int64_t> set1(axes1.begin(), axes1.end());
  for (int64_t axis : axes2) {
    if (set1.count(axis) != 0) {
      return NoMatch(candidate, "the two slices share an axis");
    }
  }
  return core::builder::MatchResult{this, {before, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SliceSlicePattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SliceSlicePattern::Apply expects two Slice nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SliceSlicePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &before = *nodes[0];
  const NodeProto &node = *nodes[1];

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SliceSlicePattern--" + node.name().value();

  const std::string new_start = builder.UniqueName(name + "_start");
  const std::string new_end = builder.UniqueName(name + "_end");
  const std::string new_axis = builder.UniqueName(name + "_axis");

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      ConcatAxis0(before.input()[1].value(), node.input()[1].value(), new_start, name + "-start"));
  replacements.push_back(
      ConcatAxis0(before.input()[2].value(), node.input()[2].value(), new_end, name + "-end"));
  replacements.push_back(
      ConcatAxis0(before.input()[3].value(), node.input()[3].value(), new_axis, name + "-axis"));

  std::vector<std::string> inputs{before.input()[0].value(), new_start, new_end, new_axis};

  const bool before_step = before.input_size() > 4;
  const bool node_step = node.input_size() > 4;
  if (before_step || node_step) {
    const std::string new_step = builder.UniqueName(name + "_step");
    std::string lhs;
    std::string rhs;
    if (before_step && node_step) {
      lhs = before.input()[4].value();
      rhs = node.input()[4].value();
    } else if (node_step) {
      // ``before`` has no steps: replace it with a vector of ones sized like its axes.
      const TensorProto *axes = graph.GetComputedConstant(before.input()[3].value());
      std::vector<int64_t> axes_values;
      if (axes == nullptr || !ReadIntegerValues(*axes, axes_values)) {
        throw BuilderError("SliceSlicePattern::Apply could not read the first slice axes.");
      }
      const std::string ones = FreeInitializerName(builder, name + "_ones");
      builder.MakeInitializer(
          MakeInitializerShape(ones.c_str(), std::vector<int64_t>(axes_values.size(), 1)));
      lhs = ones;
      rhs = node.input()[4].value();
    } else {
      // ``node`` has no steps: replace it with a vector of ones sized like its axes.
      const TensorProto *axes = graph.GetComputedConstant(node.input()[3].value());
      std::vector<int64_t> axes_values;
      if (axes == nullptr || !ReadIntegerValues(*axes, axes_values)) {
        throw BuilderError("SliceSlicePattern::Apply could not read the second slice axes.");
      }
      const std::string ones = FreeInitializerName(builder, name + "_ones");
      builder.MakeInitializer(
          MakeInitializerShape(ones.c_str(), std::vector<int64_t>(axes_values.size(), 1)));
      lhs = before.input()[4].value();
      rhs = ones;
    }
    replacements.push_back(ConcatAxis0(lhs, rhs, new_step, name + "-step"));
    inputs.push_back(new_step);
  }

  replacements.push_back(MakeNode("Slice", inputs, {node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
