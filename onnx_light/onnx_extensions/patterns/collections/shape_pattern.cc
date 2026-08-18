// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/shape_pattern.h"

#include <algorithm>
#include <cstdint>
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

// Returns the integer ``name`` attribute of ``node`` in ``out`` when present.
bool TryGetAttr(const NodeProto &node, const char *name, int64_t &out) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return false;
  }
  out = attr->i();
  return true;
}

// Normalises a possibly-negative dimension index and clamps it to ``[0, rank]``.
int64_t ClampDim(int64_t value, int64_t rank) {
  if (value < 0) {
    value += rank;
  }
  return std::max<int64_t>(0, std::min<int64_t>(value, rank));
}

bool HasRank(core::builder::GraphGraph &graph, const std::string &name) {
  return graph.HasShape(name);
}

int64_t GetRank(core::builder::GraphGraph &graph, const std::string &name) {
  return static_cast<int64_t>(graph.GetShape(name).Shape().Rank());
}

// Resolves the effective ``[start, end)`` range covered by a ``Shape`` node.
//
// Reads the optional ``start`` / ``end`` attributes (defaulting to ``0`` and
// ``rank``), normalises negative values against ``rank``, and clamps the result
// to ``[0, rank]``.
void ResolveShapeStartEnd(const NodeProto &shape, int64_t rank, int64_t &start, int64_t &end) {
  int64_t raw_start = 0;
  int64_t raw_end = rank;
  TryGetAttr(shape, "start", raw_start);
  TryGetAttr(shape, "end", raw_end);
  start = ClampDim(raw_start, rank);
  end = ClampDim(raw_end, rank);
}

} // namespace

std::set<std::string> GatherShapePattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult GatherShapePattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gather")) {
    return NoMatch(candidate, "candidate is not a default-domain Gather");
  }
  if (GetAttributeOr<int64_t>(candidate, "axis", 0) != 0) {
    return NoMatch(candidate, "the Gather does not operate on axis 0");
  }
  const std::string &index_name = candidate.input()[1].value();
  if (!graph.IsConstant(index_name)) {
    return NoMatch(candidate, "the Gather index is not constant");
  }
  const TensorProto *indices = graph.GetComputedConstant(index_name);
  if (indices == nullptr ||
      static_cast<TensorProto::DataType>(indices->data_type()) != TensorProto::DataType::INT64) {
    return NoMatch(candidate, "the Gather index is not a materialised int64 tensor");
  }
  const int rank_index = indices->dims_size();
  if (rank_index > 1) {
    return NoMatch(candidate, "the Gather index is not a scalar or a vector");
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(*indices, values) || values.empty()) {
    return NoMatch(candidate, "the Gather index could not be read");
  }
  if (rank_index == 1 && values.size() > 1) {
    for (std::size_t i = 1; i < values.size(); ++i) {
      if (values[i] != values[i - 1] + 1) {
        return NoMatch(candidate, "the Gather index is not a contiguous ascending range");
      }
    }
  }
  const NodeProto *shape = graph.NodeBefore(candidate.input()[0].value());
  if (shape == nullptr || !IsDefaultOp(*shape, "Shape")) {
    return NoMatch(candidate, "the gathered value is not produced by a Shape");
  }
  int64_t shape_start = 0;
  int64_t shape_end = 0;
  const bool has_start = TryGetAttr(*shape, "start", shape_start);
  const bool has_end = TryGetAttr(*shape, "end", shape_end);
  const bool negative_attr = (has_start && shape_start < 0) || (has_end && shape_end < 0);
  const bool negative_index = values.front() < 0;
  if ((negative_attr || negative_index) && !HasRank(graph, shape->input()[0].value())) {
    return NoMatch(candidate, "the shape origin rank is required to normalise negative bounds");
  }
  return core::builder::MatchResult{this, {shape, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
GatherShapePattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("GatherShapePattern::Apply expects a Shape and a Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GatherShapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &shape = *nodes[0];
  const NodeProto &gather = *nodes[1];

  const TensorProto *indices = graph.GetComputedConstant(gather.input()[1].value());
  std::vector<int64_t> values;
  if (indices == nullptr || !ReadIntegerValues(*indices, values) || values.empty()) {
    throw BuilderError("GatherShapePattern::Apply could not read the Gather index.");
  }
  const bool is_scalar = indices->dims_size() == 0;
  const std::string &x_name = shape.input()[0].value();

  int64_t shape_start = 0;
  int64_t shape_end = 0;
  const bool has_start = TryGetAttr(shape, "start", shape_start);
  const bool has_end = TryGetAttr(shape, "end", shape_end);

  int64_t s0 = has_start ? shape_start : 0;
  int64_t length = -1;
  const bool has_rank = HasRank(graph, x_name);
  if (has_rank) {
    const int64_t rank = GetRank(graph, x_name);
    const int64_t e0 = ClampDim(has_end ? shape_end : rank, rank);
    s0 = ClampDim(s0, rank);
    length = e0 - s0;
  } else if (has_end) {
    length = shape_end - s0;
  }

  int64_t gather_first = values.front();
  int64_t gather_last = values.back();
  if (gather_first < 0 && length >= 0) {
    gather_first += length;
  }
  if (gather_last < 0 && length >= 0) {
    gather_last += length;
  }
  const int64_t new_start = s0 + gather_first;
  const int64_t new_end = s0 + gather_last + 1;

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "GatherShapePattern--" + gather.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(shape.output()[0].value())) {
    replacements.add() = shape;
  }

  if (is_scalar) {
    const std::string shape_out = builder.UniqueName(name + "_1d");
    NodeProto new_shape = MakeNode("Shape", {x_name}, {shape_out}, "", name.c_str());
    AddAttribute<int64_t>(new_shape, "start", new_start);
    AddAttribute<int64_t>(new_shape, "end", new_end);
    replacements.push_back(new_shape);

    const std::string axes_name = FreeInitializerName(builder, name + "_axes");
    builder.MakeInitializer(MakeInitializerShape(axes_name.c_str(), {0}));
    replacements.push_back(MakeNode("Squeeze", {shape_out, axes_name}, {gather.output()[0].value()},
                                    "", (name + "--squeeze").c_str()));
    return replacements;
  }

  NodeProto new_shape = MakeNode("Shape", {x_name}, {gather.output()[0].value()}, "", name.c_str());
  AddAttribute<int64_t>(new_shape, "start", new_start);
  AddAttribute<int64_t>(new_shape, "end", new_end);
  replacements.push_back(new_shape);
  return replacements;
}

std::set<std::string> ShapeTransposePattern::FastOpType() const { return {"Shape"}; }

core::builder::MatchResult ShapeTransposePattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Shape")) {
    return NoMatch(candidate, "candidate is not a default-domain Shape");
  }
  const NodeProto *transpose = graph.NodeBefore(candidate.input()[0].value());
  if (transpose == nullptr || !IsDefaultOp(*transpose, "Transpose")) {
    return NoMatch(candidate, "the shaped value is not produced by a Transpose");
  }
  std::vector<int64_t> perm;
  if (!GetAttributeInts(*transpose, "perm", perm)) {
    return NoMatch(candidate, "the Transpose does not define a perm attribute");
  }
  int64_t start = 0;
  int64_t end = 0;
  ResolveShapeStartEnd(candidate, static_cast<int64_t>(perm.size()), start, end);
  if (start >= end) {
    return NoMatch(candidate, "the Shape start/end range is empty");
  }
  return core::builder::MatchResult{this, {transpose, &candidate}, transpose};
}

utils::RepeatedProtoField<NodeProto>
ShapeTransposePattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ShapeTransposePattern::Apply expects a Transpose and a Shape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ShapeTransposePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &transpose = *nodes[0];
  const NodeProto &shape = *nodes[1];

  std::vector<int64_t> perm;
  if (!GetAttributeInts(transpose, "perm", perm)) {
    throw BuilderError("ShapeTransposePattern::Apply could not read the perm attribute.");
  }
  int64_t start = 0;
  int64_t end = 0;
  ResolveShapeStartEnd(shape, static_cast<int64_t>(perm.size()), start, end);
  const std::vector<int64_t> perm_subset(perm.begin() + start, perm.begin() + end);

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ShapeTransposePattern--" + shape.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(transpose.output()[0].value())) {
    replacements.add() = transpose;
  }

  const std::string shape_out = builder.UniqueName(name + "_sx");
  replacements.push_back(MakeNode("Shape", {transpose.input()[0].value()}, {shape_out}, "",
                                  (name + "--shape").c_str()));

  const std::string perm_name = FreeInitializerName(builder, name + "_perm");
  builder.MakeInitializer(MakeInitializerShape(perm_name.c_str(), perm_subset));
  NodeProto gather =
      MakeNode("Gather", {shape_out, perm_name}, {shape.output()[0].value()}, "", name.c_str());
  AddAxisAttribute(gather, 0);
  replacements.push_back(gather);
  return replacements;
}

std::set<std::string> UnsqueezeShapePattern::FastOpType() const { return {"Unsqueeze"}; }

core::builder::MatchResult UnsqueezeShapePattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Unsqueeze")) {
    return NoMatch(candidate, "candidate is not a default-domain Unsqueeze");
  }
  if (candidate.input_size() < 2 || !graph.IsConstant(candidate.input()[1].value())) {
    return NoMatch(candidate, "the Unsqueeze axes are not a constant second input");
  }
  const TensorProto *axes = graph.GetComputedConstant(candidate.input()[1].value());
  std::vector<int64_t> axes_values;
  if (axes == nullptr || !ReadIntegerValues(*axes, axes_values) || axes_values.empty()) {
    return NoMatch(candidate, "the Unsqueeze axes could not be read");
  }
  const std::string &x_name = candidate.input()[0].value();
  if (!HasRank(graph, x_name)) {
    return NoMatch(candidate, "the Unsqueeze input rank is required to normalise axes");
  }
  const NodeProto *shape = nullptr;
  for (const NodeProto *consumer : graph.NextNodes(candidate.output()[0].value())) {
    if (consumer != nullptr && IsDefaultOp(*consumer, "Shape")) {
      shape = consumer;
      break;
    }
  }
  if (shape == nullptr) {
    return NoMatch(candidate, "the Unsqueeze output is not consumed by a Shape");
  }
  const int64_t output_rank = GetRank(graph, x_name) + static_cast<int64_t>(axes_values.size());
  int64_t start = 0;
  int64_t end = 0;
  ResolveShapeStartEnd(*shape, output_rank, start, end);
  if (start >= end) {
    return NoMatch(candidate, "the Shape start/end range is empty");
  }
  return core::builder::MatchResult{this, {&candidate, shape}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
UnsqueezeShapePattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("UnsqueezeShapePattern::Apply expects an Unsqueeze and a Shape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("UnsqueezeShapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &unsqueeze = *nodes[0];
  const NodeProto &shape = *nodes[1];

  const std::string &x_name = unsqueeze.input()[0].value();
  const TensorProto *axes = graph.GetComputedConstant(unsqueeze.input()[1].value());
  std::vector<int64_t> axes_values;
  if (axes == nullptr || !ReadIntegerValues(*axes, axes_values) || axes_values.empty()) {
    throw BuilderError("UnsqueezeShapePattern::Apply could not read the Unsqueeze axes.");
  }
  const int64_t input_rank = GetRank(graph, x_name);
  const int64_t output_rank = input_rank + static_cast<int64_t>(axes_values.size());
  std::vector<int64_t> sorted_axes;
  sorted_axes.reserve(axes_values.size());
  for (const int64_t axis : axes_values) {
    sorted_axes.push_back(((axis % output_rank) + output_rank) % output_rank);
  }
  std::sort(sorted_axes.begin(), sorted_axes.end());

  int64_t shape_start = 0;
  int64_t shape_end = 0;
  ResolveShapeStartEnd(shape, output_rank, shape_start, shape_end);

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "UnsqueezeShapePattern--" + shape.name().value();

  std::vector<std::string> concat_inputs;
  utils::RepeatedProtoField<NodeProto> extra_nodes;
  int64_t x_cursor = 0;   // next un-consumed position in X's dimension space
  int64_t out_cursor = 0; // next un-consumed position in the unsqueezed output

  const auto emit_shape_segment = [&](int64_t seg_out_start, int64_t seg_out_end) {
    seg_out_start = std::max(seg_out_start, shape_start);
    seg_out_end = std::min(seg_out_end, shape_end);
    if (seg_out_start >= seg_out_end) {
      return;
    }
    const int64_t x_seg_start = x_cursor + (seg_out_start - out_cursor);
    const int64_t x_seg_end = x_cursor + (seg_out_end - out_cursor);
    const std::string seg_name = builder.UniqueName(name + "_s" + std::to_string(x_seg_start));
    NodeProto seg = MakeNode("Shape", {x_name}, {seg_name}, "",
                             (name + "--shape" + std::to_string(x_seg_start)).c_str());
    AddAttribute<int64_t>(seg, "start", x_seg_start);
    AddAttribute<int64_t>(seg, "end", x_seg_end);
    extra_nodes.push_back(seg);
    concat_inputs.push_back(seg_name);
  };

  for (const int64_t axis : sorted_axes) {
    const int64_t run_len = axis - out_cursor;
    if (run_len > 0) {
      emit_shape_segment(out_cursor, axis);
      x_cursor += run_len;
    }
    if (shape_start <= axis && axis < shape_end) {
      const std::string one_name = FreeInitializerName(builder, name + "_one");
      builder.MakeInitializer(MakeInitializerShape(one_name.c_str(), {1}));
      concat_inputs.push_back(one_name);
    }
    out_cursor = axis + 1;
  }
  if (x_cursor < input_rank) {
    emit_shape_segment(out_cursor, output_rank);
  }

  NodeProto concat =
      MakeNode("Concat", concat_inputs, {shape.output()[0].value()}, "", name.c_str());
  AddAxisAttribute(concat, 0);

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(unsqueeze.output()[0].value())) {
    replacements.add() = unsqueeze;
  }
  for (const NodeProto &node : extra_nodes) {
    replacements.push_back(node);
  }
  replacements.push_back(concat);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
