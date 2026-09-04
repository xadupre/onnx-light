// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/concat_pattern.h"

#include <array>
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

using collections::GetAxis;
using collections::IsDefaultOp;
using core::builder::BuilderError;

// Shape-preserving unary operators that may be pushed through a Concat(x, x).
const std::set<std::string> &UnaryLikeOpTypes() {
  static const std::set<std::string> types = {"Abs",
                                              "Acos",
                                              "Acosh",
                                              "Asin",
                                              "Asinh",
                                              "Atan",
                                              "Atanh",
                                              "BitShift",
                                              "BitwiseNot",
                                              "Cast",
                                              "CastLike",
                                              "Ceil",
                                              "Celu",
                                              "Clip",
                                              "Cos",
                                              "Cosh",
                                              "CumSum",
                                              "DequantizeLinear",
                                              "DynamicQuantizeLinear",
                                              "Elu",
                                              "Erf",
                                              "Exp",
                                              "Floor",
                                              "HardSigmoid",
                                              "HardSwish",
                                              "IsInf",
                                              "LeakyRelu",
                                              "Log",
                                              "LogSoftmax",
                                              "LpNormalization",
                                              "LRN",
                                              "MeanVarianceNormalization",
                                              "Mish",
                                              "Neg",
                                              "Not",
                                              "PRelu",
                                              "QuantizeLinear",
                                              "Reciprocal",
                                              "Relu",
                                              "Round",
                                              "Selu",
                                              "Shrink",
                                              "Sigmoid",
                                              "Sign",
                                              "Sin",
                                              "Sinh",
                                              "Softmax",
                                              "Softplus",
                                              "Softsign",
                                              "Sqrt",
                                              "Tan",
                                              "Tanh",
                                              "ThresholdedRelu",
                                              "ThresholdRelu",
                                              "Trilu",
                                              "Trunc"};
  return types;
}

// Returns the indices of ``concat`` inputs that are empty along ``axis``.
std::vector<int> EmptyConcatInputs(core::builder::GraphGraph &graph, const NodeProto &concat) {
  const int64_t axis = GetAxis(concat, 0);
  std::vector<int> empty;
  for (int i = 0; i < concat.input_size(); ++i) {
    const std::string &name = concat.input()[i].value();
    if (!graph.HasShape(name)) {
      continue;
    }
    const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
    const int64_t rank = static_cast<int64_t>(shape.Rank());
    int64_t normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      continue;
    }
    const auto &dim = shape[static_cast<std::size_t>(normalized)];
    if (dim.IsInt() && dim.AsInt() == 0) {
      empty.push_back(i);
    }
  }
  return empty;
}

// Locates which ``concat`` input holds element ``idx`` along axis 0. On success
// returns ``true`` and fills the input name, its local index and its size.
bool ConcatInputBucket(core::builder::GraphGraph &graph, const NodeProto &concat, int64_t idx,
                       std::string &input_name, int64_t &local_index, int64_t &input_size) {
  int64_t offset = 0;
  for (const auto &input : concat.input()) {
    const std::string &name = input.value();
    if (!graph.HasShape(name)) {
      return false;
    }
    const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
    if (shape.Rank() != 1 || !shape[0].IsInt()) {
      return false;
    }
    const int64_t n = shape[0].AsInt();
    if (offset + n > idx) {
      input_name = name;
      local_index = idx - offset;
      input_size = n;
      return true;
    }
    offset += n;
  }
  return false;
}

// Returns true when ``unary`` may be pushed ahead of ``concat`` (see pattern).
bool ValidUnaryConsumer(core::builder::GraphGraph &graph, const NodeProto &concat,
                        const NodeProto &unary) {
  const std::string &op_type = unary.op_type().value();
  if (NormaliseDomain(unary.domain().value()) != kDefaultOnnxDomain) {
    return false;
  }
  if (UnaryLikeOpTypes().count(op_type) != 0) {
    return true;
  }
  if (op_type == "Unsqueeze" && unary.input_size() >= 2 &&
      graph.IsConstantScalar(unary.input()[1].value())) {
    int64_t cst = 0;
    if (!collections::ReadScalarInt(*graph.GetComputedConstant(unary.input()[1].value()), cst)) {
      return false;
    }
    const int64_t axis = GetAxis(concat, 0);
    if (axis == -1 && cst != -1 && graph.HasShape(unary.input()[0].value()) &&
        cst < static_cast<int64_t>(graph.GetShape(unary.input()[0].value()).Shape().Rank())) {
      return true;
    }
  }
  static const std::set<std::string> binary = {"Mul", "Add", "Div", "Sub"};
  if (binary.count(op_type) != 0 && unary.input_size() >= 2 &&
      graph.IsConstantScalar(unary.input()[1].value())) {
    return true;
  }
  return false;
}

bool ReadConstantInts(core::builder::GraphGraph &graph, const std::string &name,
                      std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

bool ReadSimpleSlice(core::builder::GraphGraph &graph, const NodeProto &slice, int64_t &start,
                     int64_t &end, int64_t &axis, int64_t &step) {
  if (!IsDefaultOp(slice, "Slice") || slice.input_size() < 3) {
    return false;
  }
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!ReadConstantInts(graph, slice.input()[1].value(), starts) ||
      !ReadConstantInts(graph, slice.input()[2].value(), ends) || starts.size() != 1 ||
      ends.size() != 1) {
    return false;
  }
  start = starts[0];
  end = ends[0];
  axis = 0;
  step = 1;
  if (slice.input_size() >= 4 && !slice.input()[3].value().empty()) {
    std::vector<int64_t> axes;
    if (!ReadConstantInts(graph, slice.input()[3].value(), axes) || axes.size() != 1) {
      return false;
    }
    axis = axes[0];
  }
  if (slice.input_size() >= 5 && !slice.input()[4].value().empty()) {
    std::vector<int64_t> steps;
    if (!ReadConstantInts(graph, slice.input()[4].value(), steps) || steps.size() != 1) {
      return false;
    }
    step = steps[0];
  }
  return true;
}

struct SpatialSlice {
  std::array<int64_t, 4> starts{0, 0, 0, 0};
  std::array<int64_t, 4> ends{
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};
  std::array<int64_t, 4> steps{1, 1, 1, 1};
};

bool ReadSpatialSlice(core::builder::GraphGraph &graph, const NodeProto &slice,
                      SpatialSlice &params) {
  if (!IsDefaultOp(slice, "Slice") || slice.input_size() < 3) {
    return false;
  }
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!ReadConstantInts(graph, slice.input()[1].value(), starts) ||
      !ReadConstantInts(graph, slice.input()[2].value(), ends) || starts.empty() ||
      starts.size() != ends.size()) {
    return false;
  }
  std::vector<int64_t> axes(starts.size());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    axes[i] = static_cast<int64_t>(i);
  }
  if (slice.input_size() >= 4 && !slice.input()[3].value().empty() &&
      (!ReadConstantInts(graph, slice.input()[3].value(), axes) || axes.size() != starts.size())) {
    return false;
  }
  std::vector<int64_t> steps(starts.size(), 1);
  if (slice.input_size() >= 5 && !slice.input()[4].value().empty() &&
      (!ReadConstantInts(graph, slice.input()[4].value(), steps) ||
       steps.size() != starts.size())) {
    return false;
  }
  std::array<bool, 4> seen{false, false, false, false};
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const int64_t axis = axes[i] < 0 ? axes[i] + 4 : axes[i];
    if (axis < 0 || axis >= 4 || seen[static_cast<std::size_t>(axis)]) {
      return false;
    }
    seen[static_cast<std::size_t>(axis)] = true;
    params.starts[static_cast<std::size_t>(axis)] = starts[i];
    params.ends[static_cast<std::size_t>(axis)] = ends[i];
    params.steps[static_cast<std::size_t>(axis)] = steps[i];
  }
  return true;
}

bool IsFullSliceEnd(const core::symbolic::SymShape &shape, std::size_t axis, int64_t end) {
  return end == std::numeric_limits<int64_t>::max() ||
         (shape[axis].IsInt() && end >= shape[axis].AsInt());
}

} // namespace

std::set<std::string> ConcatEmptyPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult ConcatEmptyPattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Concat")) {
    return NoMatch(candidate, "candidate is not a default-domain Concat");
  }
  const std::vector<int> empty = EmptyConcatInputs(graph, candidate);
  if (empty.empty()) {
    return NoMatch(candidate, "no Concat input is empty along the axis");
  }
  if (static_cast<int>(empty.size()) >= candidate.input_size()) {
    return NoMatch(candidate, "every Concat input is empty along the axis");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConcatEmptyPattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ConcatEmptyPattern::Apply expects one Concat node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatEmptyPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];
  const std::vector<int> empty = EmptyConcatInputs(graph, node);
  std::vector<bool> drop(node.input_size(), false);
  for (int i : empty) {
    drop[i] = true;
  }
  std::vector<std::string> kept;
  for (int i = 0; i < node.input_size(); ++i) {
    if (!drop[i]) {
      kept.push_back(node.input()[i].value());
    }
  }

  const std::string name = "ConcatEmptyPattern--" + node.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (kept.size() == 1) {
    replacements.push_back(
        MakeNode("Identity", kept, {node.output()[0].value()}, "", name.c_str()));
    return replacements;
  }
  NodeProto concat = MakeNode("Concat", kept, {node.output()[0].value()}, "", name.c_str());
  for (const auto &attribute : node.attribute()) {
    *concat.add_attribute() = attribute;
  }
  replacements.push_back(concat);
  return replacements;
}

std::set<std::string> ConcatSliceEliminationPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult ConcatSliceEliminationPattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Concat") || candidate.input_size() < 2) {
    return NoMatch(candidate, "candidate is not a multi-input default-domain Concat");
  }
  const std::string &output = candidate.output()[0].value();
  if (graph.IsOutput(output) || graph.IsUsedBySubgraph(output)) {
    return NoMatch(candidate, "the Concat output escapes the matched slices");
  }
  const std::vector<const NodeProto *> &consumers = graph.NextNodes(output);
  if (consumers.size() != static_cast<std::size_t>(candidate.input_size())) {
    return NoMatch(candidate, "the number of Slice consumers differs from the Concat inputs");
  }

  int64_t rank = -1;
  int64_t axis = GetAxis(candidate, 0);
  std::vector<int64_t> boundaries{0};
  for (const auto &input : candidate.input()) {
    if (!graph.HasShape(input.value())) {
      return NoMatch(candidate, "a Concat input has no known shape");
    }
    const core::symbolic::SymShape &shape = graph.GetShape(input.value()).Shape();
    if (rank < 0) {
      rank = static_cast<int64_t>(shape.Rank());
      axis = axis < 0 ? axis + rank : axis;
    }
    if (static_cast<int64_t>(shape.Rank()) != rank || axis < 0 || axis >= rank ||
        !shape[static_cast<std::size_t>(axis)].IsInt()) {
      return NoMatch(candidate, "the Concat axis dimensions are not statically known");
    }
    boundaries.push_back(boundaries.back() + shape[static_cast<std::size_t>(axis)].AsInt());
  }

  std::vector<const NodeProto *> ordered(static_cast<std::size_t>(candidate.input_size()), nullptr);
  for (const NodeProto *slice : consumers) {
    if (slice->input_size() == 0 || slice->input()[0].value() != output) {
      return NoMatch(candidate, "the Concat output is not the Slice data input");
    }
    int64_t start = 0;
    int64_t end = 0;
    int64_t slice_axis = 0;
    int64_t step = 1;
    if (!ReadSimpleSlice(graph, *slice, start, end, slice_axis, step)) {
      return NoMatch(candidate, "a consumer is not a constant one-axis Slice");
    }
    slice_axis = slice_axis < 0 ? slice_axis + rank : slice_axis;
    if (slice_axis != axis || step != 1) {
      return NoMatch(candidate, "a Slice uses a different axis or non-unit step");
    }
    std::size_t slot = ordered.size();
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      if (start == boundaries[i] &&
          (end == boundaries[i + 1] ||
           (i + 1 == ordered.size() && end == std::numeric_limits<int64_t>::max()))) {
        slot = i;
        break;
      }
    }
    if (slot == ordered.size() || ordered[slot] != nullptr) {
      return NoMatch(candidate, "the Slice ranges do not recover distinct Concat inputs");
    }
    ordered[slot] = slice;
  }
  std::vector<const NodeProto *> nodes{&candidate};
  nodes.insert(nodes.end(), ordered.begin(), ordered.end());
  return core::builder::MatchResult{this, nodes, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConcatSliceEliminationPattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() < 3 || nodes[0] == nullptr) {
    throw BuilderError("ConcatSliceEliminationPattern::Apply expects a Concat and its slices.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ConcatSliceEliminationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  utils::RepeatedProtoField<NodeProto> replacements;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const std::string name = "ConcatSliceEliminationPattern--" + nodes[i]->name().value();
    replacements.push_back(MakeNode("Identity", {concat.input()[static_cast<int>(i - 1)].value()},
                                    {nodes[i]->output()[0].value()}, "", name.c_str()));
  }
  return replacements;
}

std::set<std::string> SliceConcatToSpaceToDepthPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult
SliceConcatToSpaceToDepthPattern::Match(core::builder::GraphGraph &graph,
                                        const NodeProto &candidate) const {
  int64_t concat_axis = GetAxis(candidate, 0);
  concat_axis = concat_axis < 0 ? concat_axis + 4 : concat_axis;
  if (!IsDefaultOp(candidate, "Concat") || candidate.input_size() != 4 || concat_axis != 1) {
    return NoMatch(candidate, "candidate is not a four-input channel-axis Concat");
  }
  static constexpr std::array<std::array<int64_t, 2>, 4> phases{
      {{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}};
  std::string data;
  std::vector<const NodeProto *> nodes;
  nodes.reserve(5);
  for (int i = 0; i < 4; ++i) {
    const NodeProto *slice = graph.NodeBefore(candidate.input()[i].value());
    if (slice == nullptr || graph.IsUsedMoreThanOnce(slice->output()[0].value()) ||
        graph.IsOutput(slice->output()[0].value()) ||
        graph.IsUsedBySubgraph(slice->output()[0].value())) {
      return NoMatch(candidate, "a Concat input is not produced by an exclusive Slice");
    }
    if (i == 0) {
      data = slice->input()[0].value();
      if (!graph.HasShape(data) || graph.GetShape(data).Shape().Rank() != 4) {
        return NoMatch(candidate, "the common Slice input is not known to be rank 4");
      }
    } else if (slice->input()[0].value() != data) {
      return NoMatch(candidate, "the Slice nodes do not share one input");
    }
    const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
    if (!shape[2].IsInt() || !shape[3].IsInt() || shape[2].AsInt() % 2 != 0 ||
        shape[3].AsInt() % 2 != 0) {
      return NoMatch(candidate, "the spatial dimensions are not static even multiples of two");
    }
    SpatialSlice params;
    if (!ReadSpatialSlice(graph, *slice, params) || params.starts[0] != 0 ||
        params.starts[1] != 0 || params.steps[0] != 1 || params.steps[1] != 1 ||
        params.steps[2] != 2 || params.steps[3] != 2 ||
        params.starts[2] != phases[static_cast<std::size_t>(i)][0] ||
        params.starts[3] != phases[static_cast<std::size_t>(i)][1]) {
      return NoMatch(candidate, "a Slice does not select the expected canonical phase");
    }
    for (std::size_t axis = 0; axis < 4; ++axis) {
      if (!IsFullSliceEnd(shape, axis, params.ends[axis])) {
        return NoMatch(candidate, "a Slice crops rather than spanning the full input");
      }
    }
    nodes.push_back(slice);
  }
  nodes.push_back(&candidate);
  return core::builder::MatchResult{this, nodes, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SliceConcatToSpaceToDepthPattern::Apply(core::builder::GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 5 || nodes[0] == nullptr || nodes[4] == nullptr) {
    throw BuilderError("SliceConcatToSpaceToDepthPattern::Apply expects four slices and a Concat.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[4]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SliceConcatToSpaceToDepthPattern::Apply received an unsafe or inconsistent match.");
  }
  const std::string name = "SliceConcatToSpaceToDepthPattern--" + nodes[4]->name().value();
  NodeProto replacement = MakeNode("SpaceToDepth", {nodes[0]->input()[0].value()},
                                   {nodes[4]->output()[0].value()}, "", name.c_str());
  AddAttribute<int64_t>(replacement, "blocksize", 2);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(replacement);
  return replacements;
}

std::set<std::string> ConcatGatherPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult ConcatGatherPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gather")) {
    return NoMatch(candidate, "candidate is not a default-domain Gather");
  }
  const std::string &index_name = candidate.input()[1].value();
  if (!graph.IsConstant(index_name)) {
    return NoMatch(candidate, "the Gather index is not constant");
  }
  const TensorProto *index = graph.GetComputedConstant(index_name);
  if (index == nullptr ||
      static_cast<TensorProto::DataType>(index->data_type()) != TensorProto::DataType::INT64) {
    return NoMatch(candidate, "the Gather index is not a materialised int64 tensor");
  }
  if (index->dims_size() != 1 || index->dims()[0] != 1) {
    return NoMatch(candidate, "the Gather index is not a single-element vector");
  }
  const NodeProto *concat = graph.NodeBefore(candidate.input()[0].value());
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat")) {
    return NoMatch(candidate, "the gathered value is not produced by a Concat");
  }
  std::vector<int64_t> index_values;
  if (!ReadIntegerValues(*index, index_values) || index_values.empty()) {
    return NoMatch(candidate, "the Gather index could not be read");
  }
  std::string input_name;
  int64_t local_index = 0;
  int64_t input_size = 0;
  if (!ConcatInputBucket(graph, *concat, index_values[0], input_name, local_index, input_size)) {
    return NoMatch(candidate, "the index does not map to a statically-sized Concat input");
  }
  return core::builder::MatchResult{this, {concat, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConcatGatherPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ConcatGatherPattern::Apply expects a Concat and a Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatGatherPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &gather = *nodes[1];

  const TensorProto *index = graph.GetComputedConstant(gather.input()[1].value());
  std::vector<int64_t> index_values;
  if (index == nullptr || !ReadIntegerValues(*index, index_values) || index_values.empty()) {
    throw BuilderError("ConcatGatherPattern::Apply could not read the Gather index.");
  }
  std::string input_name;
  int64_t local_index = 0;
  int64_t input_size = 0;
  if (!ConcatInputBucket(graph, concat, index_values[0], input_name, local_index, input_size)) {
    throw BuilderError("ConcatGatherPattern::Apply could not locate the Concat input.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ConcatGatherPattern--" + gather.name().value();
  const std::string &output = gather.output()[0].value();

  NodeProto replacement;
  if (input_size == 1) {
    replacement = MakeNode("Identity", {input_name}, {output}, "", name.c_str());
  } else {
    const NodeProto *producer = graph.NodeBefore(input_name);
    if (producer != nullptr && IsDefaultOp(*producer, "Shape")) {
      const int64_t start = GetAttributeOr<int64_t>(*producer, "start", 0);
      replacement = MakeNode("Shape", {producer->input()[0].value()}, {output}, "", name.c_str());
      AddAttribute<int64_t>(replacement, "start", start + local_index);
      AddAttribute<int64_t>(replacement, "end", start + local_index + 1);
    } else {
      const std::string index_init = collections::FreeInitializerName(builder, name + "_idx");
      builder.MakeInitializer(MakeInitializerShape(index_init.c_str(), {local_index}));
      replacement = MakeNode("Gather", {input_name, index_init}, {output}, "", name.c_str());
    }
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  replacements.push_back(replacement);
  return replacements;
}

std::set<std::string> ConcatTwiceUnaryPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult ConcatTwiceUnaryPattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 18) {
    return NoMatch(candidate, "the default opset is older than 18");
  }
  if (!IsDefaultOp(candidate, "Concat")) {
    return NoMatch(candidate, "candidate is not a default-domain Concat");
  }
  if (candidate.input_size() != 2 || candidate.input()[0].value() != candidate.input()[1].value()) {
    return NoMatch(candidate, "the Concat does not repeat a single input twice");
  }
  for (const NodeProto *user : graph.NextNodes(candidate.output()[0].value())) {
    if (ValidUnaryConsumer(graph, candidate, *user)) {
      return core::builder::MatchResult{this, {&candidate, user}, &candidate};
    }
  }
  return NoMatch(candidate, "the Concat is not consumed by a shape-preserving unary op");
}

utils::RepeatedProtoField<NodeProto>
ConcatTwiceUnaryPattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ConcatTwiceUnaryPattern::Apply expects a Concat and a unary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatTwiceUnaryPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &unary = *nodes[1];

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string new_name = builder.UniqueName("u" + unary.output()[0].value());
  const std::string name = "ConcatTwiceUnaryPattern--" + unary.name().value();

  std::vector<std::string> unary_inputs{concat.input()[0].value()};
  for (int i = 1; i < unary.input_size(); ++i) {
    unary_inputs.push_back(unary.input()[i].value());
  }
  NodeProto new_unary =
      MakeNode(unary.op_type().value().c_str(), unary_inputs, {new_name}, "", name.c_str());
  for (const auto &attribute : unary.attribute()) {
    *new_unary.add_attribute() = attribute;
  }
  NodeProto new_concat = MakeNode(concat.op_type().value().c_str(), {new_name, new_name},
                                  {unary.output()[0].value()}, "", name.c_str());
  for (const auto &attribute : concat.attribute()) {
    *new_concat.add_attribute() = attribute;
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  replacements.push_back(new_unary);
  replacements.push_back(new_concat);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
