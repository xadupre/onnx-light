// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/gather_pattern.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;

// Returns true when ``node`` is a Gather along axis 0 (default axis).
bool IsAxis0Gather(const NodeProto &node) {
  return IsDefaultOp(node, "Gather") && GetAttributeOr<int64_t>(node, "axis", 0) == 0;
}

// Reads a constant int32/int64 tensor into ``values`` and reports its rank.
bool ReadIntTensor(const TensorProto &tensor, std::vector<int64_t> &values, int &rank) {
  const auto dtype = static_cast<TensorProto::DataType>(tensor.data_type());
  if (dtype != TensorProto::DataType::INT64 && dtype != TensorProto::DataType::INT32) {
    return false;
  }
  if (!ReadIntegerValues(tensor, values)) {
    return false;
  }
  rank = tensor.dims_size();
  return true;
}

// A fully resolved plan for rewriting a Gather with a safely representable
// constant index into a Slice (plus a Squeeze for a scalar index).
struct GatherToSlicePlan {
  int64_t axis = 0;           // Normalized (non-negative when knowable) Gather axis.
  int64_t start = 0;          // First selected index, normalized when negative.
  int64_t step = 1;           // Constant stride between consecutive selected indices.
  int64_t count = 0;          // Number of selected indices (>= 1).
  bool needs_squeeze = false; // True for a scalar (rank-0) index.
};

// Reasons why a candidate Gather cannot be rewritten as a Slice. Kept as an
// enum (rather than a runtime string) so ``Match`` can report each rejection
// through :cpp:func:`PatternOptimization::NoMatch`, which only accepts string
// literals.
enum class GatherToSliceReject {
  kNone,
  kNotGather,
  kNotUnaryGather,
  kIndexNotConstant,
  kIndexNotScalarOrVector,
  kIndexEmpty,
  kNegativeAxisNoRank,
  kAxisOutOfRange,
  kNegativeIndexNoRank,
  kAxisOutOfRangeForIndex,
  kDimNotStatic,
  kIndexOutOfRange,
  kNotAscending,
  kNotConstantStep,
  kNonUnitStepNeedsOpset10,
};

// Analyzes ``candidate`` and fills ``plan`` when it is a Gather whose index is
// a constant scalar, a constant single-element vector, or a constant
// strictly-ascending arithmetic-progression vector, all safely representable
// as a Slice. Returns false and sets ``*reject`` otherwise.
bool AnalyzeGatherToSlice(core::builder::GraphGraph &graph, const NodeProto &candidate,
                          GatherToSlicePlan &plan, GatherToSliceReject *reject) {
  auto fail = [&](GatherToSliceReject why) {
    if (reject != nullptr) {
      *reject = why;
    }
    return false;
  };
  if (!IsDefaultOp(candidate, "Gather")) {
    return fail(GatherToSliceReject::kNotGather);
  }
  if (candidate.input_size() != 2 || candidate.output_size() != 1) {
    return fail(GatherToSliceReject::kNotUnaryGather);
  }
  const std::string &data = candidate.input()[0].value();
  const std::string &index_name = candidate.input()[1].value();
  if (!graph.IsConstant(index_name)) {
    return fail(GatherToSliceReject::kIndexNotConstant);
  }
  const TensorProto *index = graph.GetComputedConstant(index_name);
  std::vector<int64_t> values;
  int rank = 0;
  if (index == nullptr || !ReadIntTensor(*index, values, rank) || rank > 1) {
    return fail(GatherToSliceReject::kIndexNotScalarOrVector);
  }
  if (values.empty()) {
    return fail(GatherToSliceReject::kIndexEmpty);
  }

  int64_t axis = GetAttributeOr<int64_t>(candidate, "axis", 0);
  if (axis < 0) {
    if (!graph.HasShape(data)) {
      return fail(GatherToSliceReject::kNegativeAxisNoRank);
    }
    const int64_t data_rank = static_cast<int64_t>(graph.GetShape(data).Shape().Rank());
    axis += data_rank;
    if (axis < 0 || axis >= data_rank) {
      return fail(GatherToSliceReject::kAxisOutOfRange);
    }
  }

  const bool needs_dim = std::any_of(values.begin(), values.end(), [](int64_t v) { return v < 0; });
  int64_t dim = -1;
  if (needs_dim) {
    if (!graph.HasShape(data)) {
      return fail(GatherToSliceReject::kNegativeIndexNoRank);
    }
    const core::symbolic::SymShape &shape = graph.GetShape(data).Shape();
    const int64_t data_rank = static_cast<int64_t>(shape.Rank());
    if (axis < 0 || axis >= data_rank) {
      return fail(GatherToSliceReject::kAxisOutOfRangeForIndex);
    }
    if (!shape[static_cast<std::size_t>(axis)].IsInt()) {
      return fail(GatherToSliceReject::kDimNotStatic);
    }
    dim = shape[static_cast<std::size_t>(axis)].AsInt();
  }
  std::vector<int64_t> normalized;
  normalized.reserve(values.size());
  for (int64_t value : values) {
    const int64_t resolved = value < 0 ? value + dim : value;
    if (resolved < 0) {
      return fail(GatherToSliceReject::kIndexOutOfRange);
    }
    normalized.push_back(resolved);
  }

  int64_t step = 1;
  if (rank == 1 && normalized.size() > 1) {
    step = normalized[1] - normalized[0];
    if (step <= 0) {
      return fail(GatherToSliceReject::kNotAscending);
    }
    for (std::size_t i = 2; i < normalized.size(); ++i) {
      if (normalized[i] - normalized[i - 1] != step) {
        return fail(GatherToSliceReject::kNotConstantStep);
      }
    }
  }

  const int opset = graph.Builder().OpsetVersion("");
  const bool slice_has_steps = opset == core::shapes::kUnknownOpsetVersion || opset >= 10;
  if (step != 1 && !slice_has_steps) {
    return fail(GatherToSliceReject::kNonUnitStepNeedsOpset10);
  }

  plan.axis = axis;
  plan.start = normalized.front();
  plan.step = step;
  plan.count = static_cast<int64_t>(normalized.size());
  plan.needs_squeeze = rank == 0;
  return true;
}

} // namespace

std::set<std::string> GatherConcatPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult GatherConcatPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsAxis0Gather(candidate)) {
    return NoMatch(candidate, "candidate is not an axis=0 Gather");
  }
  const std::string &index_name = candidate.input()[1].value();
  if (!graph.IsConstant(index_name)) {
    return NoMatch(candidate, "the Gather index is not constant");
  }
  const TensorProto *index = graph.GetComputedConstant(index_name);
  std::vector<int64_t> index_values;
  int index_rank = 0;
  if (index == nullptr || !ReadIntTensor(*index, index_values, index_rank) || index_rank > 1) {
    return NoMatch(candidate, "the Gather index is not a constant scalar or vector");
  }
  const NodeProto *concat = graph.NodeBefore(candidate.input()[0].value());
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat") ||
      GetAttributeOr<int64_t>(*concat, "axis", 0) != 0) {
    return NoMatch(candidate, "the gathered value is not produced by an axis=0 Concat");
  }

  int non_const_position = -1;
  int64_t offset = 0;
  int64_t x_size = -1;
  for (int i = 0; i < concat->input_size(); ++i) {
    const std::string &name = concat->input()[i].value();
    if (graph.IsConstant(name)) {
      if (non_const_position >= 0) {
        continue;
      }
      const TensorProto *cst = graph.GetComputedConstant(name);
      if (cst == nullptr || cst->dims_size() < 1) {
        return NoMatch(candidate, "a Concat constant input has no known size");
      }
      offset += cst->dims()[0];
    } else {
      if (non_const_position >= 0) {
        return NoMatch(candidate, "more than one Concat input is non-constant");
      }
      non_const_position = i;
      if (graph.HasShape(name)) {
        const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
        if (shape.Rank() != 1) {
          return NoMatch(candidate, "the non-constant Concat input is not 1-D");
        }
        if (shape[0].IsInt()) {
          x_size = shape[0].AsInt();
        }
      } else {
        return NoMatch(candidate, "the non-constant Concat input has no known shape");
      }
    }
  }
  if (non_const_position < 0) {
    return NoMatch(candidate, "every Concat input is constant");
  }
  for (int64_t value : index_values) {
    if (value < offset) {
      return NoMatch(candidate, "a requested index falls before the non-constant input");
    }
    if (x_size >= 0 && value >= offset + x_size) {
      return NoMatch(candidate, "a requested index falls after the non-constant input");
    }
  }
  return core::builder::MatchResult{this, {concat, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
GatherConcatPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("GatherConcatPattern::Apply expects a Concat and a Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GatherConcatPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &gather = *nodes[1];

  const TensorProto *index = graph.GetComputedConstant(gather.input()[1].value());
  std::vector<int64_t> index_values;
  int index_rank = 0;
  if (index == nullptr || !ReadIntTensor(*index, index_values, index_rank)) {
    throw BuilderError("GatherConcatPattern::Apply could not read the Gather index.");
  }

  std::string x_name;
  int64_t offset = 0;
  for (const auto &input : concat.input()) {
    const std::string &name = input.value();
    if (graph.IsConstant(name) && x_name.empty()) {
      const TensorProto *cst = graph.GetComputedConstant(name);
      offset += cst->dims()[0];
    } else if (!graph.IsConstant(name)) {
      x_name = name;
      break;
    }
  }

  std::vector<int64_t> shifted;
  shifted.reserve(index_values.size());
  for (int64_t value : index_values) {
    shifted.push_back(value - offset);
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "GatherConcatPattern--" + gather.name().value();
  const std::string index_init = collections::FreeInitializerName(builder, name + "_idx");
  const std::vector<int64_t> dims =
      index_rank == 0 ? std::vector<int64_t>{}
                      : std::vector<int64_t>{static_cast<int64_t>(shifted.size())};
  builder.MakeInitializer(MakeInitializer<int64_t>(index_init.c_str(), dims, shifted));

  NodeProto replacement =
      MakeNode("Gather", {x_name, index_init}, {gather.output()[0].value()}, "", name.c_str());
  AddAttribute<int64_t>(replacement, "axis", 0);

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  replacements.push_back(replacement);
  return replacements;
}

std::set<std::string> GatherGatherPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult GatherGatherPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsAxis0Gather(candidate)) {
    return NoMatch(candidate, "candidate is not an axis=0 Gather");
  }
  const std::string &outer_index_name = candidate.input()[1].value();
  if (!graph.IsConstant(outer_index_name)) {
    return NoMatch(candidate, "the outer Gather index is not constant");
  }
  const TensorProto *outer = graph.GetComputedConstant(outer_index_name);
  std::vector<int64_t> outer_values;
  int outer_rank = 0;
  if (outer == nullptr || !ReadIntTensor(*outer, outer_values, outer_rank)) {
    return NoMatch(candidate, "the outer Gather index is not a constant integer tensor");
  }
  const NodeProto *inner = graph.NodeBefore(candidate.input()[0].value());
  if (inner == nullptr || !IsAxis0Gather(*inner)) {
    return NoMatch(candidate, "the gathered value is not produced by an axis=0 Gather");
  }
  const std::string &inner_index_name = inner->input()[1].value();
  if (!graph.IsConstant(inner_index_name)) {
    return NoMatch(candidate, "the inner Gather index is not constant");
  }
  const TensorProto *innercst = graph.GetComputedConstant(inner_index_name);
  std::vector<int64_t> inner_values;
  int inner_rank = 0;
  if (innercst == nullptr || !ReadIntTensor(*innercst, inner_values, inner_rank) ||
      inner_rank != 1) {
    return NoMatch(candidate, "the inner Gather index is not a constant vector");
  }
  const int64_t length = static_cast<int64_t>(inner_values.size());
  for (int64_t value : outer_values) {
    const int64_t normalized = value < 0 ? value + length : value;
    if (normalized < 0 || normalized >= length) {
      return NoMatch(candidate, "an outer index is out of range of the inner indices");
    }
  }
  return core::builder::MatchResult{this, {inner, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
GatherGatherPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("GatherGatherPattern::Apply expects two Gather nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GatherGatherPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &inner = *nodes[0];
  const NodeProto &outer = *nodes[1];

  std::vector<int64_t> inner_values;
  std::vector<int64_t> outer_values;
  int inner_rank = 0;
  int outer_rank = 0;
  const TensorProto *innercst = graph.GetComputedConstant(inner.input()[1].value());
  const TensorProto *outercst = graph.GetComputedConstant(outer.input()[1].value());
  if (innercst == nullptr || outercst == nullptr ||
      !ReadIntTensor(*innercst, inner_values, inner_rank) ||
      !ReadIntTensor(*outercst, outer_values, outer_rank)) {
    throw BuilderError("GatherGatherPattern::Apply could not read the Gather indices.");
  }
  const int64_t length = static_cast<int64_t>(inner_values.size());
  std::vector<int64_t> fused;
  fused.reserve(outer_values.size());
  for (int64_t value : outer_values) {
    const int64_t normalized = value < 0 ? value + length : value;
    fused.push_back(inner_values[static_cast<std::size_t>(normalized)]);
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "GatherGatherPattern--" + outer.name().value();
  const std::string index_init = collections::FreeInitializerName(builder, name + "_idx");
  const std::vector<int64_t> dims = outer_rank == 0
                                        ? std::vector<int64_t>{}
                                        : std::vector<int64_t>{static_cast<int64_t>(fused.size())};
  builder.MakeInitializer(MakeInitializer<int64_t>(index_init.c_str(), dims, fused));

  NodeProto replacement = MakeNode("Gather", {inner.input()[0].value(), index_init},
                                   {outer.output()[0].value()}, "", name.c_str());
  AddAttribute<int64_t>(replacement, "axis", 0);

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(inner.output()[0].value())) {
    replacements.add() = inner;
  }
  replacements.push_back(replacement);
  return replacements;
}

std::set<std::string> GatherToSlicePattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult GatherToSlicePattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  GatherToSlicePlan plan;
  GatherToSliceReject reject = GatherToSliceReject::kNone;
  if (AnalyzeGatherToSlice(graph, candidate, plan, &reject)) {
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }
  switch (reject) {
  case GatherToSliceReject::kNotGather:
    return NoMatch(candidate, "candidate is not a default-domain Gather");
  case GatherToSliceReject::kNotUnaryGather:
    return NoMatch(candidate, "candidate is not a binary-input, unary-output Gather");
  case GatherToSliceReject::kIndexNotConstant:
    return NoMatch(candidate, "the Gather index is not constant");
  case GatherToSliceReject::kIndexNotScalarOrVector:
    return NoMatch(candidate, "the Gather index is not a constant scalar or vector");
  case GatherToSliceReject::kIndexEmpty:
    return NoMatch(candidate, "the Gather index is empty");
  case GatherToSliceReject::kNegativeAxisNoRank:
    return NoMatch(candidate, "a negative axis requires a known input rank");
  case GatherToSliceReject::kAxisOutOfRange:
    return NoMatch(candidate, "the normalized axis is out of range");
  case GatherToSliceReject::kNegativeIndexNoRank:
    return NoMatch(candidate, "a negative index requires a known input rank");
  case GatherToSliceReject::kAxisOutOfRangeForIndex:
    return NoMatch(candidate, "the Gather axis is out of range");
  case GatherToSliceReject::kDimNotStatic:
    return NoMatch(candidate, "a negative index requires a statically known axis dimension");
  case GatherToSliceReject::kIndexOutOfRange:
    return NoMatch(candidate, "a negative index remains out of range after normalization");
  case GatherToSliceReject::kNotAscending:
    return NoMatch(candidate, "the Gather indices are not a strictly ascending progression");
  case GatherToSliceReject::kNotConstantStep:
    return NoMatch(candidate, "the Gather indices are not a constant-step arithmetic progression");
  case GatherToSliceReject::kNonUnitStepNeedsOpset10:
    return NoMatch(candidate, "a non-unit step requires the Slice input form (opset >= 10)");
  case GatherToSliceReject::kNone:
    break;
  }
  return NoMatch(candidate, "the Gather cannot be rewritten as a Slice");
}

utils::RepeatedProtoField<NodeProto>
GatherToSlicePattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("GatherToSlicePattern::Apply expects one Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GatherToSlicePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &gather = *nodes[0];
  GatherToSlicePlan plan;
  GatherToSliceReject reject = GatherToSliceReject::kNone;
  if (!AnalyzeGatherToSlice(graph, gather, plan, &reject)) {
    throw BuilderError("GatherToSlicePattern::Apply could not analyze the matched Gather.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "GatherToSlicePattern--" + gather.name().value();
  const int opset = builder.OpsetVersion("");
  const bool slice_input_form = opset == core::shapes::kUnknownOpsetVersion || opset >= 10;
  const bool squeeze_input_form = opset == core::shapes::kUnknownOpsetVersion || opset >= 13;

  const int64_t end = plan.start + plan.step * plan.count;
  const std::string &data = gather.input()[0].value();
  const std::string &final_output = gather.output()[0].value();
  const std::string slice_output =
      plan.needs_squeeze ? builder.UniqueName(name + "_sliced") : final_output;

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto slice;
  if (slice_input_form) {
    const std::string starts_init = collections::FreeInitializerName(builder, name + "_starts");
    const std::string ends_init = collections::FreeInitializerName(builder, name + "_ends");
    const std::string axes_init = collections::FreeInitializerName(builder, name + "_axes");
    builder.MakeInitializer(MakeInitializerShape(starts_init.c_str(), {plan.start}));
    builder.MakeInitializer(MakeInitializerShape(ends_init.c_str(), {end}));
    builder.MakeInitializer(MakeInitializerShape(axes_init.c_str(), {plan.axis}));
    std::vector<std::string> inputs{data, starts_init, ends_init, axes_init};
    if (plan.step != 1) {
      const std::string steps_init = collections::FreeInitializerName(builder, name + "_steps");
      builder.MakeInitializer(MakeInitializerShape(steps_init.c_str(), {plan.step}));
      inputs.push_back(steps_init);
    }
    slice = MakeNode("Slice", inputs, {slice_output}, "", (name + "-slice").c_str());
  } else {
    slice = MakeNode("Slice", {data}, {slice_output}, "", (name + "-slice").c_str());
    AddAttribute<std::vector<int64_t>>(slice, "starts", {plan.start});
    AddAttribute<std::vector<int64_t>>(slice, "ends", {end});
    AddAttribute<std::vector<int64_t>>(slice, "axes", {plan.axis});
  }
  if (gather.has_doc_string() && !plan.needs_squeeze) {
    slice.set_doc_string(gather.doc_string().value());
  }
  replacements.push_back(std::move(slice));

  if (plan.needs_squeeze) {
    NodeProto squeeze;
    if (squeeze_input_form) {
      const std::string axes_init = collections::FreeInitializerName(builder, name + "_sq_axes");
      builder.MakeInitializer(MakeInitializerShape(axes_init.c_str(), {plan.axis}));
      squeeze = MakeNode("Squeeze", {slice_output, axes_init}, {final_output}, "", name.c_str());
    } else {
      squeeze = MakeNode("Squeeze", {slice_output}, {final_output}, "", name.c_str());
      AddAttribute<std::vector<int64_t>>(squeeze, "axes", {plan.axis});
    }
    if (gather.has_doc_string()) {
      squeeze.set_doc_string(gather.doc_string().value());
    }
    replacements.push_back(std::move(squeeze));
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
