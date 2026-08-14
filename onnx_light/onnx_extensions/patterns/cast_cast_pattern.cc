// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/cast_cast_pattern.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/detail/cast_pattern_helpers.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;
using core::builder::GraphGraph;
using core::symbolic::TensorType;

bool IsFloatingRoundTripType(TensorType type) {
  return type == TensorType::kFloat16 || type == TensorType::kBfloat16 ||
         type == TensorType::kFloat;
}

} // namespace

std::set<std::string> CastCastPattern::FastOpType() const { return {"Cast"}; }

TensorType CastCastPattern::OneCastType(TensorType input_type, TensorType middle_type,
                                        TensorType final_type) {
  if (input_type == final_type) {
    if (middle_type == input_type) {
      return input_type;
    }
    if (IsFloatingRoundTripType(input_type) &&
        (middle_type == TensorType::kFloat || middle_type == TensorType::kDouble)) {
      return input_type;
    }
  } else if (final_type == middle_type) {
    return middle_type;
  } else if (input_type == middle_type) {
    return final_type;
  }
  return TensorType::kUndefined;
}

core::builder::MatchResult CastCastPattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!detail::IsDefaultCast(candidate)) {
    return {};
  }
  const NodeProto *inner = graph.NodeBefore(candidate.input()[0].value());
  if (inner == nullptr || !detail::IsDefaultCast(*inner) ||
      !graph.HasType(inner->input()[0].value())) {
    return {};
  }

  TensorType middle_type;
  TensorType final_type;
  if (!detail::CastTarget(*inner, middle_type) || !detail::CastTarget(candidate, final_type) ||
      OneCastType(graph.GetType(inner->input()[0].value()), middle_type, final_type) ==
          TensorType::kUndefined) {
    return {};
  }
  return core::builder::MatchResult{
      this,
      {inner, &candidate},
      graph.IsUsedMoreThanOnce(inner->output()[0].value()) ? nullptr : &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastCastPattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      !detail::IsDefaultCast(*nodes[0]) || !detail::IsDefaultCast(*nodes[1])) {
    throw BuilderError("CastCastPattern::Apply expects two consecutive Cast nodes.");
  }
  const NodeProto &inner = *nodes[0];
  const NodeProto &outer = *nodes[1];
  TensorType middle_type;
  TensorType final_type;
  if (!graph.HasType(inner.input()[0].value()) || !detail::CastTarget(inner, middle_type) ||
      !detail::CastTarget(outer, final_type)) {
    throw BuilderError("CastCastPattern::Apply received an invalid Cast match.");
  }
  const TensorType input_type = graph.GetType(inner.input()[0].value());
  const TensorType replacement_type = OneCastType(input_type, middle_type, final_type);
  if (replacement_type == TensorType::kUndefined) {
    throw BuilderError("CastCastPattern::Apply received an unsafe Cast match.");
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(inner.output()[0].value())) {
    replacements.add() = inner;
  }

  NodeProto replacement =
      MakeNode(replacement_type == input_type ? "Identity" : "Cast", {inner.input()[0].value()},
               {outer.output()[0].value()}, "", "CastCastPattern");
  if (replacement_type != input_type) {
    AddAttribute<int64_t>(
        replacement, "to",
        static_cast<int64_t>(core::symbolic::TensorTypeToDataType(replacement_type)));
  }
  replacements.add() = std::move(replacement);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
