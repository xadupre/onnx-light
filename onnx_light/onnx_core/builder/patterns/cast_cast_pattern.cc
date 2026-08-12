// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/patterns/cast_cast_pattern.h"

#include "onnx_core/builder/graph_builder_pattern_optimization.h"
#include "onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

bool IsDefaultCast(const NodeProto &node) {
  return node.op_type().value() == "Cast" &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain && node.input_size() == 1 &&
         node.output_size() == 1;
}

bool CastTarget(const NodeProto &node, TensorType &type) {
  const AttributeProto *attribute = FindAttribute(node, "to");
  if (attribute == nullptr || attribute->type() != AttributeProto::AttributeType::INT) {
    return false;
  }
  type = symbolic::DataTypeToTensorType(static_cast<TensorProto::DataType>(attribute->i()));
  return type != TensorType::kUndefined;
}

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

MatchResult CastCastPattern::Match(GraphBuilderPatternOptimization &opt,
                                   const NodeProto &candidate) const {
  const GraphGraph &graph = opt.Graph();
  if (!IsDefaultCast(candidate)) {
    return {};
  }
  const NodeProto *inner = graph.NodeBefore(candidate.input()[0].value());
  if (inner == nullptr || !IsDefaultCast(*inner) || !graph.HasType(inner->input()[0].value())) {
    return {};
  }

  TensorType middle_type;
  TensorType final_type;
  if (!CastTarget(*inner, middle_type) || !CastTarget(candidate, final_type) ||
      OneCastType(graph.GetType(inner->input()[0].value()), middle_type, final_type) ==
          TensorType::kUndefined) {
    return {};
  }
  return MatchResult{this,
                     {inner, &candidate},
                     graph.IsUsedMoreThanOnce(inner->output()[0].value()) ? nullptr : &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastCastPattern::Apply(GraphBuilderPatternOptimization &opt,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr ||
      !IsDefaultCast(*nodes[0]) || !IsDefaultCast(*nodes[1])) {
    throw BuilderError("CastCastPattern::Apply expects two consecutive Cast nodes.");
  }
  const NodeProto &inner = *nodes[0];
  const NodeProto &outer = *nodes[1];
  const GraphGraph &graph = opt.Graph();
  TensorType middle_type;
  TensorType final_type;
  if (!graph.HasType(inner.input()[0].value()) || !CastTarget(inner, middle_type) ||
      !CastTarget(outer, final_type)) {
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
    AddAttribute<int64_t>(replacement, "to",
                          static_cast<int64_t>(symbolic::TensorTypeToDataType(replacement_type)));
  }
  replacements.add() = std::move(replacement);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
