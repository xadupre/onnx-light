// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/cast_pattern.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/detail/cast_pattern_helpers.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

std::set<std::string> CastPattern::FastOpType() const { return {"Cast"}; }

core::builder::MatchResult CastPattern::Match(core::builder::GraphGraph &graph,
                                              const NodeProto &candidate) const {
  if (!detail::IsDefaultCast(candidate) || !graph.HasType(candidate.input()[0].value())) {
    return {};
  }
  core::symbolic::TensorType target_type;
  if (!detail::CastTarget(candidate, target_type) ||
      target_type != graph.GetType(candidate.input()[0].value())) {
    return {};
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastPattern::Apply(core::builder::GraphGraph &graph,
                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr || !detail::IsDefaultCast(*nodes[0])) {
    throw core::builder::BuilderError("CastPattern::Apply expects one default-domain Cast node.");
  }
  const NodeProto &cast = *nodes[0];
  core::symbolic::TensorType target_type;
  if (!graph.HasType(cast.input()[0].value()) || !detail::CastTarget(cast, target_type) ||
      target_type != graph.GetType(cast.input()[0].value())) {
    throw core::builder::BuilderError("CastPattern::Apply received a non-redundant Cast.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  const std::string name = "CastPattern--" + cast.name().value();
  NodeProto replacement =
      MakeNode("Identity", {cast.input()[0].value()}, {cast.output()[0].value()}, "", name.c_str());
  if (cast.has_doc_string()) {
    replacement.set_doc_string(cast.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
