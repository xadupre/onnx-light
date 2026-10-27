// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/shape_pattern.h"

#include <string>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;

} // namespace

std::set<std::string> ShapeBasedShapeShapeAddPattern::FastOpType() const { return {"Add"}; }

core::builder::MatchResult ShapeBasedShapeShapeAddPattern::Match(core::builder::GraphGraph &graph,
                                                                 const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Add") || candidate.input_size() != 2) {
    return NoMatch(candidate, "candidate is not a default-domain binary Add");
  }
  const NodeProto *shape1 = graph.NodeBefore(candidate.input()[0].value());
  if (shape1 == nullptr || !IsDefaultOp(*shape1, "Shape")) {
    return NoMatch(candidate, "the first Add input is not produced by a default-domain Shape");
  }
  const NodeProto *shape2 = graph.NodeBefore(candidate.input()[1].value());
  if (shape2 == nullptr || !IsDefaultOp(*shape2, "Shape")) {
    return NoMatch(candidate, "the second Add input is not produced by a default-domain Shape");
  }
  return NoMatch(candidate, "the upstream Shape plus Shape rewrite is not implemented");
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedShapeShapeAddPattern::Apply(core::builder::GraphGraph &,
                                      const std::vector<const NodeProto *> &) const {
  throw BuilderError("ShapeBasedShapeShapeAddPattern is not implemented.");
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
