// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Creates a rewrite from two ONNX function bodies with fixed input/output lists.
 *
 * Inputs and outputs correspond by position, not by name. Node arities, domains,
 * overloads, literal attributes and imported opset versions must match exactly.
 * A template attribute with ref_attr_name captures the candidate attribute; the
 * replacement may reference that capture, including under another attribute name.
 * Repeated references must capture equal values of the declared type.
 *
 * Before:
 *   inputs ──→ ┌──────────────────┐ ──→ outputs
 *              │ template body    │
 *              └──────────────────┘
 * After:
 *   inputs ──→ ┌──────────────────┐ ──→ outputs
 *              │ replacement body │
 *              └──────────────────┘
 *
 * Bodies must be topologically sorted SSA graphs without graph-valued
 * attributes. Template inputs must be used, and template outputs must be
 * distinct node results. Replacement outputs may alias inputs or each other.
 * Intermediates may not escape the match. Subgraph captures and matches needing
 * interleaved external consumers are conservatively rejected. Function definitions
 * are copied. Forwarded or aliased outputs need a default ONNX opset for Identity.
 *
 * Unlike built-in rewrites, this configurable class has no default factory:
 * register an instance with the desired function pair instead.
 */
class FunctionPattern : public core::builder::PatternOptimization {
public:
  /// Creates a pattern from a template and an equivalent replacement.
  FunctionPattern(const FunctionProto &pattern, const FunctionProto &replacement, int priority = 1,
                  std::string name = "FunctionPattern",
                  core::symbolic::Device device = core::symbolic::Device::kUndefined);

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  FunctionProto pattern_;
  FunctionProto replacement_;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
