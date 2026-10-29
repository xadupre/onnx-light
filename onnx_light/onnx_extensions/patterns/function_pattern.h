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
 * @code
 * Before:
 *   inputs ──→ ┌──────────────────┐ ──→ outputs
 *              │ template body    │
 *              └──────────────────┘
 * After:
 *   inputs ──→ ┌──────────────────┐ ──→ outputs
 *              │ replacement body │
 *              └──────────────────┘
 * @endcode
 *
 * With ``as_function=true``, After is a call to the replacement function instead
 * of its inlined body. One compatible definition is shared by all occurrences,
 * including calls in subgraphs. Captures become call attributes; shape inference
 * follows the registered local body. Function-call replacement outputs must be
 * distinct. The pattern does not rewrite that body into a recursive call to itself.
 *
 * Bodies must be topologically sorted SSA graphs without graph-valued
 * attributes. Template inputs must be used, and template outputs must be
 * distinct node results. Template nodes must be connected through nonempty
 * values (including shared inputs). Matching follows local producers and
 * consumers from the selected anchor, never scans unrelated graph nodes. Replacement outputs may
 * alias inputs, and in inline mode may alias each other. Configured target nodes are conservatively
 * rejected. Intermediates
 * may not escape the match. Subgraph captures and matches needing interleaved external consumers
 * are conservatively rejected. Function definitions are copied. Forwarded or aliased outputs need a
 * default ONNX opset for Identity.
 *
 * Unlike built-in rewrites, this configurable class has no default factory:
 * register an instance with the desired function pair instead.
 */
class FunctionPattern : public core::builder::PatternOptimization {
public:
  /// Creates a pattern; anchor selects a zero-based template node (-1 means the last).
  FunctionPattern(const FunctionProto &pattern, const FunctionProto &replacement, int priority = 1,
                  std::string name = "FunctionPattern",
                  core::symbolic::Device device = core::symbolic::Device::kUndefined,
                  int anchor = -1, bool as_function = false);

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  FunctionProto pattern_;
  FunctionProto replacement_;
  bool as_function_;
  std::size_t anchor_index_;
  std::vector<std::size_t> match_order_;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
