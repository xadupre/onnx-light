// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces no-op operations by an Identity node.
 *
 * The pattern recognises redundant arithmetic and layout operations such as
 * ``Add(x, 0)``, ``Mul(x, 1)``, ``Sub(x, 0)``, ``Div(x, 1)``, ``Or(x, false)``,
 * ``And(x, true)``,
 * ``Transpose(x, perm=[0, 1, 2, ...])``, ``Reshape(x, [0, 0, ...])``,
 * a full-range ``Slice``, an ``Expand`` with an all-one target shape of the
 * input rank, and a half-precision inference ``BatchNormalization`` with
 * scale/bias/mean/variance constants ``1/0/0/1``.
 *
 * @code
 * Before:
 *               ┌─────┐
 *   x, zero ──▶ │ Add │ ───▶ y
 *               └─────┘
 *
 * After:
 *          ┌──────────┐
 *   x ───▶ │ Identity │ ───▶ y
 *          └──────────┘
 * @endcode
 *
 * The no-op node is replaced while its selected data input and first output
 * name are retained; neutral constants no longer feed that output.
 */
class IdentityPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit IdentityPattern(int priority = 0) : PatternOptimization(priority, "Identity") {}

  /// Returns the operator types this pattern can start from.
  std::set<std::string> FastOpType() const override;

  /// Finds a redundant operation rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched operation with an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
