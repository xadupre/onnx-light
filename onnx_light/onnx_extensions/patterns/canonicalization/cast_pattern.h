// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Canonicalizes a Cast to Identity when its input already has the target type.
 *
 * @code
 * Before:
 *            ┌───────────┐
 *   x:T ───▶ │ Cast to T │ ───▶ y:T
 *            └───────────┘
 *
 * After:
 *            ┌──────────┐
 *   x:T ───▶ │ Identity │ ───▶ y:T
 *            └──────────┘
 * @endcode
 */
class CastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastPattern(int priority = 0) : PatternOptimization(priority, "Cast") {}

  /// Returns ``Cast`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a redundant type-preserving Cast rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces one matched Cast with an Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Canonicalizes consecutive Cast nodes when a single conversion is equivalent.
 *
 * @code
 * Before:
 *            ┌───────────┐               ┌───────────┐
 *   x:A ───▶ │ Cast to B │ ───▶ m:B ───▶ │ Cast to C │ ───▶ y:C
 *            └───────────┘               └───────────┘
 *
 * After:
 *            ┌────────────┐
 *   x:A ───▶ │ Cast to C* │ ───▶ y:C
 *            └────────────┘
 *
 * After (no conversion is needed):
 *            ┌──────────┐
 *   x:A ───▶ │ Identity │ ───▶ y:C
 *            └──────────┘
 *
 * After (the intermediate m:B is shared):
 *            ┌───────────┐
 *   x:A ───▶ │ Cast to B │ ───▶ m:B ───▶ other consumer
 *            └───────────┘
 *
 *            ┌────────────┐
 *   x:A ───▶ │ Cast to C* │ ───▶ y:C
 *            └────────────┘
 * @endcode
 *
 * ``C*`` denotes the single safe replacement type selected from the input,
 * intermediate, and final types. If ``m`` has another consumer, the first Cast
 * is retained for that consumer while ``y`` is still produced directly from
 * ``x``.
 */
class CastCastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastCastPattern(int priority = 1) : PatternOptimization(priority, "CastCast") {}

  /// Returns ``Cast`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a safe consecutive-Cast rewrite rooted at ``candidate``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces a matched Cast pair with one Cast or Identity node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  static core::symbolic::TensorType OneCastType(core::symbolic::TensorType input_type,
                                                core::symbolic::TensorType middle_type,
                                                core::symbolic::TensorType final_type);
};

/**
 * Moves matching floating-point Cast nodes after a binary arithmetic operation.
 *
 * @code
 * Before:
 *            ┌───────────┐
 *   x:A ───▶ │ Cast to B │ ───▶ xb:B
 *            └───────────┘
 *
 *            ┌───────────┐
 *   y:A ───▶ │ Cast to B │ ───▶ yb:B
 *            └───────────┘
 *
 *                 ┌────────┐
 *   xb:B, yb:B ─▶ │ Binary │ ───▶ z:B
 *                 └────────┘
 *
 * After:
 *                ┌────────┐
 *   x:A, y:A ──▶ │ Binary │ ───▶ t:A
 *                └────────┘
 *                     │
 *                     ▼
 *                ┌───────────┐
 *                │ Cast to B │ ───▶ z:B
 *                └───────────┘
 * @endcode
 */
class CastCastBinaryPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastCastBinaryPattern(int priority = 1)
      : PatternOptimization(priority, "CastCastBinary") {}

  /// Returns the supported binary arithmetic operator types.
  std::set<std::string> FastOpType() const override;

  /// Finds two compatible Cast nodes feeding a binary arithmetic operation.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Cast-Cast-binary sequence with binary-Cast.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves an operation from a temporary floating-point type to its result type.
 *
 * @code
 * Before:
 *            ┌───────────┐
 *   y:R ───▶ │ Cast to C │ ───▶ yc:C
 *            └───────────┘
 *
 *                 ┌────┐
 *   x:C, yc:C ──▶ │ Op │ ───▶ t:C
 *                 └────┘
 *                   │
 *                   ▼
 *              ┌───────────┐
 *              │ Cast to R │ ───▶ z:R
 *              └───────────┘
 *
 * After:
 *            ┌───────────┐
 *   x:C ───▶ │ Cast to R │ ───▶ xr:R
 *            └───────────┘
 *
 *                ┌────┐
 *   xr:R, y:R ─▶ │ Op │ ───▶ z:R
 *                └────┘
 *
 * After (the original Op output t:C is shared):
 *            ┌───────────┐
 *   z:R ───▶ │ Cast to C │ ───▶ t:C
 *            └───────────┘
 *
 * The unary form uses the same rewrite with one input.
 * @endcode
 *
 * Inputs already produced in ``R`` lose their Cast; other inputs gain one.
 * The unary case follows the same transformation with a single input. If the
 * original ``Op:C`` output is retained elsewhere, a Cast from ``z:R`` back to
 * that output in ``C`` is also emitted.
 */
class CastOpCastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit CastOpCastPattern(int priority = 1) : PatternOptimization(priority, "CastOpCast") {}

  /// Returns the supported unary and binary operator types.
  std::set<std::string> FastOpType() const override;

  /// Finds an operation between compatible input and output Cast nodes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the operation in the output type and removes the surrounding Cast nodes.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
