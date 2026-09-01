// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces dynamic dimensions in a concatenated Reshape target with one inferred dimension.
 *
 * @code
 * Before:
 *                                    ┌────────┐
 *   constant, dynamic, Shape dims ──▶│ Concat │────▶ target
 *                                    └────────┘
 *
 *                 ┌─────────┐
 *   x, target ───▶│ Reshape │────▶ y
 *                 └─────────┘
 *
 * After:
 *                                 ┌────────┐
 *   constant, [-1], Shape dims ──▶│ Concat │────▶ target2
 *                                 └────────┘
 *
 *                  ┌─────────┐
 *   x, target2 ───▶│ Reshape │────▶ y
 *                  └─────────┘
 * @endcode
 *
 * Every non-``Shape`` dynamic slot is replaced by ``[-1]``; when all dynamic
 * slots come from ``Shape``, the last such slot is replaced. Dynamic inputs
 * must each describe one dimension and the target may contain only one inferred
 * dimension. A shared original ``Concat`` is retained for its other consumers.
 */
class ConcatReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ConcatReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ConcatReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes a Reshape whose constant target already equals its input shape.
 *
 * @code
 * Before:
 *                    ┌─────────┐
 *   x, target=S ────▶│ Reshape │────▶ y
 *                    └─────────┘
 *
 * After:
 *                  ┌──────────┐
 *   x ────────────▶│ Identity │────▶ y
 *                  └──────────┘
 * @endcode
 *
 * The input shape must be fully static and match every rank and dimension of
 * the materialized target ``S``.
 */
class ReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapePattern(int priority = 0) : PatternOptimization(priority, "Reshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a dimension-removing Reshape into a reduction.
 *
 * @code
 * Before:
 *                ┌───────────────────┐
 *   x, axes ────▶│ Reduce keepdims=1 │────▶ t
 *                └───────────────────┘
 *                          │
 *                          ▼
 *                     ┌─────────┐
 *                     │ Reshape │◀──── target without reduced axes
 *                     └─────────┘
 *                          │
 *                          ▼
 *                          y
 *
 * After:
 *                ┌───────────────────┐
 *   x, axes ────▶│ Reduce keepdims=0 │────▶ y
 *                └───────────────────┘
 * @endcode
 *
 * The reduction output must be unshared, and the final shape must equal the
 * input shape with the reduced axes removed.
 */
class ReduceReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReduceReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ReduceReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a binary operation to the common outer shape represented by surrounding Reshapes.
 *
 * @code
 * Before:
 *          ┌──────────┐
 *   x ────▶│ Reshape? │────▶ rx
 *          └──────────┘
 *
 *          ┌──────────┐
 *   z ────▶│ Reshape? │────▶ rz
 *          └──────────┘
 *
 *               ┌────────┐
 *   rx, rz ────▶│ Binary │────▶ b
 *               └────────┘
 *                   │
 *                   ▼
 *              ┌──────────┐
 *              │ Reshape? │────▶ y
 *              └──────────┘
 *
 * At least two of the three optional Reshape nodes are present.
 *
 * After:
 *          ┌───────────────────┐
 *   x ────▶│ Reshape if needed │────▶ rx
 *          └───────────────────┘
 *
 *          ┌───────────────────┐
 *   z ────▶│ Reshape if needed │────▶ rz
 *          └───────────────────┘
 *
 *               ┌────────┐
 *   rx, rz ────▶│ Binary │────▶ t
 *               └────────┘
 *                   │
 *                   ▼
 *           ┌───────────────────┐
 *           │ Reshape if needed │────▶ y
 *           └───────────────────┘
 * @endcode
 *
 * The binary operation must not broadcast, and the available source/final
 * shapes must be equal. Existing input Reshapes are removed; a missing input
 * Reshape or output Reshape is inserted only when needed to preserve the
 * original output shape.
 * Shared input Reshapes are retained for their other consumers.
 */
class Reshape2Of3Pattern final : public core::builder::PatternOptimization {
public:
  explicit Reshape2Of3Pattern(int priority = 0) : PatternOptimization(priority, "Reshape2Of3") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves equal input Reshapes after an element-wise binary operation.
 *
 * @code
 * Before:
 *                    ┌─────────┐
 *   x, target=s ────▶│ Reshape │────▶ rx
 *                    └─────────┘
 *
 *                    ┌─────────┐
 *   z, target=s ────▶│ Reshape │────▶ rz
 *                    └─────────┘
 *
 *              ┌────────┐
 *   rx, rz ───▶│ Binary │────▶ y
 *              └────────┘
 *
 * After:
 *             ┌────────┐
 *   x, z ────▶│ Binary │────▶ t
 *             └────────┘
 *                 │
 *                 ▼
 *            ┌─────────┐
 *            │ Reshape │◀──── target=s
 *            └─────────┘
 *                 │
 *                 ▼
 *                 y
 * @endcode
 *
 * The two source tensors must have equal shapes, both targets must be the same
 * constant, and neither input Reshape output may be shared.
 */
class ReshapeReshapeBinaryPattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeReshapeBinaryPattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeReshapeBinary") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Composes two consecutive Reshapes into one.
 *
 * @code
 * Before:
 *                     ┌─────────┐
 *   x, target=s1 ────▶│ Reshape │────▶ t
 *                     └─────────┘
 *                          │
 *                          ▼
 *                     ┌─────────┐
 *                     │ Reshape │◀──── target=s2
 *                     └─────────┘
 *                          │
 *                          ▼
 *                          y
 *
 * After:
 *                     ┌─────────┐
 *   x, target=s2' ───▶│ Reshape │────▶ y
 *                     └─────────┘
 * @endcode
 *
 * The first output must be unshared. The replacement normally uses ``s2``;
 * copied zero dimensions are substituted or rebuilt when they refer to
 * ``s1``, and unsafe combinations of inferred dimensions are rejected.
 */
class ReshapeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a Squeeze into the constant target of a preceding Reshape.
 *
 * @code
 * Before:
 *                              ┌─────────┐
 *   x, target=[...,1,...] ────▶│ Reshape │────▶ t
 *                              └─────────┘
 *                                   │
 *                                   ▼
 *                              ┌─────────┐
 *                              │ Squeeze │◀──── axes=a
 *                              └─────────┘
 *                                   │
 *                                   ▼
 *                                   y
 *
 * After:
 *                               ┌─────────┐
 *   x, target without axes ────▶│ Reshape │────▶ y
 *                               └─────────┘
 * @endcode
 *
 * Every explicit Squeeze axis must select a unit target dimension. Removed
 * dimensions are deleted from the replacement target; copied zero dimensions
 * may not follow a removed axis.
 */
class ReshapeSqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeSqueezePattern(int priority = 0)
      : PatternOptimization(priority, "ReshapeSqueeze") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Materializes a Concat-built Reshape target from inferred input and output shapes.
 *
 * @code
 * Before:
 *                     ┌────────┐
 *   shape pieces ────▶│ Concat │────▶ target
 *                     └────────┘
 *                          │
 *                          ▼
 *                     ┌─────────┐
 *                     │ Reshape │◀──── x
 *                     └─────────┘
 *                          │
 *                          ▼
 *                          y
 *
 * After:
 *                                      ┌─────────┐
 *   x, aligned target initializer ────▶│ Reshape │────▶ y
 *                                      └─────────┘
 * @endcode
 *
 * Input and output dimensions must admit an unambiguous alignment with at most
 * one inferred dimension. Only the Reshape is replaced; the old target branch
 * remains if it has other uses.
 */
class ShapeBasedEditDistanceReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedEditDistanceReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedEditDistanceReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a Reshape or all-one Expand that only inserts or removes unit dimensions.
 *
 * @code
 * Before:
 *                  ┌────────────────┐
 *   x, target ────▶│ Reshape/Expand │────▶ y
 *                  └────────────────┘
 *
 * After (unit dimensions are removed):
 *                  ┌─────────┐
 *   x, axes=a ────▶│ Squeeze │────▶ y
 *                  └─────────┘
 *
 * After (unit dimensions are inserted):
 *                  ┌───────────┐
 *   x, axes=a ────▶│ Unsqueeze │────▶ y
 *                  └───────────┘
 * @endcode
 *
 * Known input and output shapes must differ only by dimensions of size one.
 * ``Expand`` additionally requires an all-one constant target. The rewrite
 * requires opset 18 or later.
 */
class ShapeBasedReshapeIsSqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedReshapeIsSqueezePattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedReshapeIsSqueeze") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes a same-rank Reshape whose target copies every dimension but the last.
 *
 * @code
 * Before:
 *                       ┌─────────┐
 *   x, [0,...,0,d] ────▶│ Reshape │────▶ y
 *                       └─────────┘
 *
 * After:
 *                      ┌──────────┐
 *   x ────────────────▶│ Identity │────▶ y
 *                      └──────────┘
 * @endcode
 *
 * The target must be constant, non-empty, have the input rank, contain only
 * leading zeros, and end in a nonzero dimension.
 */
class ShapedBasedReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ShapedBasedReshapePattern(int priority = 0)
      : PatternOptimization(priority, "ShapedBasedReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces the sole dynamic element of a concatenated Reshape target with ``[-1]``.
 *
 * @code
 * Before:
 *                                 ┌────────┐
 *   constant dims, dynamic dim ──▶│ Concat │────▶ target
 *                                 └────────┘
 *
 *                 ┌─────────┐
 *   x, target ───▶│ Reshape │────▶ y
 *                 └─────────┘
 *
 * After:
 *                          ┌────────┐
 *   constant dims, [-1] ──▶│ Concat │────▶ target2
 *                          └────────┘
 *
 *                  ┌─────────┐
 *   x, target2 ───▶│ Reshape │────▶ y
 *                  └─────────┘
 * @endcode
 *
 * Constants may not already contain ``-1`` and exactly one dynamic input must
 * have shape ``[1]``. A shared original ``Concat`` is retained for its other
 * consumers.
 */
class StaticConcatReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit StaticConcatReshapePattern(int priority = 0)
      : PatternOptimization(priority, "StaticConcatReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes a Squeeze or Unsqueeze immediately before a Reshape.
 *
 * @code
 * Before:
 *                  ┌───────────────────┐
 *   x, axes=a ────▶│ Squeeze/Unsqueeze │────▶ t
 *                  └───────────────────┘
 *                            │
 *                            ▼
 *                       ┌─────────┐
 *                       │ Reshape │◀──── target
 *                       └─────────┘
 *                            │
 *                            ▼
 *                            y
 *
 * After:
 *                  ┌─────────┐
 *   x, target ────▶│ Reshape │────▶ y
 *                  └─────────┘
 * @endcode
 *
 * The intermediate value must be unshared. If the constant Reshape target uses
 * copied zero dimensions, all such dimensions must precede the first changed
 * axis so that they still refer to the same source dimensions.
 */
class UnsqueezeOrSqueezeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit UnsqueezeOrSqueezeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeOrSqueezeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Collapses a specific Unsqueeze-Reshape sequence into one Unsqueeze.
 *
 * @code
 * Before:
 *                    ┌───────────┐
 *   x, axes=[2] ────▶│ Unsqueeze │────▶ t
 *                    └───────────┘
 *                          │
 *                          ▼
 *                     ┌─────────┐
 *                     │ Reshape │◀──── [0,1,-1,0]
 *                     └─────────┘
 *                          │
 *                          ▼
 *                          y
 *
 * After:
 *                    ┌───────────┐
 *   x, axes=[1] ────▶│ Unsqueeze │────▶ y
 *                    └───────────┘
 * @endcode
 *
 * The source rank must be three and the intermediate Unsqueeze output must be
 * unshared.
 */
class UnsqueezeReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit UnsqueezeReshapePattern(int priority = 0)
      : PatternOptimization(priority, "UnsqueezeReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
