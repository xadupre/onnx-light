// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Moves equal ``Squeeze`` operations after an ``Add``.
 *
 * @code
 * Before:
 *                  ┌─────────┐
 *   x, axes=a ────▶│ Squeeze │────▶ sx
 *                  └─────────┘
 *
 *                  ┌─────────┐
 *   z, axes=a ────▶│ Squeeze │────▶ sz
 *                  └─────────┘
 *
 *              ┌─────┐
 *   sx, sz ───▶│ Add │────▶ y
 *              └─────┘
 *
 * After:
 *             ┌─────┐
 *   x, z ────▶│ Add │────▶ t
 *             └─────┘
 *
 *                  ┌─────────┐
 *   t, axes=a ────▶│ Squeeze │────▶ y
 *                  └─────────┘
 * @endcode
 *
 * The axes must be equal constants (or inferred axis ``0`` for a ``[1]``
 * input). A ``Squeeze`` is retained if its output has another consumer.
 */
class SqueezeAddPattern final : public core::builder::PatternOptimization {
public:
  explicit SqueezeAddPattern(int priority = 0) : PatternOptimization(priority, "SqueezeAdd") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves equal, unshared ``Unsqueeze`` operations after a ``Mul``.
 *
 * @code
 * Before:
 *                  ┌───────────┐
 *   x, axes=a ────▶│ Unsqueeze │────▶ ux
 *                  └───────────┘
 *
 *                  ┌───────────┐
 *   z, axes=a ────▶│ Unsqueeze │────▶ uz
 *                  └───────────┘
 *
 *              ┌─────┐
 *   ux, uz ───▶│ Mul │────▶ y
 *              └─────┘
 *
 * After:
 *             ┌─────┐
 *   x, z ────▶│ Mul │────▶ t
 *             └─────┘
 *
 *                  ┌───────────┐
 *   t, axes=a ────▶│ Unsqueeze │────▶ y
 *                  └───────────┘
 * @endcode
 */
class MulUnsqueezeUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit MulUnsqueezeUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "MulUnsqueezeUnsqueeze") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Cancels a scalar-producing ``Squeeze``/binary/``Unsqueeze`` chain by
 * expanding the scalar right operand instead.
 *
 * @code
 * Before:
 *          ┌─────────┐
 *   x ────▶│ Squeeze │────▶ s
 *          └─────────┘
 *               │
 *               ▼
 *          ┌────────┐
 *          │ Binary │◀──── c
 *          └────────┘
 *               │
 *               ▼
 *        ┌───────────┐
 *        │ Unsqueeze │◀──── axes=[0]
 *        └───────────┘
 *               │
 *               ▼
 *               y
 *
 * After:
 *                    ┌───────────┐
 *   c, axes=[0] ────▶│ Unsqueeze │────▶ uc
 *                    └───────────┘
 *                          │
 *                          ▼
 *                       ┌────────┐
 *   x ─────────────────▶│ Binary │────▶ y
 *                       └────────┘
 * @endcode
 *
 * The binary operator may be ``Add``, ``Div``, ``Mul``, or ``Sub``. Its left
 * input and output must be unshared, and the final ``Unsqueeze`` axis must be
 * the scalar constant ``0``.
 */
class SqueezeBinaryUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  explicit SqueezeBinaryUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "SqueezeBinaryUnsqueeze") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Swaps an unshared ``Unsqueeze`` and ``Transpose``, remapping both axes and
 * permutation.
 *
 * @code
 * Before:
 *                    ┌───────────┐
 *   x, axes=[1] ────▶│ Unsqueeze │────▶ u
 *                    └───────────┘
 *                          │
 *                          ▼
 *               ┌───────────────────┐
 *               │ Transpose [0,2,1] │────▶ y
 *               └───────────────────┘
 *
 * After:
 *          ┌───────────────────┐
 *   x ────▶│ Transpose [0,1]   │────▶ t
 *          └───────────────────┘
 *                     │
 *                     ▼
 *              ┌───────────┐
 *              │ Unsqueeze │◀──── axes=[2]
 *              └───────────┘
 *                     │
 *                     ▼
 *                     y
 * @endcode
 */
class SwapUnsqueezeTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit SwapUnsqueezeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "SwapUnsqueezeTranspose") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a ``Transpose`` that only repositions size-one axes around at most
 * one other dimension with a ``Reshape``.
 *
 * @code
 * Before:
 *          ┌──────────────────────┐
 *   x ────▶│ Transpose [0,2,1,3]  │────▶ y
 *          └──────────────────────┘
 *
 * After:
 *   x, [0,3,1,0]
 *         │
 *         ▼
 *    ┌─────────┐
 *    │ Reshape │────▶ y
 *    └─────────┘
 *
 * Both graphs produce shape [2,3,1,1] from input shape [2,1,3,1].
 * @endcode
 */
class TransposeEqualReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeEqualReshapePattern(int priority = 0)
      : PatternOptimization(priority, "TransposeEqualReshape") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a constant ``Reshape`` across one of two adjacent ``Transpose`` nodes
 * when the input and target dimensions can be aligned by merging or splitting
 * contiguous dimensions.
 *
 * @code
 * Before:
 *          ┌───────────────────┐
 *   x ────▶│ Transpose [1,0,2] │────▶ t0
 *          └───────────────────┘
 *                     │
 *                     ▼
 *                ┌─────────┐
 *                │ Reshape │◀──── [6,4]
 *                └─────────┘
 *                     │
 *                     ▼
 *              ┌─────────────────┐
 *              │ Transpose [1,0] │────▶ y
 *              └─────────────────┘
 *
 * After:
 *          ┌───────────────────┐
 *   x ────▶│ Transpose [1,0,2] │────▶ t0
 *          └───────────────────┘
 *                     │
 *                     ▼
 *              ┌───────────────────┐
 *              │ Transpose [2,0,1] │────▶ t1
 *              └───────────────────┘
 *                     │
 *                     ▼
 *                ┌─────────┐
 *                │ Reshape │◀──── [4,6]
 *                └─────────┘
 *                     │
 *                     ▼
 *                     y
 *
 * Before (dual rank-expanding form):
 *          ┌───────────┐
 *   x ────▶│ Transpose │────▶ t0
 *          └───────────┘
 *                │
 *                ▼
 *           ┌─────────┐
 *           │ Reshape │────▶ r
 *           └─────────┘
 *                │
 *                ▼
 *          ┌───────────┐
 *          │ Transpose │────▶ y
 *          └───────────┘
 *
 * After (dual rank-expanding form):
 *                  ┌─────────┐
 *   x, target ────▶│ Reshape │────▶ r
 *                  └─────────┘
 *                       │
 *                       ▼
 *                ┌────────────────────┐
 *                │ Remapped Transpose │────▶ y
 *                └────────────────────┘
 * @endcode
 *
 * In the dual rank-expanding case, the first ``Transpose`` is removed and the
 * ``Reshape`` is moved before a remapped ``Transpose``; the final
 * ``Transpose`` and output are retained.
 */
class TransposeReshapeTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeReshapeTransposePattern(int priority = 0)
      : PatternOptimization(priority, "TransposeReshapeTranspose") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
