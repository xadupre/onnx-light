// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Removes an ``Expand`` that does not change the shape of its input.
 *
 * @code
 * Before:
 *                             ┌────────┐
 *   x, shape=[D0,...,Dn] ────→│ Expand │────→ y
 *                             └────────┘
 *
 * After:
 *                            ┌──────────┐
 *   x ──────────────────────→│ Identity │────→ y
 *                            └──────────┘
 * @endcode
 *
 * The rewrite replaces ``Expand(x, shape)`` with ``Identity(x)`` when the shape
 * of ``x`` is fully known and equals the constant ``shape`` fed to ``Expand``.
 */
class ExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandPattern(int priority = 0) : PatternOptimization(priority, "Expand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` whose constant target shape equals its input shape.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the redundant ``Expand`` with ``Identity``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Drops an ``Expand`` feeding an element-wise binary operator.
 *
 * @code
 * Before:
 *             ┌────────┐
 *   x, S ────→│ Expand │────→ e
 *             └────────┘
 *
 *            ┌────────┐
 *   e, z ───→│ Binary │────→ y
 *            └────────┘
 *
 * After:
 *            ┌────────┐
 *   x, z ───→│ Binary │────→ y
 *            └────────┘
 * @endcode
 *
 * ``S`` is constant, ``x`` and ``z`` have the same rank, and every aligned
 * dimension is equal or one. The unshared ``Expand`` is removed because
 * ``Op`` can broadcast ``x`` directly; operand order is preserved.
 */
class ExpandBroadcastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandBroadcastPattern(int priority = 0)
      : PatternOptimization(priority, "ExpandBroadcast") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single broadcasting binary operator.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the binary operator on the pre-expanded input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Simplifies a dynamic ``Concat`` used as an ``Expand`` target shape.
 *
 * @code
 * Before:
 *                         ┌────────┐
 *   p0,...,pk,...,pn ────→│ Concat │────→ s
 *                         └────────┘
 *                             │
 *                             ↓
 *                        ┌────────┐
 *                        │ Expand │←──── x
 *                        └────────┘
 *                             │
 *                             ↓
 *                             y
 *
 * After:
 *                         ┌────────┐
 *   [1],...,pk,...,[1] ──→│ Concat │────→ s2
 *                         └────────┘
 *                             │
 *                             ↓
 *                        ┌────────┐
 *                        │ Expand │←──── x
 *                        └────────┘
 *                             │
 *                             ↓
 *                             y
 * @endcode
 *
 * Every ``pi`` is a one-element vector. The known input and output shapes have
 * equal rank and differ only at dimension ``k``; ``pk`` is retained while all
 * unchanged dimensions become ``1`` so broadcasting preserves them.
 */
class ShapeBasedConcatExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedConcatExpandPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedConcatExpand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` whose dynamic Concat target changes one dimension.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the target shape with ``[1]`` for every unchanged dimension.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes dynamic ``Expand`` nodes before a broadcasting binary operator.
 *
 * @code
 * Before:
 *              ┌────────┐
 *   a, sa ────→│ Expand │────→ ea
 *              └────────┘
 *
 *              ┌────────┐
 *   b, sb ────→│ Expand │────→ eb
 *              └────────┘
 *
 *              ┌────────┐
 *   ea, eb ───→│ Binary │────→ y
 *              └────────┘
 *
 * After:
 *            ┌────────┐
 *   a, b ───→│ Binary │────→ y
 *            └────────┘
 *
 * Either input may already be unexpanded.
 * @endcode
 *
 * The input, pre-expansion, and output symbolic shapes must prove that
 * broadcasting ``a`` and ``b`` produces the shape of ``y``. Either input may
 * already be unexpanded. Shared ``Expand`` nodes are preserved for other
 * consumers, but the rebuilt binary operator bypasses them.
 */
class ShapeBasedExpandBroadcastPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedExpandBroadcastPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedExpandBroadcast") {}

  /// Returns broadcasting binary operators as possible roots.
  std::set<std::string> FastOpType() const override;

  /// Finds a binary operator whose input Expand nodes are unnecessary.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the binary operator on the pre-expanded inputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes dynamic ``Expand`` nodes before ``MatMul``.
 *
 * @code
 * Before:
 *              ┌────────┐
 *   a, sa ────→│ Expand │────→ ea
 *              └────────┘
 *
 *              ┌────────┐
 *   b, sb ────→│ Expand │────→ eb
 *              └────────┘
 *
 *              ┌────────┐
 *   ea, eb ───→│ MatMul │────→ y
 *              └────────┘
 *
 * After:
 *            ┌────────┐
 *   a, b ───→│ MatMul │────→ y
 *            └────────┘
 *
 * Either input may already be unexpanded.
 * @endcode
 *
 * Either input may already be unexpanded. Only the leading batch dimensions
 * must broadcast to the output batch shape; the trailing two dimensions retain
 * standard ``MatMul`` semantics. Shared ``Expand`` nodes remain for other uses.
 */
class ShapeBasedExpandBroadcastMatMulPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedExpandBroadcastMatMulPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedExpandBroadcastMatMul") {}

  /// Returns ``MatMul`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a MatMul whose input Expand nodes are unnecessary.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds MatMul on the pre-expanded inputs.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a dynamic ``Expand`` target with an equivalent constant target.
 *
 * @code
 * Before:
 *                               ┌────────┐
 *   x, dynamic shape graph ────→│ Expand │────→ y
 *                               └────────┘
 *
 * After:
 *                               ┌────────┐
 *   x, constant [1,M,1] ───────→│ Expand │────→ y
 *                               └────────┘
 * @endcode
 *
 * Dimensions that are unchanged become ``1``; changed dimensions must be
 * statically known in both input and output and become their output size. The
 * ``Expand`` remains, but its dynamic shape computation is replaced.
 */
class ShapeBasedStaticExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedStaticExpandPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedStaticExpand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a dynamic Expand whose effective broadcast target is static.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds Expand with a constant broadcast target.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves input ``Expand`` nodes after a broadcasting binary operator.
 *
 * @code
 * Before:
 *             ┌────────┐
 *   a, S ────→│ Expand │────→ ea
 *             └────────┘
 *
 *             ┌────────┐
 *   b, S ────→│ Expand │────→ eb
 *             └────────┘
 *
 *              ┌────────┐
 *   ea, eb ───→│ Binary │────→ y
 *              └────────┘
 *
 * After:
 *             ┌────────┐
 *   a, b ────→│ Binary │────→ t
 *             └────────┘
 *
 *            ┌────────┐
 *   t, S ───→│ Expand │────→ y
 *            └────────┘
 *
 * Either input may already be unexpanded.
 * @endcode
 *
 * One input may already be unexpanded. Symbolic shapes must prove that the
 * smaller inputs broadcast safely and that the trailing ``Expand`` restores
 * the original output shape. Shared input ``Expand`` nodes remain for other
 * consumers.
 */
class ShapeBasedExpandSwapPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedExpandSwapPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedExpandSwap") {}

  /// Returns broadcasting binary operators as possible roots.
  std::set<std::string> FastOpType() const override;

  /// Finds a binary operator whose input expansions can be delayed.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the binary operator before one trailing Expand.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves an ``Expand`` after ``Cast`` and ``Where``.
 *
 * @code
 * Before:
 *             ┌────────┐                  ┌──────┐
 *   x, S ────→│ Expand │────→ e ───┬─────→│ Cast │────→ condition
 *             └────────┘           │      └──────┘
 *                                  │
 *                                  └────────────────────┐
 *                                                       │
 *                                                       ↓
 *                                                    ┌───────┐
 *   z ──────────────────────────────────────────────→│ Where │────→ y
 *                                                    └───────┘
 *
 * After:
 *          ┌──────┐
 *   x ────→│ Cast │────→ condition
 *          └──────┘
 *              │
 *              └──────────────────────────┐
 *                                         │
 *                                         ↓
 *                                     ┌───────┐
 *   z ───────────────────────────────→│ Where │────→ w
 *                                     └───────┘
 *                                         │
 *                                         ↓
 *                                    ┌────────┐
 *                                    │ Expand │←──── S
 *                                    └────────┘
 *                                         │
 *                                         ↓
 *                                         y
 *
 * The expanded value may occupy either Where branch.
 * @endcode
 *
 * The expanded value may instead be the false branch. ``Cast(e)`` must be the
 * unshared condition, and symbolic shapes must prove that the smaller
 * ``Where`` broadcasts to the original output shape.
 */
class ShapeBasedExpandCastWhereSwapPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ShapeBasedExpandCastWhereSwapPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedExpandCastWhereSwap") {}

  /// Returns ``Where`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an Expand/Cast/Where chain that can run before expansion.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits Cast and Where on the pre-expanded input, then expands the result.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves an ``Expand`` past a following unary-like operator.
 *
 * @code
 * Before:
 *             ┌────────┐
 *   x, S ────→│ Expand │────→ e
 *             └────────┘
 *                  │
 *                  ↓
 *             ┌──────────┐
 *             │ UnaryOp  │←──── optional inputs
 *             └──────────┘
 *                  │
 *                  ↓
 *                  y
 *
 * After:
 *          ┌──────────┐
 *   x ────→│ UnaryOp  │←──── optional inputs
 *          └──────────┘
 *               │
 *               ↓
 *          ┌────────┐
 *          │ Expand │←──── S
 *          └────────┘
 *               │
 *               ↓
 *               y
 * @endcode
 *
 * The ``Expand`` output must be unshared and ``Op`` must preserve the shape of
 * its first input. Attributes and any additional inputs of ``Op`` are retained,
 * so the operator runs on the smaller tensor.
 */
class ExpandSwapPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandSwapPattern(int priority = 0) : PatternOptimization(priority, "ExpandSwap") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single unary-like operator.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the unary operator first and re-expands its output.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Swaps an ``Expand`` and a following ``Unsqueeze``.
 *
 * @code
 * Before:
 *                 ┌────────┐
 *   x, [A,B] ────→│ Expand │────→ e
 *                 └────────┘
 *                     │
 *                     ↓
 *               ┌───────────┐
 *               │ Unsqueeze │←──── axes=[1]
 *               └───────────┘
 *                     │
 *                     ↓
 *                     y
 *
 * After:
 *                    ┌───────────┐
 *   x, axes=[1] ────→│ Unsqueeze │────→ u
 *                    └───────────┘
 *                          │
 *                          ↓
 *                     ┌────────┐
 *                     │ Expand │←──── [A,1,B]
 *                     └────────┘
 *                          │
 *                          ↓
 *                          y
 * @endcode
 *
 * The axes are constant (opset 13 or later), the ``Expand`` output is
 * unshared, and its target or output shape is known. The new target inserts a
 * ``1`` at every normalized axis, letting ``Unsqueeze`` run before expansion.
 */
class SwapExpandUnsqueezePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SwapExpandUnsqueezePattern(int priority = 0)
      : PatternOptimization(priority, "SwapExpandUnsqueeze") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``Expand`` consumed by a single ``Unsqueeze`` with constant axes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits the ``Unsqueeze`` first and re-expands its output.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses ``Expand``, ``Unsqueeze`` and ``Expand`` into ``Unsqueeze`` then
 * ``Expand``.
 *
 * @code
 * Before:
 *                 ┌────────┐
 *   x, [2,3] ────→│ Expand │────→ e1
 *                 └────────┘
 *                     │
 *                     ↓
 *               ┌───────────┐
 *               │ Unsqueeze │←──── axes=[1]
 *               └───────────┘
 *                     │
 *                     ↓
 *                 ┌────────┐
 *                 │ Expand │←──── [1,4,3]
 *                 └────────┘
 *                     │
 *                     ↓
 *                     y
 *
 * After:
 *                    ┌───────────┐
 *   x, axes=[1] ────→│ Unsqueeze │────→ u2
 *                    └───────────┘
 *                          │
 *                          ↓
 *                     ┌────────┐
 *                     │ Expand │←──── [2,4,3]
 *                     └────────┘
 *                          │
 *                          ↓
 *                          y
 * @endcode
 *
 * Both targets and the axes are constant (opset 13 or later), intermediate
 * outputs are unshared, and the ranks agree. The new target is
 * ``max(insert_ones(first_target, axes), second_target)`` element by element.
 */
class ExpandUnsqueezeExpandPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ExpandUnsqueezeExpandPattern(int priority = 0)
      : PatternOptimization(priority, "ExpandUnsqueezeExpand") {}

  /// Returns ``Expand`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds ``Expand`` then ``Unsqueeze`` then ``Expand`` with constant shapes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits ``Unsqueeze`` on the original tensor and a single trailing ``Expand``.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Swaps an ``Expand`` with a following constant-shape ``Reshape``.
 *
 * @code
 * Before:
 *                     ┌────────┐
 *   x, S=[D,1,1] ────→│ Expand │────→ e
 *                     └────────┘
 *                         │
 *                         ↓
 *                    ┌─────────┐
 *                    │ Reshape │←──── [0,1,-1]
 *                    └─────────┘
 *                         │
 *                         ↓
 *                         y
 *
 * After:
 *                    ┌─────────┐
 *   x, [0,1,-1] ────→│ Reshape │────→ r
 *                    └─────────┘
 *                         │
 *                         ↓
 *                    ┌────────┐
 *                    │ Expand │←──── S=[D,1,1]
 *                    └────────┘
 *                         │
 *                         ↓
 *                         y
 * @endcode
 *
 * ``x`` must have rank three, the ``Expand`` output must be unshared, and its
 * inferred target value must end in ``[1,1]``. This specialized form moves the
 * fixed ``Reshape`` before expansion.
 */
class SwapExpandReshapePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit SwapExpandReshapePattern(int priority = 0)
      : PatternOptimization(priority, "SwapExpandReshape") {}

  /// Returns ``Reshape`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds the supported Expand/Reshape rank-three form.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Emits Reshape before Expand.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
