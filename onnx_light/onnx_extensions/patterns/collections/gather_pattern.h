// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Simplifies ``Gather(Concat(..., X, ..., axis=0), cst)`` into
 * ``Gather(X, cst - offset)`` when exactly one ``Concat`` input ``X`` is
 * non-constant, all others are constants of known size, ``X`` is 1-D, and every
 * requested index falls inside ``X``'s slice of the concatenation.
 *
 * ``offset`` is the total size of the constant inputs preceding ``X``.
 *
 * @code
 * Before:
 *                  ┌──────────────┐
 *   c0, X, c1 ────→│ Concat axis0 │────→ t
 *                  └──────────────┘
 *                         │
 *                         ↓
 *                    ┌──────────────┐
 *                    │ Gather axis0 │←──── indices=[2,4]
 *                    └──────────────┘
 *                         │
 *                         ↓
 *                         y
 *
 * After:
 *                         ┌──────────────┐
 *   X, indices=[0,2] ────→│ Gather axis0 │────→ y
 *                         └──────────────┘
 * @endcode
 *
 * If ``t`` has another consumer, the original ``Concat`` is retained.
 */
class GatherConcatPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherConcatPattern(int priority = 0) : PatternOptimization(priority, "GatherConcat") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds an ``axis=0`` constant-index ``Gather`` over a ``Concat``.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the ``Gather`` against the single non-constant ``Concat`` input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a scalar or vector constant-index ``Gather`` upstream across one
 * compatible producer node, so that the producer runs on the smaller
 * (already gathered) tensor(s) instead of the original full-size ones.
 *
 * The producer is left in place and the pattern is applied again on its
 * (possibly further) inputs by the optimizer's fix-point loop, so a chain of
 * compatible producers is crossed one hop per iteration until an
 * unsupported, shared, or otherwise unsafe producer is reached.
 *
 * @code
 * Before:
 *   a [M,N,K]     b [N,K]
 *          \       /
 *           ┌─────┐
 *           │ Add │────→ t [M,N,K]
 *           └─────┘
 *              │
 *              ↓
 *      ┌──────────────┐
 *      │ Gather axis1 │←──── indices
 *      └──────────────┘
 *              │
 *              ↓
 *              y
 *
 * After:
 *   a [M,N,K]              b [N,K]
 *      │                       │
 *      ↓                       ↓
 * ┌──────────────┐     ┌──────────────┐
 * │ Gather axis1 │     │ Gather axis0 │←──── indices (both)
 * └──────────────┘     └──────────────┘
 *              \             /
 *               ┌─────┐
 *               │ Add │────→ y
 *               └─────┘
 * @endcode
 *
 * Supported producers (default ``ai.onnx`` domain only), one hop per match:
 *   - ``Add``, ``Div``, ``Cast``, ``Dropout``: every tensor input whose
 *     dimension at the (right-aligned) sliced axis is provably equal to the
 *     producer's own output dimension is gathered too; an input that does not
 *     reach that axis (a lower-rank, right-aligned broadcast input, e.g. a
 *     bias vector) is left untouched. An input with an actual broadcasting
 *     dimension of ``1`` at that axis is left untouched for a vector index,
 *     but the whole match is rejected for a scalar index, because removing
 *     the axis from the gathered inputs would silently shift that
 *     broadcasting input out of alignment (no rank-adaptation ``Squeeze`` is
 *     inserted).
 *   - ``Transpose``: only a rank-preserving (vector-index) ``Gather`` is
 *     propagated, remapping the axis through ``perm``. A scalar-index
 *     ``Gather`` directly downstream of a ``Transpose`` is intentionally left
 *     to :cpp:class:`TransposeGatherPattern`, which already rebuilds the
 *     ``perm`` for that rank-reducing case; duplicating that logic here would
 *     risk two patterns disagreeing on the same rewrite.
 *   - ``Reshape``: only propagated when the sliced axis can be traced back to
 *     exactly one input axis (matched left-aligned or right-aligned, as
 *     ``Reshape`` never redistributes elements across the boundary of an
 *     unrelated axis) and the ``shape`` constant either already spells ``0``
 *     (copy the input dimension) at that position, or, for a vector index,
 *     holds a concrete value that is rewritten to the new gathered length. A
 *     negative (``-1``, inferred) entry, or a scalar index paired with a
 *     non-``0`` entry, is rejected because the new dimension cannot be
 *     recomputed without re-deriving the whole reshape.
 *   - ``MatMul``: only a batch axis (strictly before the last two,
 *     contracted-adjacent dimensions) is propagated, using the same equal /
 *     broadcast / not-present rule as the pointwise operators, independently
 *     for each of the two inputs. The contracted dimension and the two
 *     matrix dimensions are never touched.
 *   - ``Softmax``, ``LayerNormalization``: only a rank-preserving
 *     (vector-index) ``Gather`` strictly before the operator's own reduction
 *     ``axis`` is propagated to the (sole) data input; the reduction ``axis``
 *     attribute is never rewritten, so its semantics are always preserved.
 *
 * The match is rejected whenever the producer's matched output is a graph
 * output, is captured by a nested subgraph, or feeds more than the candidate
 * ``Gather`` (:cpp:func:`GraphGraph::IsUsedMoreThanOnce`), or when the
 * producer has another output that is used at all -- resizing that producer
 * would otherwise silently change a value some other part of the graph still
 * relies on at the original size. The ``Gather`` index must be a materialised
 * scalar or 1-D integer constant; a negative index is only accepted when the
 * sliced dimension is statically known and the normalised index is in range.
 */
class GatherUpstreamPropagationPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherUpstreamPropagationPattern(int priority = 0)
      : PatternOptimization(priority, "GatherUpstreamPropagation") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a constant-index ``Gather`` fed by one compatible, unshared
  /// producer node.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Duplicates the ``Gather`` onto the producer's compatible inputs and
  /// rewrites the producer to output directly under the original ``Gather``
  /// name.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Composes two consecutive ``axis=0`` ``Gather`` nodes with constant indices
 * into a single ``Gather``.
 *
 * The inner indices ``cst1`` must be a 1-D constant array and the outer indices
 * ``cst2`` a constant integer tensor; the fused indices are ``cst1[cst2]``.
 * Scalar outer indices remain scalar, while non-scalar fused indices are
 * materialised as a vector.
 *
 * @code
 * Before:
 *   x, [4,1,7]
 *         │
 *         ↓
 *   ┌──────────────┐
 *   │ Gather axis0 │──┐
 *   └──────────────┘  │
 *                     │ t
 *                     ↓
 *                    ┌──────────────┐
 *                    │ Gather axis0 │←──── [2,0]
 *                    └──────────────┘
 *                           │
 *                           ↓
 *                           y
 *
 * After:
 *   x, [7,4]
 *       │
 *       ↓
 *   ┌──────────────┐
 *   │ Gather axis0 │────→ y
 *   └──────────────┘
 * @endcode
 *
 * If ``t`` has another consumer, the inner ``Gather`` is retained.
 */
class GatherGatherPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherGatherPattern(int priority = 0) : PatternOptimization(priority, "GatherGather") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds two consecutive constant-index ``axis=0`` ``Gather`` nodes.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the pair with a single ``Gather`` on the composed indices.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a ``Gather`` selecting a constant scalar index, a constant
 * single-element vector index, or a constant arithmetic-progression vector of
 * indices (a "range" with a fixed, strictly ascending step) by an equivalent
 * ``Slice``. ``Slice`` is generally cheaper to execute than ``Gather`` and,
 * unlike ``Gather``, it never triggers a data-dependent gather kernel.
 *
 * A negative ``axis`` is normalized against the rank of ``x`` and negative
 * index values are normalized against the static size of that axis; both
 * normalizations require the corresponding dimension to be statically known,
 * otherwise the rewrite is rejected. A non-unit step is rejected when the
 * default domain opset is known to predate opset 10 (the attribute-only
 * ``Slice`` form has no ``steps`` input). The ``Slice`` uses the input form
 * (with an ``axes`` initializer) from opset 10 onward and the attribute form
 * (``starts`` / ``ends`` / ``axes``) before that.
 *
 * A scalar index reduces the rank of its axis, so the replacement additionally
 * appends a ``Squeeze`` of that axis (input form from opset 13 onward,
 * attribute form before that) to preserve the original output rank. A vector
 * index, singleton or not, keeps the axis (with a size equal to the number of
 * selected elements), so ``Slice`` alone reproduces the ``Gather`` output.
 *
 * @code
 * Before (scalar index):
 *              ┌────────┐
 *   x, idx ────→│ Gather │────→ y
 *              └────────┘
 *
 * After:
 *                             ┌───────┐         ┌─────────┐
 *   x, [idx],[idx+1],[axis] ─→│ Slice │────────→│ Squeeze │────→ y
 *                             └───────┘  [axis]→└─────────┘
 *
 * Before (vector index, singleton or arithmetic progression):
 *              ┌────────┐
 *   x, idx ────→│ Gather │────→ y
 *              └────────┘
 *
 * After:
 *                                          ┌───────┐
 *   x, [start],[end],[axis],[step] ───────→│ Slice │────→ y
 *                                          └───────┘
 * @endcode
 */
class GatherToSlicePattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit GatherToSlicePattern(int priority = 0)
      : PatternOptimization(priority, "GatherToSlice") {}

  /// Returns ``Gather`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a ``Gather`` whose index is a safely representable constant.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rewrites the ``Gather`` into a ``Slice`` (plus ``Squeeze`` for a scalar
  /// index).
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
