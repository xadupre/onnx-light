// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses per-token expert selection into an opset-29 GroupedMatMul.
 *
 * @code
 * Before (batched Gather):
 *             ┌─────────────────────┐
 *   x[M,K] ──→│ Unsqueeze axes=1,2  │────┐
 *             └─────────────────────┘    │
 *             ┌─────────────────────┐    ↓
 *   W, I ────→│ Gather axis=0       │→┌────────┐
 *             └─────────────────────┘ │ MatMul │
 *                                     └────────┘
 *                                          ↓
 *                                 ┌────────────────┐
 *                                 │ Squeeze axis=2 │──→ y[M,S,N]
 *                                 └────────────────┘
 *
 * Before (unrolled per-token MatMuls, M >= 2):
 *                  ┌────────────────────────────────┐
 *   x, W, I, m ───→│ Gather rows and expert weights │
 *                  └────────────────────────────────┘
 *                                  ↓
 *                  ┌────────────────────────────────┐
 *                  │ MatMul, optional gathered bias │
 *                  └────────────────────────────────┘
 *                                  ↓
 *                  ┌────────────────────────────────┐
 *                  │ Unsqueeze(0), Concat(0)        │──→ y[M,S,N]
 *                  └────────────────────────────────┘
 *
 * Before (stateless, fixed-trip Loop):
 *                         ┌────────────────────────────────────┐
 *   M, true, x, W, I, B ─→│ Loop: Gather rows/weights by iter   │
 *                         │ MatMul, optional gathered bias      │
 *                         │ condition forwarded, scan [S,N]    │──→ y[M,S,N]
 *                         └────────────────────────────────────┘
 *
 * After (all three forms):
 *                         ┌────────────────────────┐
 *   x, W, I, optional B ─→│ GroupedMatMul          │──→ y[M,S,N]
 *                         └────────────────────────┘
 * @endcode
 *
 * W has shape [G,K,N] and I has shape [M,S]. G must be a positive static
 * dimension representable in the index dtype. Negative Gather indices are
 * normalized with Less, Where, and Add before GroupedMatMul. A following Add
 * of Gather(bias[G,N], I, axis=0) may be absorbed in either input order.
 * Removed intermediates must not be shared, exported, or captured.
 * Concat and Loop forms require fully static compatible shapes and positive M.
 * Concat branches must cover every token in order. The Loop condition is true
 * or omitted, and the body cannot terminate early, carry state, shadow captures,
 * contain extra operations, or reference nested subgraphs.
 * The default priority precedes GatherToSlice, which would obscure scalar rows.
 */
class GroupedMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit GroupedMatMulPattern(int priority = 2)
      : PatternOptimization(priority, "GroupedMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  core::builder::MatchResult MatchConcat(core::builder::GraphGraph &graph,
                                         const NodeProto &candidate) const;
  core::builder::MatchResult MatchLoop(core::builder::GraphGraph &graph,
                                       const NodeProto &candidate) const;
};

/**
 * Absorbs one or both matrix input Transposes into Gemm transpose flags.
 *
 * @code
 * Before:
 *          ┌─────────────────┐
 *   A ────→│ Transpose [1,0] │────→ At
 *          └─────────────────┘
 *          ┌─────────────────┐
 *   B ────→│ Transpose [1,0] │────→ Bt
 *          └─────────────────┘
 *                         ┌─────────────────┐
 *   At, Bt, optional C ──→│ Gemm transA/B   │────→ y
 *                         └─────────────────┘
 *
 * After:
 *                       ┌───────────────────────┐
 *   A, B, optional C ──→│ Gemm toggled transA/B │────→ y
 *                       └───────────────────────┘
 * @endcode
 *
 * At least one input must be produced by a rank-two Transpose with ``perm=[1,0]``
 * or the equivalent default permutation.
 * Only flags for absorbed sides are toggled. Inputs may be dynamic or constant;
 * alpha, beta, legacy broadcast, bias and output names are preserved. Shared Transposes are
 * retained, and on GPU a shared side is not absorbed.
 */
class GemmTransposePattern final : public core::builder::PatternOptimization {
public:
  explicit GemmTransposePattern(int priority = 1)
      : PatternOptimization(priority, "GemmTranspose") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a two-input Sum following an unbiased Gemm into the Gemm bias input.
 *
 * The Gemm output must be private to the Sum, and the other Sum input must be
 * unidirectionally broadcastable to the rank-two Gemm output.
 *
 * @code
 * Before:
 *          +------+       +-----+
 *   A, B ->| Gemm |------>| Sum |---> y
 *          +------+       +-----+
 *                            ^
 *                            |
 *                            C
 *
 * After:
 *             +--------------+
 *   A, B, C ->| Gemm(beta=1) |---> y
 *             +--------------+
 * @endcode
 */
class GemmSumFusionPattern final : public core::builder::PatternOptimization {
public:
  explicit GemmSumFusionPattern(int priority = 2)
      : PatternOptimization(priority, "GemmSumFusion") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses an Add bias into a MatMul or Gemm.
 *
 * @code
 * Before:
 *                          ┌─────────────┐
 *    A, B, optional C ────→│ MatMul/Gemm │
 *                          └─────────────┘
 *                                 │
 *                                 ↓
 *                             ┌─────┐
 *    bias ───────────────────→│ Add │────→ y
 *                             └─────┘
 *
 * After (rank-two A):
 *                       ┌─────┐
 *    optional C, bias ─→│ Add │
 *                       └─────┘
 *                          │ combined bias
 *                          ↓
 *                      ┌──────┐
 *    A, B ────────────→│ Gemm │────→ y
 *                      └──────┘
 *
 * Without C, bias feeds the boxed Gemm directly.
 *
 * After (higher-rank A):
 *                                              B
 *                                              │
 *                                              ↓
 *              ┌───────────────────┐           │
 *    A ───────→│ Reshape to rank 2 │──┐        │
 *              └───────────────────┘  │        │
 *                                     │        │
 *              ┌───────────────────┐  │    ┌──────┐
 *    bias ────→│ Reshape if needed │──┴───→│ Gemm │────→ t
 *              └───────────────────┘       └──────┘
 *                                              │
 *                                              ↓
 *                                           ┌───────────────────────┐
 *    shape ────────────────────────────────→│ Reshape to final rank │────→ y
 *                                           └───────────────────────┘
 * @endcode
 *
 * The bias last dimension must match the output width. For a higher-rank left
 * input, when enabled, the rewrite flattens it to rank two, applies ``Gemm``,
 * and restores the output shape; a higher-rank bias is flattened as needed.
 */
class MatMulAddPattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulAddPattern(int priority = 2, bool allow_reshape = false)
      : PatternOptimization(priority, "MatMulAdd"), allow_reshape_(allow_reshape) {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  bool allow_reshape_;
};

/**
 * Normalizes adjacent Reshapes around a MatMul to the final batch shape.
 *
 * @code
 * Before:
 *                  ┌──────────┐
 *    A ───────────→│ Reshape? │──┐
 *                  └──────────┘  │
 *                                │
 *                  ┌──────────┐  │    ┌────────────────────┐
 *    B ───────────→│ Reshape? │──┴───→│ MatMul/FusedMatMul │────→ m
 *                  └──────────┘       └────────────────────┘
 *                                                 │
 *                                                 ↓
 *                                            ┌──────────┐
 *                                            │ Reshape? │────→ y
 *                                            └──────────┘
 *
 * At least two of the three optional Reshape nodes are present.
 *
 * After:
 *                  ┌───────────────────┐
 *    A ───────────→│ Reshape if needed │──┐
 *                  └───────────────────┘  │
 *                                         │
 *                  ┌───────────────────┐  │    ┌────────────────────┐
 *    B ───────────→│ Reshape if needed │──┴───→│ MatMul/FusedMatMul │────→ y
 *                  └───────────────────┘       └────────────────────┘
 *
 * A boxed restoring Reshape is emitted when the intermediate MatMul shape is
 * still required.
 * @endcode
 *
 * All participating shapes must be static, preserve element counts and the
 * last two matrix dimensions, and describe equal batch element counts. Shared
 * pre-Reshapes are retained for their other consumers. If the old MatMul
 * result also needs its intermediate shape, a restoring Reshape is emitted.
 */
class MatMulReshape2Of3Pattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulReshape2Of3Pattern(int priority = 1)
      : PatternOptimization(priority, "MatMulReshape2Of3") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves scalar factors from both MatMul inputs to one multiplication after the MatMul.
 *
 * @code
 * Before:
 *                  ┌─────┐
 *    x, c1 ───────→│ Mul │──┐
 *                  └─────┘  │
 *                           │
 *                  ┌─────┐  │    ┌────────┐
 *    z, c2 ───────→│ Mul │──┴───→│ MatMul │────→ y
 *                  └─────┘       └────────┘
 *
 * After:
 *             ┌────────┐
 *   x, z ────→│ MatMul │────→ t
 *             └────────┘
 *
 *                          ┌─────┐
 *   t, combined scalar ───→│ Mul │────→ y
 *                          └─────┘
 * @endcode
 *
 * Each input Mul must contain exactly one scalar constant and have an unshared
 * output. The constants are multiplied into one initializer.
 */
class MulMulMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit MulMulMatMulPattern(int priority = 1) : PatternOptimization(priority, "MulMulMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Folds inference BatchNormalization parameters into a constant MatMul weight.
 *
 * Both MatMul inputs and the BatchNormalization input/output must be rank two.
 * The MatMul weight and all four BatchNormalization parameters must be
 * floating-point constants. The replacement is a standard Gemm with folded
 * weight and bias initializers.
 *
 * @code
 * Before:
 *          +--------+       +--------------------+
 *   x, W ->| MatMul |------>| BatchNormalization |---> y
 *          +--------+       +--------------------+
 *                              ^    ^    ^    ^
 *                              |    |    |    |
 *                           scale  bias mean variance
 *
 * After:
 *                              +------+
 *   x, folded_W, folded_bias ->| Gemm |---> y
 *                              +------+
 * @endcode
 */
class MatMulBatchNormalizationFusionPattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulBatchNormalizationFusionPattern(int priority = 2)
      : PatternOptimization(priority, "MatMulBatchNormalizationFusion") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Absorbs one scalar Mul or safe Div adjacent to a rank-two MatMul.
 *
 * A constant matrix operand is folded when available. Otherwise the MatMul is
 * replaced by a standard Gemm whose ``alpha`` attribute carries the scalar.
 * Divisions with the scalar in the numerator and zero or non-finite factors
 * are rejected.
 *
 * @code
 * Before:
 *          +--------+       +-----+
 *   x, W ->| MatMul |------>| Mul |---> y
 *          +--------+       +-----+
 *                            ^
 *                            |
 *                          scale
 *
 * After:
 *                  +--------+
 *   x, folded_W -->| MatMul |---> y
 *                  +--------+
 * @endcode
 */
class MatMulScaleFusionPattern final : public core::builder::PatternOptimization {
public:
  explicit MatMulScaleFusionPattern(int priority = 2)
      : PatternOptimization(priority, "MatMulScaleFusion") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes batch-flattening Reshapes around a MatMul.
 *
 * @code
 * Before:
 *                  ┌───────────────────┐
 *    A ───────────→│ Reshape to rank 3 │──┐
 *                  └───────────────────┘  │
 *                                         │
 *                  ┌───────────────────┐  │    ┌────────┐
 *    B ───────────→│ Reshape to rank 3 │──┴───→│ MatMul │────→ m
 *                  └───────────────────┘       └────────┘
 *                                                   │
 *                                                   ↓
 *                                           ┌──────────────────────────┐
 *                                           │ Reshape to original rank │────→ y
 *                                           └──────────────────────────┘
 *
 * After:
 *   A ─────────────────────┐
 *                          │
 *                          ↓
 *                      ┌────────┐
 *                      │ MatMul │────→ y
 *                      └────────┘
 *                          ↑
 *   B ─────────────────────┘
 * @endcode
 *
 * The original inputs and final output have equal rank of at least four.
 * Reshapes must preserve element counts and matrix suffixes, and the original
 * batch prefixes must broadcast directly to the final shape. Shared input
 * Reshapes are retained for their other consumers.
 */
class ReshapeMatMulReshapePattern final : public core::builder::PatternOptimization {
public:
  explicit ReshapeMatMulReshapePattern(int priority = 1)
      : PatternOptimization(priority, "ReshapeMatMulReshape") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a MatMul with unit reduction dimensions by element-wise Mul.
 *
 * @code
 * Before:
 *             ┌───────────────────────┐
 *   A, B ────→│ MatMul unit reduction │────→ m
 *             └───────────────────────┘
 *                   │
 *                   ↓
 *           ┌──────────────────────┐
 *           │ Optional Transpose   │────→ y
 *           │ swapping last 2 axes │
 *           └──────────────────────┘
 *
 * After (no Transpose):
 *             ┌─────┐
 *   A, B ────→│ Mul │────→ y
 *             └─────┘
 *
 * After (last-two Transpose present):
 *                  ┌──────────────────┐
 *    A ───────────→│ Reshape ...,1,-1 │──┐
 *                  └──────────────────┘  │
 *                                        │
 *                  ┌──────────────────┐  │    ┌─────┐
 *    B ───────────→│ Reshape ...,-1,1 │──┴───→│ Mul │────→ y
 *                  └──────────────────┘       └─────┘
 * @endcode
 *
 * The last dimension of ``A`` and penultimate dimension of ``B`` must both be
 * one. A following last-two-dimension Transpose is removed only when both input
 * ranks exceed two.
 */
class ShapeBasedMatMulToMulPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedMatMulToMulPattern(int priority = 1)
      : PatternOptimization(priority, "ShapeBasedMatMulToMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves an element-wise activation before a Reshape or Transpose.
 *
 * @code
 * Before:
 *             ┌─────────────┐
 *   A, B ────→│ MatMul/Gemm │────→ m
 *             └─────────────┘
 *                   │
 *                   ↓
 *           ┌───────────────────┐
 *           │ Reshape/Transpose │────→ l
 *           └───────────────────┘
 *                   │
 *                   ↓
 *              ┌────────────┐
 *              │ Activation │────→ y
 *              └────────────┘
 *
 * After:
 *             ┌─────────────┐
 *   A, B ────→│ MatMul/Gemm │────→ m
 *             └─────────────┘
 *                   │
 *                   ↓
 *              ┌────────────┐
 *              │ Activation │────→ a
 *              └────────────┘
 *                   │
 *                   ↓
 *           ┌───────────────────┐
 *           │ Reshape/Transpose │────→ y
 *           └───────────────────┘
 * @endcode
 *
 * ``Layout`` is a ``Reshape`` or ``Transpose``. Its input and output must be
 * unshared, so the activation can commute across the layout without changing
 * other consumers.
 */
class SwitchReshapeActivationPattern final : public core::builder::PatternOptimization {
public:
  explicit SwitchReshapeActivationPattern(int priority = 1)
      : PatternOptimization(priority, "SwitchReshapeActivation") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Absorbs rank-two input Transposes into Gemm transpose attributes.
 *
 * @code
 * Before:
 *          ┌────────────────────────┐
 *   A ────→│ Optional Transpose 1,0 │────→ At
 *          └────────────────────────┘
 *
 *          ┌────────────────────────┐
 *   B ────→│ Optional Transpose 1,0 │────→ Bt
 *          └────────────────────────┘
 *
 *                                   ┌─────────────┐
 *   A or At, B or Bt, optional C ──→│ MatMul/Gemm │────→ y
 *                                   └─────────────┘
 *
 * At least one boxed Transpose is present.
 *
 * After:
 *                         ┌────────────────────────┐
 *   A, B, optional C ────→│ Gemm toggled transA/B  │────→ y
 *                         └────────────────────────┘
 * @endcode
 *
 * At least one input must be transposed. Existing Gemm flags are toggled for
 * the absorbed sides; shared Transposes are retained, and on GPU a shared side
 * is not absorbed.
 */
class TransposeMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeMatMulPattern(int priority = 1)
      : PatternOptimization(priority, "TransposeMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Swaps a last-two-dimension Transpose with a following Reshape on one MatMul input.
 *
 * @code
 * Before:
 *          ┌───────────────────────┐
 *   A ────→│ Transpose last 2 axes │────→ t
 *          └───────────────────────┘
 *                     │
 *                     ↓
 *                ┌─────────┐
 *                │ Reshape │←──── matrix-preserving target
 *                └─────────┘
 *                     │
 *                     └───────────────────────┐
 *                                             │
 *                                             ↓
 *                                          ┌────────┐
 *   B ────────────────────────────────────→│ MatMul │────→ y
 *                                          └────────┘
 *
 * After:
 *          ┌────────────────────────┐
 *   A ────→│ Reshape swapped target │────→ r
 *          └────────────────────────┘
 *                     │
 *                     ↓
 *              ┌───────────────────────┐
 *              │ Transpose last 2 axes │
 *              └───────────────────────┘
 *                     │
 *                     └───────────────────────┐
 *                                             │
 *                                             ↓
 *                                          ┌────────┐
 *   B ────────────────────────────────────→│ MatMul │────→ y
 *                                          └────────┘
 *
 * The same rewrite may be applied to the right MatMul input.
 * @endcode
 *
 * The selected Reshape target must be constant and preserve the transposed
 * matrix suffix. Both intermediate outputs must be unshared. The same rewrite
 * may be applied to the right input instead.
 */
class TransposeReshapeMatMulPattern final : public core::builder::PatternOptimization {
public:
  explicit TransposeReshapeMatMulPattern(int priority = 1)
      : PatternOptimization(priority, "TransposeReshapeMatMul") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
