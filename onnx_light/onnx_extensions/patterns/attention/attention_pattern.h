// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Replaces a half-rotary local function and doubled caches with RotaryEmbedding.
 *
 * @code
 * Before:
 *               +--------+
 *   cos, cos -->| Concat |----> doubled cosine
 *               +--------+
 *               +--------+
 *   sin, sin -->| Concat |----> doubled sine
 *               +--------+
 *
 *                                     +---------------------+
 *   x, doubled cosine, doubled sine ->| HalfRotaryEmbedding |----> rotated
 *                                     +---------------------+
 *
 *   Optional partial rotation:
 *               +-------+
 *   full x ---->| Split |----> rotary part, tail
 *               +-------+
 *                                               +---------------------+
 *   rotary part, doubled cosine, doubled sine ->| HalfRotaryEmbedding |----> rotated part
 *                                               +---------------------+
 *                               +--------+
 *   rotated part, tail -------->| Concat |----> y
 *                               +--------+
 *
 * After:
 *            +-------+       +--------+
 *   x ------>| Shape |------>| Concat |----> cache shape
 *            +-------+       +--------+
 *                                ^
 *                                |
 *                              one, one
 *
 *          +---------+       +--------+
 *   cos -->| Squeeze |------>| Expand |----> expanded cosine
 *          +---------+       +--------+
 *                                 ^
 *                                 |
 *                            cache shape
 *
 *          +---------+       +--------+
 *   sin -->| Squeeze |------>| Expand |----> expanded sine
 *          +---------+       +--------+
 *                                 ^
 *                                 |
 *                            cache shape
 *
 *                                          +-----------------+
 *   x, expanded cosine, expanded sine ---->| RotaryEmbedding |----> y
 *                                          +-----------------+
 * @endcode
 *
 * The rewrite requires opset 23, rank-four inputs, equal ``[*,1,*,*]``
 * caches doubled on the last axis, and a static head count. A partial rotation
 * additionally sets ``rotary_embedding_dim`` from the first Split size.
 */
class RotaryEmbeddingPattern final : public core::builder::PatternOptimization {
public:
  explicit RotaryEmbeddingPattern(int priority = 1)
      : PatternOptimization(priority, "RotaryEmbedding") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Collapses two zero-padded rotary branches into Split, Neg, and Concat.
 *
 * @code
 * Before:
 *   Concat form:
 *               +-------+
 *   x, sizes -> | Split | ---> first, second
 *               +-------+
 *
 *                  +--------+
 *   first, zero -> | Concat | ---> padded first
 *                  +--------+
 *
 *             +-----+
 *   second -> | Neg | ---> negated second
 *             +-----+
 *
 *                           +--------+
 *   zero, negated second -> | Concat | ---> padded second
 *                           +--------+
 *
 *                                  +-----+
 *   padded first, padded second -> | Add | ---> y
 *                                  +-----+
 *
 *   ScatterND form:
 *               +-------+
 *   x, sizes -> | Split | ---> first, second
 *               +-------+
 *
 *             +-----------+
 *   zero A -> | Transpose | ---> data A
 *             +-----------+
 *
 *            +-----------+
 *   first -> | Transpose | ---> updates A
 *            +-----------+
 *
 *                                   +-----------+
 *   data A, indices A, updates A -> | ScatterND | ---> scattered A
 *                                   +-----------+
 *
 *                  +-----------+
 *   scattered A -> | Transpose | ---> branch A
 *                  +-----------+
 *
 *             +-----------+
 *   zero B -> | Transpose | ---> data B
 *             +-----------+
 *
 *             +-----+
 *   second -> | Neg | ---> negated second
 *             +-----+
 *
 *                          +-----------+
 *   negated second ------> | Transpose | ---> updates B
 *                          +-----------+
 *
 *                                   +-----------+
 *   data B, indices B, updates B -> | ScatterND | ---> scattered B
 *                                   +-----------+
 *
 *                  +-----------+
 *   scattered B -> | Transpose | ---> branch B
 *                  +-----------+
 *
 *                         +-----+
 *   branch A, branch B -> | Add | ---> y
 *                         +-----+
 *
 * After:
 *               +-------+
 *   x, sizes -> | Split | ---> first, second
 *               +-------+
 *
 *   Second part negated:
 *             +-----+
 *   second -> | Neg | -> negated second
 *             +-----+
 *                                      +--------+
 *   first, negated second -----------> | Concat | ---> y
 *                                      +--------+
 *
 *   First part negated:
 *            +-----+
 *   first -> | Neg | -> negated first
 *            +-----+
 *                                     +--------+
 *   negated first, second ----------> | Concat | ---> y
 *                                     +--------+
 * @endcode
 *
 * A pair of contiguous Slice nodes may provide ``first`` and ``second`` in
 * either input form; Apply then emits a Split. The two parts cover the source
 * axis, and existing Split parts are equal in the ScatterND form. Padding is
 * exactly zero, branch placements or indices are complementary, and removed
 * branch results must be unshared.
 */
class RotaryConcatPartPattern final : public core::builder::PatternOptimization {
public:
  explicit RotaryConcatPartPattern(int priority = 1)
      : PatternOptimization(priority, "RotaryConcatPart") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a causal-mask index construction into a local function.
 *
 * @code
 * Before:
 *         +---------+
 *   A --->| Squeeze |----> start
 *         +---------+
 *         +---------+
 *   B --->| Squeeze |----> end
 *         +---------+
 *
 *                    +-------+       +-----------+
 *   zero, end, one ->| Range |------>| Unsqueeze |----> row indices
 *                    +-------+       +-----------+
 *
 *                     +-------+       +-----------+
 *   start, end, one ->| Range |------>| Unsqueeze |----> column indices
 *                     +-------+       +-----------+
 *
 *                                 +-------------+
 *   row indices, column indices ->| LessOrEqual |----> mask
 *                                 +-------------+
 *
 *   Shifted form:
 *                           +-----+
 *   column indices, shift ->| Sub |----> shifted columns
 *                           +-----+
 *                                  +---------+
 *   row indices, shifted columns ->| Greater |----> mask
 *                                  +---------+
 *
 * After:
 *            +-----------------------------+
 *   A, B --->| intermediate::CausalMask    |----> mask
 *            +-----------------------------+
 *
 *                   +---------------------------------+
 *   A, B, shift --->| intermediate::ShiftedCausalMask |----> mask
 *                   +---------------------------------+
 * @endcode
 *
 * The Range nodes use unit steps. The row and column Unsqueeze axes are
 * respectively 0, 1, 2 and 0, 1, 3. Opset 13 or newer is required; shared
 * upstream construction nodes are preserved.
 */
class FunctionCausalMaskPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCausalMaskPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCausalMask") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses an additive causal-mask index construction into a local function.
 *
 * @code
 * Before:
 *         +---------+       +-------+       +-----------+
 *   A --->| Squeeze |------>| Range |------>| Unsqueeze |----> first indices
 *         +---------+       +-------+       +-----------+
 *
 *         +---------+       +-------+       +-----------+       +-----+
 *   B --->| Squeeze |------>| Range |------>| Unsqueeze |------>| Mul |----> scaled
 *         +---------+       +-------+       +-----------+       +-----+
 *                                                                  ^
 *                                                                  |
 *                                                                  C
 *
 *                           +-----+
 *   first indices, scaled ->| Add |----> mask
 *                           +-----+
 *
 * After:
 *               +--------------------------------+
 *   A, B, C --->| intermediate::CausalMaskMulAdd |----> mask
 *               +--------------------------------+
 * @endcode
 *
 * Both ranges must be zero-based with unit steps. The first and second
 * Unsqueeze axes are respectively 0, 1, 2 and 1, 2, 3. All removed
 * intermediates must be unshared, and opset 13 is required.
 */
class FunctionCausalMaskMulAddPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCausalMaskMulAddPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCausalMaskMulAdd") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses cosine and sine rotary-cache generation into a local function.
 *
 * @code
 * Before:
 *   position ids ----------------------------------------------------------> ids
 *
 *            +---------+
 *   dim 1 -->| Squeeze |----> range start
 *            +---------+
 *            +---------+
 *   dim 2 -->| Squeeze |----> range end
 *            +---------+
 *                                 +-------+
 *   range start, range end, one ->| Range |----> ids
 *                                 +-------+
 *
 *          +------+       +-----------+       +---------+
 *   ids -->| Cast |------>| Unsqueeze |------>| Reshape |----> shaped ids
 *          +------+       +-----------+       +---------+
 *
 *   Alternate ordering:
 *          +-----------+       +------+       +---------+
 *   ids -->| Unsqueeze |------>| Cast |------>| Reshape |----> shaped ids
 *          +-----------+       +------+       +---------+
 *
 *                         +-----+
 *   weights, shaped ids ->| Mul |----> angle
 *                         +-----+
 *              +-----+
 *   angle ---->| Cos |----> raw cosine
 *              +-----+
 *                  +------+
 *   raw cosine --->| Cast |----> cosine
 *                  +------+
 *              +-----+
 *   angle ---->| Sin |----> raw sine
 *              +-----+
 *                +------+
 *   raw sine --->| Cast |----> sine
 *                +------+
 *
 * After:
 *                           +---------------------------+
 *   position ids, weights ->| intermediate::CosSinCache |----> cosine, sine
 *                           +---------------------------+
 *
 *                                   +------------------------------------+
 *   dim 1, dim 2, weights --------->| intermediate::CosSinCacheWithRange |---> cosine, sine
 *                                   +------------------------------------+
 * @endcode
 *
 * The output Cast boxes are optional but must either both appear with the
 * same target or both be absent. Weights and the
 * position Cast are float, the reshape target is exactly ``[0,-1,1]``, and
 * removed intermediates must be unshared. Direct ids use Unsqueeze axis 1;
 * ranged ids use axes ``[0,1]`` or ``[1]``.
 */
class FunctionCosSinCachePattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionCosSinCachePattern(int priority = 1)
      : PatternOptimization(priority, "FunctionCosSinCache") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses the canonical half-rotary decomposition into a local function.
 *
 * @code
 * Before:
 *          +-------+
 *   X ---->| Split |----> left, right
 *          +-------+
 *              +-----+
 *   right ---->| Neg |----> negated right
 *              +-----+
 *                                +--------+
 *   negated right, left -------->| Concat |----> rotated halves
 *                                +--------+
 *
 *                     +-----+
 *   X, cosine cache ->| Mul |----> cosine term
 *                     +-----+
 *                                  +-----+
 *   rotated halves, sine cache --->| Mul |----> sine term
 *                                  +-----+
 *                            +-----+
 *   cosine term, sine term ->| Add |----> Y
 *                            +-----+
 *
 * After:
 *                                  +-----------------------------------+
 *   X, cosine cache, sine cache -->| intermediate::HalfRotaryEmbedding |----> Y
 *                                  +-----------------------------------+
 * @endcode
 *
 * ``X`` must be rank four, opset 18 or newer is required, and the Split and
 * Concat use the last axis. Every removed intermediate must be unshared.
 */
class FunctionHalfRotaryEmbeddingPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionHalfRotaryEmbeddingPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionHalfRotaryEmbedding") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a masked scaled-dot-product attention graph into a local function.
 *
 * @code
 * Before:
 *               +-----+
 *   Q, scale -->| Mul |----> scaled query
 *               +-----+
 *               +-----+       +-----------+
 *   K, scale -->| Mul |------>| Transpose |----> transposed keys
 *               +-----+       +-----------+
 *                                         +--------+
 *   scaled query, transposed keys ------->| MatMul |----> scores
 *                                         +--------+
 *
 *   Alternate score path:
 *                               +-------------+
 *   scaled query, scaled keys ->| FusedMatMul |----> scores
 *                               +-------------+
 *
 *                                  +-------+
 *   mask, scores, minus infinity ->| Where |----> masked scores
 *                                  +-------+
 *
 *   Alternate additive mask:
 *                                +-------+
 *   mask, zero, minus infinity ->| Where |----> additive mask
 *                                +-------+
 *                           +-----+
 *   scores, additive mask ->| Add |----> masked scores
 *                           +-----+
 *
 *                    +---------+
 *   masked scores -->| Softmax |----> probabilities
 *                    +---------+
 *                         +-------+
 *   probabilities ------->| IsNaN |----> invalid
 *                         +-------+
 *                                   +-------+
 *   invalid, zero, probabilities -->| Where |----> clean probabilities
 *                                   +-------+
 *                              +--------+
 *   clean probabilities, V --->| MatMul |----> Y
 *                              +--------+
 *
 *   Optional GQA key branch:
 *          +-----------+       +-----+       +--------+       +---------+
 *   K ---->| Unsqueeze |------>| Mul |------>| Expand |------>| Reshape |----> repeated keys
 *          +-----------+       +-----+       +--------+       +---------+
 *
 *   Optional GQA value branch:
 *          +-----------+       +--------+       +---------+
 *   V ---->| Unsqueeze |------>| Expand |------>| Reshape |----> repeated values
 *          +-----------+       +--------+       +---------+
 *
 * After:
 *   Optional rank-three preparation:
 *          +---------+       +-----------+
 *   Q ---->| Reshape |------>| Transpose |----> prepared query
 *          +---------+       +-----------+
 *          +---------+       +-----------+
 *   K ---->| Reshape |------>| Transpose |----> prepared keys
 *          +---------+       +-----------+
 *
 *                                                      +---------------------------------+
 *   query, keys, V, mask, scale, optional GQA shapes ->| intermediate::LocalAttention... |---> Y
 *                                                      +---------------------------------+
 * @endcode
 *
 * Query and keys are the original tensors for rank-four input and the prepared
 * tensors for rank-three input. Rank-three inputs use matching Reshape targets
 * before their Transpose nodes. The
 * rank-four key Transpose uses permutation 0, 1, 3, 2; FusedMatMul must
 * transpose only B. Scales must be equal floating scalars, GQA key/value
 * repeat shapes must match, and all removed results are unshared. Direct
 * masking may place negative infinity on either Where value branch. Concrete
 * local-function names encode GQA, mask orientation, transpose form, and type.
 */
class FunctionAttentionPattern : public core::builder::PatternOptimization {
public:
  explicit FunctionAttentionPattern(int priority = 0)
      : PatternOptimization(priority, "FunctionAttention") {}
  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves matching GQA repeat-interleave branches into a LocalAttention function.
 *
 * @code
 * Before:
 *          +-----------+       +--------+       +--------------------+
 *   K ---->| Unsqueeze |------>| Expand |------>| Reshape or Squeeze |----> repeated keys
 *          +-----------+       +--------+       +--------------------+
 *          +-----------+       +--------+       +--------------------+
 *   V ---->| Unsqueeze |------>| Expand |------>| Reshape or Squeeze |----> repeated values
 *          +-----------+       +--------+       +--------------------+
 *
 *                                                    +------------------------------+
 *   Q, repeated keys, repeated values, mask, scale ->| intermediate::LocalAttention |----> Y
 *                                                    +------------------------------+
 *
 * After:
 *                                                   +---------------------------------+
 *   Q, K, V, mask, scale, expand shape, GQA shape ->| intermediate::LocalAttentionGQA |---> Y
 *                                                   +---------------------------------+
 * @endcode
 *
 * Key and value branches must use identical constant shapes, rank-five Expand
 * targets with singleton non-repeat dimensions, and the same final operation.
 * Inputs must be floating and every removed repeat intermediate unshared. The
 * concrete local-function name also records Reshape versus Squeeze and retains
 * the original attention suffix.
 */
class FunctionAttentionGQAPattern final : public core::builder::PatternOptimization {
public:
  explicit FunctionAttentionGQAPattern(int priority = 1)
      : PatternOptimization(priority, "FunctionAttentionGQA") {}
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Converts cached grouped-query attention into ONNX Attention cache inputs and outputs.
 *
 * @code
 * Before:
 *   ONNX decomposition:
 *               +--------+       +-----------+       +--------+
 *   past K, K ->| Concat |------>| Unsqueeze |------>| Expand |----+
 *               +--------+       +-----------+       +--------+    |
 *                                                                  v
 *                                                        +--------------------+
 *                                                        | Reshape or Squeeze |---> repeated K
 *                                                        +--------------------+
 *
 *               +--------+       +-----------+       +--------+
 *   past V, V ->| Concat |------>| Unsqueeze |------>| Expand |----+
 *               +--------+       +-----------+       +--------+    |
 *                                                                  v
 *                                                        +--------------------+
 *                                                        | Reshape or Squeeze |---> repeated V
 *                                                        +--------------------+
 *
 *                                         +-----------+
 *   Q, repeated K, repeated V, mask ----->| Attention |----> Y
 *                                         +-----------+
 *
 *   Local-function form:
 *               +--------+
 *   past K, K ->| Concat |----> cached K
 *               +--------+
 *               +--------+
 *   past V, V ->| Concat |----> cached V
 *               +--------+
 *                                              +---------------------------------+
 *   Q, cached K, cached V, mask, square root ->| intermediate::LocalAttentionGQA |---> Y
 *                                              +---------------------------------+
 *
 * After:
 *            +-----+
 *   mask --->| Not |----> final mask
 *            +-----+
 *
 *                                         +-----------+
 *   Q, K, V, final mask, past K, past V ->| Attention |----> Y, present K, present V
 *                                         +-----------+
 * @endcode
 *
 * The caches and current tensors must be compatible rank-four values, and
 * both cache Concat nodes use normalized axis 2. Opset 23 is required. The
 * Not box appears only for switched local masks, which must be bool. The
 * local ``scale_sqrt`` is squared for the Attention ``scale`` attribute.
 */
class AttentionGQAPattern final : public core::builder::PatternOptimization {
public:
  explicit AttentionGQAPattern(int priority = 2) : PatternOptimization(priority, "AttentionGQA") {}
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
