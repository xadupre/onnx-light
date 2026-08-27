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
 *   Full rotation:
 *
 *     x        cos            sin
 *     |         |              |
 *     |         +----+         +----+
 *     |         |    |         |    |
 *     |         v    v         v    v
 *     |       +--------+     +--------+
 *     |       | Concat |     | Concat |
 *     |       +--------+     +--------+
 *     |          |               |
 *     v          v               v
 *   +-------------------------------+
 *   |      HalfRotaryEmbedding      |
 *   +-------------------------------+
 *                   |
 *                   v
 *                   y
 *
 *   Partial rotation:
 *
 *     full x     sizes
 *       |          |
 *       v          v
 *     +--------------+
 *     |    Split     |
 *     +--------------+
 *        |      |
 *        |      +------ tail -------------------------+
 *        |                                            |
 *        | first                                      |
 *        |                                            |
 *        |         cos            sin                 |
 *        |          |              |                  |
 *        |          +----+         +----+             |
 *        |          |    |         |    |             |
 *        |          v    v         v    v             |
 *        |        +--------+     +--------+           |
 *        |        | Concat |     | Concat |           |
 *        |        +--------+     +--------+           |
 *        |             |              |               |
 *        v             v              v               |
 *      +---------------------------------------+      |
 *      |          HalfRotaryEmbedding          |      |
 *      +---------------------------------------+      |
 *                         |                           |
 *                         | rotated part              |
 *                         v                           v
 *                      +---------------------------------+
 *                      |             Concat              |
 *                      +---------------------------------+
 *                                       |
 *                                       v
 *                                       y
 *
 * After:
 *   ``x`` is the rotary input of the full rotation and the Split input of the
 *   partial rotation.
 *
 *   x
 *   |
 *   +-------+
 *   |       |
 *   |       v
 *   |   +-------+
 *   |   | Shape |
 *   |   +-------+
 *   |       |
 *   |       | batch    ones
 *   |       |            |
 *   |       v            v
 *   |   +------------------+
 *   |   |      Concat      |
 *   |   +------------------+
 *   |           |
 *   |           | cache shape
 *   |           +---------+---------------------------+
 *   |                     |                           |
 *   |   cos   one         |         sin   one         |
 *   |    |     |          |          |     |          |
 *   |    v     v          |          v     v          |
 *   |   +-----------+     |         +-----------+     |
 *   |   |  Squeeze  |     |         |  Squeeze  |     |
 *   |   +-----------+     |         +-----------+     |
 *   |         |           |               |           |
 *   |         v           v               v           v
 *   |     +--------------------+      +--------------------+
 *   |     |       Expand       |      |       Expand       |
 *   |     +--------------------+      +--------------------+
 *   |               |                           |
 *   v               v                           v
 * +---------------------------------------------------------+
 * |                     RotaryEmbedding                     |
 * +---------------------------------------------------------+
 *                              |
 *                              v
 *                              y
 * @endcode
 *
 * The rewrite requires opset 23, rank-four inputs, equal ``[*,1,*,*]`` caches
 * doubled on the last axis, and a static head count that becomes ``num_heads``.
 * ``one`` is the constant ``[1]`` and ``ones`` the constant ``[1,1]``. A partial
 * rotation additionally sets ``rotary_embedding_dim`` from the first Split size.
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
 *   Concat form, Split source:
 *
 *                               x       sizes
 *                               |         |
 *                               v         v
 *                           +---------------+
 *                           |     Split     |
 *                           +---------------+
 *                             |       |
 *                       +-----+       +-------------------------------+
 *                       |                                             |
 *                       | first                                       | second
 *                       |                                             v
 *                       |                                          +-----+
 *                       |                                          | Neg |
 *                       |                                          +-----+
 *                       |                                             |
 *                       |                                             | negated second
 *                       |         shape A          shape B            |
 *                       |            |                |               |
 *                       |            v                v               |
 *                       |   +-----------------+ +-----------------+   |
 *                       |   | ConstantOfShape | | ConstantOfShape |   |
 *                       |   +-----------------+ +-----------------+   |
 *                       |            |                |               |
 *                       |            | zeros A        | zeros B       |
 *                       v            v                v               v
 *                   +-----------------------+       +-----------------------+
 *                   |        Concat         |       |        Concat         |
 *                   +-----------------------+       +-----------------------+
 *                               |                               |
 *                               | padded first                  | padded second
 *                               v                               v
 *                          +--------------------------------------------+
 *                          |                     Add                    |
 *                          +--------------------------------------------+
 *                                               |
 *                                               v
 *                                               y
 *
 *   Concat form, Slice source:
 *
 *                                              x
 *                                              |
 *                       +----------------------+----------------------+
 *                       |                                             |
 *                       v                                             v
 *                   +-------+                                     +-------+
 * 0, mid, axis ---> | Slice |                 mid, dim, axis ---> | Slice |
 *                   +-------+                                     +-------+
 *                       |                                             |
 *                       | first                                       | second
 *                       |                                             v
 *                       |                                          +-----+
 *                       |                                          | Neg |
 *                       |                                          +-----+
 *                       |                                             |
 *                       |                                             | negated second
 *                       |         shape A          shape B            |
 *                       |            |                |               |
 *                       |            v                v               |
 *                       |   +-----------------+ +-----------------+   |
 *                       |   | ConstantOfShape | | ConstantOfShape |   |
 *                       |   +-----------------+ +-----------------+   |
 *                       |            |                |               |
 *                       |            | zeros A        | zeros B       |
 *                       v            v                v               v
 *                   +-----------------------+       +-----------------------+
 *                   |        Concat         |       |        Concat         |
 *                   +-----------------------+       +-----------------------+
 *                               |                               |
 *                               | padded first                  | padded second
 *                               v                               v
 *                          +--------------------------------------------+
 *                          |                     Add                    |
 *                          +--------------------------------------------+
 *                                               |
 *                                               v
 *                                               y
 *
 *   ScatterND form, Split source:
 *
 *                                           x       sizes
 *                                           |         |
 *                                           v         v
 *                                         +---------------+
 *                                         |     Split     |
 *                                         +---------------+
 *                                           |       |
 *                         +-----------------+       +-------------------------------+
 *                         |                                                         |
 *                         | first                                                   | second
 *                         |                                                         v
 *                         |                                                      +-----+
 *                         |                                                      | Neg |
 *                         |                                                      +-----+
 *                         |                                                         |
 *         shape A         |                             shape B                     |
 *            |            |                                |                        |
 *            v            |                                v                        |
 * +-----------------+     |                     +-----------------+                 |
 * | ConstantOfShape |     |                     | ConstantOfShape |                 |
 * +-----------------+     |                     +-----------------+                 |
 *          |              |                              |                          |
 *          v              v                              v                          v
 *    +-----------+  +-----------+                  +-----------+              +-----------+
 *    | Transpose |  | Transpose |                  | Transpose |              | Transpose |
 *    +-----------+  +-----------+                  +-----------+              +-----------+
 *          |              |                              |                          |
 *          | indices A    |                              | indices B                |
 *          |     |        |                              |     |                    |
 *          v     v        v                              v     v                    v
 *    +---------------------------+                 +---------------------------------+
 *    |         ScatterND         |                 |            ScatterND            |
 *    +---------------------------+                 +---------------------------------+
 *                 |                                                |
 *                 v                                                v
 *          +-----------+                                    +-----------+
 *          | Transpose |                                    | Transpose |
 *          +-----------+                                    +-----------+
 *                 |                                                |
 *                 v                                                v
 *          +-------------------------------------------------------------+
 *          |                             Add                             |
 *          +-------------------------------------------------------------+
 *                                        |
 *                                        v
 *                                        y
 *
 * After:
 *   Second part negated:
 *
 *     x       sizes
 *     |         |
 *     v         v
 *   +---------------+
 *   |     Split     |
 *   +---------------+
 *     |       |
 *     |       +------------+
 *     |                    |
 *     |                    v
 *     |                 +-----+
 *     |                 | Neg |
 *     |                 +-----+
 *     |                    |
 *     | first              | negated second
 *     v                    v
 *   +------------------------------+
 *   |            Concat            |
 *   +------------------------------+
 *                  |
 *                  v
 *                  y
 *
 *   First part negated:
 *
 *     x       sizes
 *     |         |
 *     v         v
 *   +---------------+
 *   |     Split     |
 *   +---------------+
 *     |       |
 *     |       +-----------------------+
 *     v                               |
 *   +-----+                           |
 *   | Neg |                           |
 *   +-----+                           |
 *     |                               |
 *     | negated first                 | second
 *     v                               v
 *   +-----------------------------------------+
 *   |                 Concat                  |
 *   +-----------------------------------------+
 *                       |
 *                       v
 *                       y
 * @endcode
 *
 * The ScatterND form accepts the same contiguous Slice pair as the Concat form,
 * and Apply then emits the Split node drawn above. The zero padding sits on
 * opposite Concat inputs, so Apply mirrors the emitted Concat operand order when
 * the matched pair negated the first part. The two parts cover the source axis,
 * existing Split parts are equal in the ScatterND form, padding is exactly zero,
 * branch placements or indices are complementary, and removed branch results
 * must be unshared.
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
 *   Unshifted form:
 *
 *                     B
 *                     |
 *                     v
 *                +---------+
 *                | Squeeze |
 *                +---------+
 *                     |
 *                     | limit
 *         +-----------+---------------------------------+
 *         |                                             |
 *   zero  |  one                                        |
 *     |   |   |                                         |
 *     v   v   v                                         |
 * +----------------+                                    |
 * |     Range      |                                    |
 * +----------------+                                    |
 *         |                                             |
 *         |                           A                 |
 *         |                           |                 |
 *         |                           v                 |
 *         |                      +---------+            |
 *         |                      | Squeeze |            |
 *         |                      +---------+            |
 *         |                           |                 |
 *         | axes 0, 1, 2              | start           |    one
 *         |         |                 |                 |     |
 *         v         v                 v                 v     v
 *     +-----------------+         +--------------------------------+
 *     |    Unsqueeze    |         |             Range              |
 *     +-----------------+         +--------------------------------+
 *              |                                  |
 *              |                                  |   axes 0, 1, 3
 *              |                                  |           |
 *              |                                  v           v
 *              |                              +-----------------+
 *              |                              |    Unsqueeze    |
 *              |                              +-----------------+
 *              |                                    |
 *              v                                    v
 *          +-------------------------------------------------+
 *          |                   LessOrEqual                   |
 *          +-------------------------------------------------+
 *                                  |
 *                                  v
 *                                 mask
 *
 *   Shifted form:
 *
 *                     B
 *                     |
 *                     v
 *                +---------+
 *                | Squeeze |
 *                +---------+
 *                     |
 *                     | limit
 *         +-----------+---------------------------------+
 *         |                                             |
 *   zero  |  one                                        |
 *     |   |   |                                         |
 *     v   v   v                                         |
 * +----------------+                                    |
 * |     Range      |                                    |
 * +----------------+                                    |
 *         |                                             |
 *         |                           A                 |
 *         |                           |                 |
 *         |                           v                 |
 *         |                      +---------+            |
 *         |                      | Squeeze |            |
 *         |                      +---------+            |
 *         |                           |                 |
 *         | axes 0, 1, 2              | start           |    one
 *         |         |                 |                 |     |
 *         v         v                 v                 v     v
 *     +-----------------+         +--------------------------------+
 *     |    Unsqueeze    |         |             Range              |
 *     +-----------------+         +--------------------------------+
 *              |                                  |
 *              |                                  |   axes 0, 1, 3
 *              |                                  |           |
 *              |                                  v           v
 *              |                              +-----------------+
 *              |                              |    Unsqueeze    |
 *              |                              +-----------------+
 *              |                                    |
 *              |                                    |     shift
 *              |                                    |       |
 *              |                                    v       v
 *              |                                 +-------------+
 *              |                                 |     Sub     |
 *              |                                 +-------------+
 *              |                                        |
 *              v                                        v
 *          +----------------------------------------------------+
 *          |                      Greater                       |
 *          +----------------------------------------------------+
 *                                    |
 *                                    v
 *                                   mask
 *
 * After:
 *   Unshifted form:
 *
 *          A         B
 *          |         |
 *          v         v
 *    +--------------------------+
 *    | intermediate::CausalMask |
 *    +--------------------------+
 *                 |
 *                 v
 *                mask
 *
 *   Shifted form:
 *
 *        A         B        shift
 *        |         |          |
 *        v         v          v
 *    +---------------------------------+
 *    | intermediate::ShiftedCausalMask |
 *    +---------------------------------+
 *                    |
 *                    v
 *                   mask
 * @endcode
 *
 * Both Range nodes share the same limit, the row Range starts at zero, and both
 * use unit steps. The row and column Unsqueeze axes are respectively 0, 1, 2 and
 * 0, 1, 3. Opset 13 or newer is required; shared upstream construction nodes are
 * preserved instead of being removed.
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
 *
 *         A                               B
 *         |                               |
 *         v                               v
 *    +---------+                     +---------+
 *    | Squeeze |                     | Squeeze |
 *    +---------+                     +---------+
 *         |                               |
 *         | limit                         | limit
 *   zero  |  one                   zero   |  one
 *     |   |   |                      |    |   |
 *     v   v   v                      v    v   v
 * +----------------+             +----------------+
 * |     Range      |             |     Range      |
 * +----------------+             +----------------+
 *         |                               |
 *         |  axes 0, 1, 2                 |  axes 1, 2, 3
 *         |        |                      |        |
 *         v        v                      v        v
 *     +-----------------+             +-----------------+
 *     |    Unsqueeze    |             |    Unsqueeze    |
 *     +-----------------+             +-----------------+
 *              |                               |
 *              |                               |     C
 *              |                               |     |
 *              |                               v     v
 *              |                            +---------------+
 *              |                            |      Mul      |
 *              |                            +---------------+
 *              |                                   |
 *              v                                   v
 *          +---------------------------------------------+
 *          |                     Add                     |
 *          +---------------------------------------------+
 *                                |
 *                                v
 *                               mask
 *
 * After:
 *
 *        A         B         C
 *        |         |         |
 *        v         v         v
 *    +--------------------------------+
 *    | intermediate::CausalMaskMulAdd |
 *    +--------------------------------+
 *                    |
 *                    v
 *                   mask
 * @endcode
 *
 * Both Range nodes are zero-based with unit steps and keep their own constant
 * inputs. The first and second Unsqueeze axes are respectively 0, 1, 2 and
 * 1, 2, 3. All removed intermediates must be unshared, and opset 13 is required.
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
 *   Direct position ids, Unsqueeze before Cast, uncast outputs:
 *
 *     position ids      axes 1
 *           |             |
 *           v             v
 *       +---------------------+
 *       |      Unsqueeze      |
 *       +---------------------+
 *                  |
 *                  v
 *           +-----------+
 *           |   Cast    |
 *           +-----------+
 *                  |
 *                  | float positions   shape 0, -1, 1
 *                  |                          |
 *                  v                          v
 *           +--------------------------------------+
 *           |                Reshape               |
 *           +--------------------------------------+
 *                              |
 *                              | reshaped
 *                weights       |
 *                   |          |
 *                   v          v
 *             +--------------------+
 *             |         Mul        |
 *             +--------------------+
 *                       |
 *                       | weighted
 *              +--------+--------+
 *              |                 |
 *              v                 v
 *         +---------+       +---------+
 *         |   Cos   |       |   Sin   |
 *         +---------+       +---------+
 *              |                 |
 *              v                 v
 *           cosine              sine
 *
 *   Ranged ids, Cast before Unsqueeze, cast outputs:
 *
 *        dim 1                        dim 2
 *          |                            |
 *          v                            v
 *     +---------+                  +---------+
 *     | Squeeze |                  | Squeeze |
 *     +---------+                  +---------+
 *          |                            |
 *          | start                      | limit         one
 *          |                            |                |
 *          v                            v                v
 *     +--------------------------------------------------------+
 *     |                          Range                         |
 *     +--------------------------------------------------------+
 *                                |
 *                                | positions
 *                                v
 *                          +-----------+
 *                          |   Cast    |
 *                          +-----------+
 *                                |
 *                                |          axes 0, 1
 *                                |               |
 *                                v               v
 *                          +------------------------+
 *                          |        Unsqueeze       |
 *                          +------------------------+
 *                                     |
 *                                     |     shape 0, -1, 1
 *                                     |            |
 *                                     v            v
 *                          +---------------------------+
 *                          |          Reshape          |
 *                          +---------------------------+
 *                                        |
 *                                        | reshaped
 *                          weights       |
 *                             |          |
 *                             v          v
 *                       +--------------------+
 *                       |         Mul        |
 *                       +--------------------+
 *                                 |
 *                                 | weighted
 *                        +--------+--------+
 *                        |                 |
 *                        v                 v
 *                   +---------+       +---------+
 *                   |   Cos   |       |   Sin   |
 *                   +---------+       +---------+
 *                        |                 |
 *                        v                 v
 *                   +---------+       +---------+
 *                   |  Cast   |       |  Cast   |
 *                   +---------+       +---------+
 *                        |                 |
 *                        v                 v
 *                     cosine              sine
 *
 * After:
 *   Direct position ids:
 *
 *      position ids   weights
 *            |           |
 *            v           v
 *    +---------------------------+
 *    | intermediate::CosSinCache |
 *    +---------------------------+
 *            |           |
 *            v           v
 *         cosine        sine
 *
 *   Ranged ids:
 *
 *        dim 1       dim 2      weights
 *          |           |           |
 *          v           v           v
 *    +------------------------------------+
 *    | intermediate::CosSinCacheWithRange |
 *    +------------------------------------+
 *              |               |
 *              v               v
 *           cosine            sine
 * @endcode
 *
 * The three choices drawn above are independent: ids may be direct or ranged,
 * Cast and Unsqueeze may appear in either order, and the two output Cast boxes
 * must either both appear with the same target or both be absent. Weights and
 * the position Cast are float, the Reshape target is exactly ``[0,-1,1]``, both
 * ranges use a unit step, and removed intermediates must be unshared. Direct ids
 * use Unsqueeze axis 1; ranged ids use axes ``[0,1]`` or ``[1]``. The emitted
 * function name also encodes the output Cast target and non-default axes.
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
 *
 *       X
 *       |
 *       +-----------------------+
 *       |                       |
 *       |                       |             sizes
 *       |                       |               |
 *       |                       v               v
 *       |                    +---------------------+
 *       |                    |        Split        |
 *       |                    +---------------------+
 *       |                       |               |
 *       |                       | first         | second
 *       |                       |               v
 *       |                       |            +-----+
 *       |                       |            | Neg |
 *       |                       |            +-----+
 *       |                       |               |
 *       |                       |               | negated second
 *       |                       v               v
 *       |                    +---------------------+
 *       |                    |        Concat       |   inputs in order: negated second, first
 *       |                    +---------------------+
 *       |                              |
 *       |                              |          sine cache
 *       |                              |               |
 *       |                              v               v
 *       |                       +----------------------------+
 *       |                       |            Mul             |
 *       |                       +----------------------------+
 *       |                                     |
 *       |   cosine cache                      |
 *       |        |                            |
 *       v        v                            |
 *    +--------------------+                   |
 *    |         Mul        |                   |
 *    +--------------------+                   |
 *              |                              |
 *              v                              v
 *    +---------------------------------------------------+
 *    |                        Add                        |
 *    +---------------------------------------------------+
 *                             |
 *                             v
 *                             Y
 *
 * After:
 *
 *       X       cosine cache      sine cache
 *       |            |                 |
 *       v            v                 v
 *    +-----------------------------------+
 *    | intermediate::HalfRotaryEmbedding |
 *    +-----------------------------------+
 *                    |
 *                    v
 *                    Y
 * @endcode
 *
 * ``X`` must be rank four, opset 18 or newer is required, and Split and Concat
 * both act on the last axis. The Split either receives equal INT64 sizes, as
 * drawn, or carries ``num_outputs=2`` with no second input. The Concat is
 * canonical only when it rebuilds ``Neg(second), first``. Either Mul may hold
 * its cache on either input, both Mul results reach the same Add, and every
 * removed intermediate must be unshared.
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
 *   Rank-four MatMul with an additive mask:
 *
 *         Q     scale             K     scale
 *         |       |               |       |
 *         v       v               v       v
 *     +---------------+       +---------------+
 *     |      Mul      |       |      Mul      |
 *     +---------------+       +---------------+
 *             |                       |
 *             |                       v
 *             |                 +-----------+
 *             |                 | Transpose |   perm 0, 1, 3, 2
 *             |                 +-----------+
 *             |                       |
 *             v                       v
 *         +-------------------------------+
 *         |            MatMul             |
 *         +-------------------------------+
 *                         |
 *                         | scores
 *                         |                     mask    zero   minus infinity
 *                         |                       |       |           |
 *                         |                       v       v           v
 *                         |                   +-----------------------------+
 *                         |                   |            Where            |
 *                         |                   +-----------------------------+
 *                         |                                  |
 *                         v                                  v
 *                     +------------------------------------------+
 *                     |                   Add                    |
 *                     +------------------------------------------+
 *                                          |
 *                                          v
 *                                     +---------+
 *                                     | Softmax |
 *                                     +---------+
 *                                          |
 *                       +------------------+------------+
 *                       |                               |
 *                       v                               |
 *                 +-----------+                         |
 *                 |   IsNaN   |                         |
 *                 +-----------+                         |
 *                       |                               |
 *                       |             zero              |
 *                       |               |               |
 *                       v               v               v
 *                   +---------------------------------------+
 *                   |                 Where                 |
 *                   +---------------------------------------+
 *                                       |
 *                                       |             V
 *                                       |             |
 *                                       v             v
 *                                   +---------------------+
 *                                   |       MatMul        |
 *                                   +---------------------+
 *                                             |
 *                                             v
 *                                             Y
 *
 *   FusedMatMul with a direct Where mask:
 *
 *                         Q     scale             K     scale
 *                         |       |               |       |
 *                         v       v               v       v
 *                     +---------------+       +---------------+
 *                     |      Mul      |       |      Mul      |
 *                     +---------------+       +---------------+
 *                             |                       |
 *                             v                       v
 *                         +---------------------------------+
 *                         |           FusedMatMul           |
 *                         +---------------------------------+
 *                                          |
 *                                          | scores
 *                            mask          |         minus infinity
 *                              |           |               |
 *                              v           v               v
 *                        +-----------------------------------+
 *                        |               Where               |
 *                        +-----------------------------------+
 *                                          |
 *                                          v
 *                                     +---------+
 *                                     | Softmax |
 *                                     +---------+
 *                                          |
 *                       +------------------+------------+
 *                       |                               |
 *                       v                               |
 *                 +-----------+                         |
 *                 |   IsNaN   |                         |
 *                 +-----------+                         |
 *                       |                               |
 *                       |             zero              |
 *                       |               |               |
 *                       v               v               v
 *                   +---------------------------------------+
 *                   |                 Where                 |
 *                   +---------------------------------------+
 *                                       |
 *                                       |             V
 *                                       |             |
 *                                       v             v
 *                                   +---------------------+
 *                                   |       MatMul        |
 *                                   +---------------------+
 *                                             |
 *                                             v
 *                                             Y
 *
 *   Rank-three inputs with a switched Where mask:
 *
 *             Q     scale                         K     scale
 *             |       |                           |       |
 *             v       v                           v       v
 *         +---------------+                   +---------------+
 *         |      Mul      |                   |      Mul      |
 *         +---------------+                   +---------------+
 *                 |                                   |
 *                 |     shape                         |     shape
 *                 |       |                           |       |
 *                 v       v                           v       v
 *             +-------------------+               +-------------------+
 *             |      Reshape      |               |      Reshape      |
 *             +-------------------+               +-------------------+
 *                       |                                   |
 *                       v                                   v
 *                 +-----------+                       +-----------+
 *                 | Transpose |   perm 0, 2, 1, 3     | Transpose |   perm 0, 2, 3, 1
 *                 +-----------+                       +-----------+
 *                       |                                   |
 *                       v                                   v
 *                   +---------------------------------------------+
 *                   |                   MatMul                    |
 *                   +---------------------------------------------+
 *                                          |
 *                mask   minus infinity     | scores
 *                  |           |           |
 *                  v           v           v
 *              +-------------------------------------------------------+
 *              |                         Where                         |
 *              +-------------------------------------------------------+
 *                                          |
 *                                          v
 *                                     +---------+
 *                                     | Softmax |
 *                                     +---------+
 *                                          |
 *                       +------------------+------------+
 *                       |                               |
 *                       v                               |
 *                 +-----------+                         |
 *                 |   IsNaN   |                         |
 *                 +-----------+                         |
 *                       |                               |
 *                       |             zero              |
 *                       |               |               |
 *                       v               v               v
 *                   +---------------------------------------+
 *                   |                 Where                 |
 *                   +---------------------------------------+
 *                                       |
 *                                       |             V
 *                                       |             |
 *                                       v             v
 *                                   +---------------------+
 *                                   |       MatMul        |
 *                                   +---------------------+
 *                                             |
 *                                             v
 *                                             Y
 *
 *   Grouped-query repeat-interleave on keys and values:
 *
 *         Q     scale             K     axes 2
 *         |       |               |       |
 *         v       v               v       v
 *     +---------------+       +---------------+
 *     |      Mul      |       |   Unsqueeze   |
 *     +---------------+       +---------------+
 *             |                       |     scale
 *             |                       |       |
 *             |                       v       v
 *             |                   +---------------+
 *             |                   |      Mul      |
 *             |                   +---------------+
 *             |                           |   expand shape
 *             |                           |       |
 *             |                           v       v
 *             |                       +---------------+
 *             |                       |    Expand     |
 *             |                       +---------------+
 *             |                               |   gqa shape
 *             |                               |       |
 *             |                               v       v
 *             |                           +---------------+
 *             |                           |    Reshape    |
 *             |                           +---------------+
 *             |                                   |
 *             |                                   v
 *             |                             +-----------+
 *             |                             | Transpose |   perm 0, 1, 3, 2
 *             |                             +-----------+
 *             |                                   |
 *             v                                   v
 *         +-------------------------------------------+
 *         |                  MatMul                   |
 *         +-------------------------------------------+
 *                               |
 *                               | scores
 *                               |         mask      zero   minus infinity
 *                               |           |         |           |
 *                               |           v         v           v
 *                               |       +-------------------------------+
 *                               |       |             Where             |
 *                               |       +-------------------------------+
 *                               |                       |
 *                               v                       v
 *                         +---------------------------------+
 *                         |               Add               |
 *                         +---------------------------------+
 *                                          |
 *                                          v
 *                                     +---------+
 *                                     | Softmax |
 *                                     +---------+
 *                                          |
 *                       +------------------+------------+
 *                       |                               |
 *                       v                               |
 *                 +-----------+                         |
 *                 |   IsNaN   |                         |
 *                 +-----------+                         |
 *                       |                               |
 *                       |             zero              |
 *                       |               |               |
 *                       v               v               v
 *                   +---------------------------------------+
 *                   |                 Where                 |
 *                   +---------------------------------------+
 *                                       |
 *                                       |           V    axes 2
 *                                       |           |       |
 *                                       |           v       v
 *                                       |       +---------------+
 *                                       |       |   Unsqueeze   |
 *                                       |       +---------------+
 *                                       |               |   expand shape
 *                                       |               |       |
 *                                       |               v       v
 *                                       |           +---------------+
 *                                       |           |    Expand     |
 *                                       |           +---------------+
 *                                       |                   |   gqa shape
 *                                       |                   |       |
 *                                       |                   v       v
 *                                       |               +---------------+
 *                                       |               |    Reshape    |
 *                                       |               +---------------+
 *                                       |                       |
 *                                       v                       v
 *                                   +-------------------------------+
 *                                   |            MatMul             |
 *                                   +-------------------------------+
 *                                                   |
 *                                                   v
 *                                                   Y
 *
 * After:
 *   Rank-four inputs:
 *
 *         Q         K         V         mask        scale
 *         |         |         |           |           |
 *         v         v         v           v           v
 *     +-----------------------------------------------------+
 *     |            intermediate::LocalAttention             |
 *     +-----------------------------------------------------+
 *                                |
 *                                v
 *                                Y
 *
 *   Rank-three inputs:
 *
 *             Q     shape                         K     shape
 *             |       |                           |       |
 *             v       v                           v       v
 *         +-------------------+               +-------------------+
 *         |      Reshape      |               |      Reshape      |
 *         +-------------------+               +-------------------+
 *                   |                                   |
 *                   v                                   v
 *             +-----------+                       +-----------+
 *             | Transpose |   perm 0, 2, 1, 3     | Transpose |   perm 0, 2, 1, 3
 *             +-----------+                       +-----------+
 *                   |                                   |       V       mask        scale
 *                   |                                   |       |         |           |
 *                   v                                   v       v         v           v
 *               +---------------------------------------------------------------------------+
 *               |                       intermediate::LocalAttention                        |
 *               +---------------------------------------------------------------------------+
 *                                                     |
 *                                                     v
 *                                                     Y
 * @endcode
 *
 * The four score paths and the three masking forms combine freely; each drawing
 * shows one legal pairing. The additive form needs ``Where(mask, zero, minus
 * infinity)``, the direct form places negative infinity on the third Where
 * input, and the switched form places it on the second one. Softmax uses axis
 * -1 and opset 18 or newer is required. The rank-four key Transpose uses
 * permutation 0, 1, 3, 2 and the rank-three one 0, 2, 3, 1, while FusedMatMul
 * must set ``transA=0``, ``transB=1``, and ``alpha=1``. Rank-three inputs need
 * matching Reshape targets, and Apply rebuilds both branches with permutation
 * 0, 2, 1, 3. Query and key scales must be equal floating scalars, the GQA key
 * and value repeat shapes must match, and every removed result is unshared. The
 * emitted function name encodes GQA, the switched mask, the missing transpose,
 * and the element type; GQA additionally appends the expand and repeat shapes
 * to the call.
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
 *   Reshape form:
 *
 *     Q         K    axes 2                     V    axes 2               mask      scale
 *     |         |       |                       |       |                   |         |
 *     |         v       v                       v       v                   |         |
 *     |     +---------------+               +---------------+               |         |
 *     |     |   Unsqueeze   |               |   Unsqueeze   |               |         |
 *     |     +---------------+               +---------------+               |         |
 *     |             |   expand shape                |   expand shape        |         |
 *     |             |       |                       |       |               |         |
 *     |             v       v                       v       v               |         |
 *     |         +---------------+               +---------------+           |         |
 *     |         |    Expand     |               |    Expand     |           |         |
 *     |         +---------------+               +---------------+           |         |
 *     |                 |   gqa shape                   |   gqa shape       |         |
 *     |                 |       |                       |       |           |         |
 *     |                 v       v                       v       v           |         |
 *     |             +---------------+               +---------------+       |         |
 *     |             |    Reshape    |               |    Reshape    |       |         |
 *     |             +---------------+               +---------------+       |         |
 *     |                     |                               |               |         |
 *     v                     v                               v               v         v
 *   +-------------------------------------------------------------------------------------+
 *   |                            intermediate::LocalAttention                             |
 *   +-------------------------------------------------------------------------------------+
 *                                              |
 *                                              v
 *                                              Y
 *
 *   Squeeze form:
 *
 *     Q         K    axes 2                     V    axes 2               mask      scale
 *     |         |       |                       |       |                   |         |
 *     |         v       v                       v       v                   |         |
 *     |     +---------------+               +---------------+               |         |
 *     |     |   Unsqueeze   |               |   Unsqueeze   |               |         |
 *     |     +---------------+               +---------------+               |         |
 *     |             |   expand shape                |   expand shape        |         |
 *     |             |       |                       |       |               |         |
 *     |             v       v                       v       v               |         |
 *     |         +---------------+               +---------------+           |         |
 *     |         |    Expand     |               |    Expand     |           |         |
 *     |         +---------------+               +---------------+           |         |
 *     |                 |   squeeze axes                |   squeeze axes    |         |
 *     |                 |       |                       |       |           |         |
 *     |                 v       v                       v       v           |         |
 *     |             +---------------+               +---------------+       |         |
 *     |             |    Squeeze    |               |    Squeeze    |       |         |
 *     |             +---------------+               +---------------+       |         |
 *     |                     |                               |               |         |
 *     v                     v                               v               v         v
 *   +-------------------------------------------------------------------------------------+
 *   |                            intermediate::LocalAttention                             |
 *   +-------------------------------------------------------------------------------------+
 *                                              |
 *                                              v
 *                                              Y
 *
 * After:
 *
 *       Q         K         V       mask      scale     expand shape  gqa shape
 *       |         |         |         |         |           |             |
 *       v         v         v         v         v           v             v
 *   +---------------------------------------------------------------------------+
 *   |                      intermediate::LocalAttentionGQA                      |
 *   +---------------------------------------------------------------------------+
 *                                         |
 *                                         v
 *                                         Y
 * @endcode
 *
 * Key and value branches must use identical constant shapes, rank-five Expand
 * targets with singleton non-repeat dimensions, and the same final operation.
 * The Reshape form ends on a rank-four shape while the Squeeze form ends on a
 * single axis, so the last call input is a shape or an axis accordingly. Inputs
 * must be floating and every removed repeat intermediate unshared. The concrete
 * local-function name records Reshape versus Squeeze and retains the original
 * attention suffix.
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
 *   ONNX decomposition, Reshape form:
 *
 *     Q  past K     K                            past V     V                     mask
 *     |     |       |                               |       |                       |
 *     |     v       v                               v       v                       |
 *     | +---------------+                       +---------------+                   |
 *     | |    Concat     |                       |    Concat     |                   |
 *     | +---------------+                       +---------------+                   |
 *     |         |    axes 2                             |    axes 2                 |
 *     |         |       |                               |       |                   |
 *     |         v       v                               v       v                   |
 *     |     +---------------+                       +---------------+               |
 *     |     |   Unsqueeze   |                       |   Unsqueeze   |               |
 *     |     +---------------+                       +---------------+               |
 *     |             |   expand shape                        |   expand shape        |
 *     |             |       |                               |       |               |
 *     |             v       v                               v       v               |
 *     |         +---------------+                       +---------------+           |
 *     |         |    Expand     |                       |    Expand     |           |
 *     |         +---------------+                       +---------------+           |
 *     |                 |   gqa shape                           |   gqa shape       |
 *     |                 |       |                               |       |           |
 *     |                 v       v                               v       v           |
 *     |             +---------------+                       +---------------+       |
 *     |             |    Reshape    |                       |    Reshape    |       |
 *     |             +---------------+                       +---------------+       |
 *     |                     |                                       |               |
 *     v                     v                                       v               v
 *   +-----------------------------------------------------------------------------------+
 *   |                                     Attention                                     |
 *   +-----------------------------------------------------------------------------------+
 *                                             |
 *                                             v
 *                                             Y
 *
 *   ONNX decomposition, Squeeze form:
 *
 *     Q  past K     K                            past V     V                     mask
 *     |     |       |                               |       |                       |
 *     |     v       v                               v       v                       |
 *     | +---------------+                       +---------------+                   |
 *     | |    Concat     |                       |    Concat     |                   |
 *     | +---------------+                       +---------------+                   |
 *     |         |    axes 2                             |    axes 2                 |
 *     |         |       |                               |       |                   |
 *     |         v       v                               v       v                   |
 *     |     +---------------+                       +---------------+               |
 *     |     |   Unsqueeze   |                       |   Unsqueeze   |               |
 *     |     +---------------+                       +---------------+               |
 *     |             |   expand shape                        |   expand shape        |
 *     |             |       |                               |       |               |
 *     |             v       v                               v       v               |
 *     |         +---------------+                       +---------------+           |
 *     |         |    Expand     |                       |    Expand     |           |
 *     |         +---------------+                       +---------------+           |
 *     |                 |   squeeze axes                        |   squeeze axes    |
 *     |                 |       |                               |       |           |
 *     |                 v       v                               v       v           |
 *     |             +---------------+                       +---------------+       |
 *     |             |    Squeeze    |                       |    Squeeze    |       |
 *     |             +---------------+                       +---------------+       |
 *     |                     |                                       |               |
 *     v                     v                                       v               v
 *   +-----------------------------------------------------------------------------------+
 *   |                                     Attention                                     |
 *   +-----------------------------------------------------------------------------------+
 *                                             |
 *                                             v
 *                                             Y
 *
 *   Local-function form:
 *
 *     Q  past K     K          past V     V       mask    scale sqrt    expand shape  gqa shape
 *     |     |       |             |       |         |         |             |             |
 *     |     v       v             v       v         |         |             |             |
 *     | +---------------+     +---------------+     |         |             |             |
 *     | |    Concat     |     |    Concat     |     |         |             |             |
 *     | +---------------+     +---------------+     |         |             |             |
 *     |         |                     |             |         |             |             |
 *     v         v                     v             v         v             v             v
 *   +-------------------------------------------------------------------------------------------+
 *   |                              intermediate::LocalAttentionGQA                              |
 *   +-------------------------------------------------------------------------------------------+
 *                                                 |
 *                                                 v
 *                                                 Y
 *
 * After:
 *   Direct mask:
 *
 *       Q         K         V       mask       past K        past V
 *       |         |         |         |           |             |
 *       v         v         v         v           v             v
 *   +---------------------------------------------------------------------+
 *   |                              Attention                              |
 *   +---------------------------------------------------------------------+
 *             |                   |                     |
 *             v                   v                     v
 *             Y               present K             present V
 *
 *   Switched local mask:
 *
 *                                   mask
 *                                     |
 *                                     v
 *                                 +-------+
 *                                 |  Not  |
 *                                 +-------+
 *                                     |
 *       Q         K         V         |        past K        past V
 *       |         |         |         |           |             |
 *       v         v         v         v           v             v
 *   +---------------------------------------------------------------------+
 *   |                              Attention                              |
 *   +---------------------------------------------------------------------+
 *             |                   |                     |
 *             v                   v                     v
 *             Y               present K             present V
 * @endcode
 *
 * The caches and current tensors must be compatible rank-four values, and both
 * cache Concat nodes use normalized axis 2. Opset 23 is required. The two cache
 * Concat results become the present-cache outputs of the new Attention node.
 * The Not box appears only for switched local masks, which must be bool. The
 * ONNX form copies the ``scale`` and ``is_causal`` attributes, while the local
 * ``scale_sqrt`` input is squared into the Attention ``scale`` attribute.
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
