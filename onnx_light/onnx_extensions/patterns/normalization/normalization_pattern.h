// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses an explicit layer-normalization decomposition.
 *
 * @code
 * Before:
 *          +------------+
 *   x ---->| ReduceMean |----> mean
 *          +------------+
 *
 *               +-----+
 *   x, mean --->| Sub |----> centered
 *               +-----+
 *                   +-----+
 *   centered, two ->| Pow |----> squared
 *                   +-----+
 *             +------------+
 *   squared ->| ReduceMean |----> variance
 *             +------------+
 *                       +-----+       +------+
 *   variance, epsilon ->| Add |------>| Sqrt |----> deviation
 *                       +-----+       +------+
 *
 *                           +-----+
 *   centered, deviation --->| Div |----> y
 *                           +-----+
 *
 *   Alternate final path:
 *                +------------+
 *   deviation -->| Reciprocal |----> inverse
 *                +------------+
 *                            +-----+
 *   centered, inverse ------>| Mul |----> y
 *                            +-----+
 *
 * After:
 *          +-------+
 *   x ---->| Shape |----> normalized shape
 *          +-------+
 *
 *                       +-----------------+
 *   normalized shape -->| ConstantOfShape |----> scale
 *                       +-----------------+
 *                       +-----------------+
 *   normalized shape -->| ConstantOfShape |----> bias
 *                       +-----------------+
 *
 *                      +--------------------+
 *   x, scale, bias --->| LayerNormalization |----> y
 *                      +--------------------+
 * @endcode
 *
 * Both means must keep dimensions and reduce the same trailing axes. All
 * removed intermediates must be unshared. Scale and bias are initializers when
 * the final dimension is static; otherwise the shown Shape branches create
 * one and zero tensors.
 */
class LayerNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit LayerNormalizationPattern(int priority = 1)
      : PatternOptimization(priority, "LayerNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Folds a following affine transform into LayerNormalization parameters.
 *
 * @code
 * Before:
 *                     +--------------------+
 *   x, scale, bias -->| LayerNormalization |----> normalized
 *                     +--------------------+
 *
 *                               +-----+                       +-----+
 *   normalized, extra scale --->| Mul |----> scaled --------->| Add |----> y
 *                               +-----+                       +-----+
 *                                                                ^
 *                                                                |
 *                                                           extra bias
 *
 * After:
 *                           +-----+
 *   scale, extra scale ---->| Mul |----> new scale
 *                           +-----+
 *
 *                           +-----+                    +-----+
 *   bias, extra scale ----->| Mul |----> scaled bias ->| Add |----> new bias
 *                           +-----+                    +-----+
 *                                                          ^
 *                                                          |
 *                                                     extra bias
 *
 *                            +--------------------+
 *   x, new scale, new bias ->| LayerNormalization |----> y
 *                            +--------------------+
 * @endcode
 *
 * The extra scale, optional bias, and original scale must have equal shapes.
 * The normalization and Mul outputs must not have unrelated consumers.
 * Parameter Mul or Add nodes are omitted when an existing unit scale or a
 * missing bias makes them unnecessary.
 */
class LayerNormalizationScalePattern final : public core::builder::PatternOptimization {
public:
  explicit LayerNormalizationScalePattern(int priority = 1)
      : PatternOptimization(priority, "LayerNormalizationScale") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a normalization from its stash type back to the original element type.
 *
 * @code
 * Before:
 *          +------+                    +---------------+       +------+
 *   x ---->| Cast |----> promoted ---->| Normalization |------>| Cast |----> y
 *          +------+                    +---------------+       +------+
 *                                          ^
 *                                          |
 *                                    scale, bias, ...
 *
 * After:
 *                       +------+
 *   scale, bias, ... -->| Cast |----> converted parameters
 *                       +------+
 *
 *                             +---------------+
 *   x, converted parameters ->| Normalization |----> y
 *                             +---------------+
 * @endcode
 *
 * The normalization may be GroupNormalization, LayerNormalization,
 * RMSNormalization, or SimplifiedLayerNormalization in the ONNX or Microsoft
 * domain. A separate Cast is emitted for every non-data input. Optional
 * outputs must be unused, and both outer Cast results must be exclusive to the
 * chain.
 */
class CastLayerNormalizationCastPattern final : public core::builder::PatternOptimization {
public:
  explicit CastLayerNormalizationCastPattern(int priority = 1)
      : PatternOptimization(priority, "CastLayerNormalizationCast") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Removes an identity BatchNormalization in inference mode.
 *
 * @code
 * Before:
 *                                        +--------------------+
 *   x, scale, bias, mean, variance ----->| BatchNormalization |----> y
 *                                        +--------------------+
 *
 * After:
 *          +----------+
 *   x ---->| Identity |----> y
 *          +----------+
 * @endcode
 *
 * Scale and variance must be constant ones; bias and mean must be constant
 * zeros. Epsilon is zero, training mode is disabled, and running-statistic
 * outputs must be unused.
 */
class BatchNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit BatchNormalizationPattern(int priority = 0)
      : PatternOptimization(priority, "BatchNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Expands training BatchNormalization into primitive operators.
 *
 * @code
 * Before:
 *                                                    +--------------------+
 *   x, scale, bias, running mean, running variance ->| BatchNormalization |----> y
 *                                                    +--------------------+
 *
 * After:
 *          +------------+
 *   x ---->| ReduceMean |----> mean
 *          +------------+
 *               +-----+
 *   x, mean --->| Sub |----> centered
 *               +-----+
 *                          +-----+       +------------+
 *   centered, centered --->| Mul |------>| ReduceMean |----> variance
 *                          +-----+       +------------+
 *                                             |
 *                                             v
 *                                  +-----+       +------+
 *                       epsilon -->| Add |------>| Sqrt |----> deviation
 *                                  +-----+       +------+
 *                                     ^
 *                                     |
 *                                 variance
 *
 *                           +-----+
 *   centered, deviation --->| Div |----> normalized
 *                           +-----+
 *
 *            +---------+
 *   scale -->| Reshape |----> broadcast scale
 *            +---------+
 *                                 +-----+
 *   normalized, broadcast scale ->| Mul |----> scaled
 *                                 +-----+
 *           +---------+
 *   bias -->| Reshape |----> broadcast bias
 *           +---------+
 *                            +-----+
 *   scaled, broadcast bias ->| Add |----> y
 *                            +-----+
 * @endcode
 *
 * The rewrite requires opset 18, a known input rank of at least two, known
 * scale and bias ranks, and unused running-statistic outputs. Both ReduceMean
 * nodes reduce every axis except channel axis 1. Reshape is emitted only for
 * rank-one scale or bias inputs.
 */
class BatchNormalizationTrainingPattern final : public core::builder::PatternOptimization {
public:
  explicit BatchNormalizationTrainingPattern(int priority = 0)
      : PatternOptimization(priority, "BatchNormalizationTraining") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses an explicit root-mean-square normalization decomposition.
 *
 * @code
 * Before:
 *   Optional input conversion:
 *          +------+
 *   x ---->| Cast |----> z
 *          +------+
 *
 *            +-----+
 *   z, two ->| Pow |----> squared
 *            +-----+
 *             +------------+
 *   squared ->| ReduceMean |----> mean square
 *             +------------+
 *                            +-----+       +------+
 *   mean square, epsilon --->| Add |------>| Sqrt |----> deviation
 *                            +-----+       +------+
 *
 *                +------------+
 *   deviation -->| Reciprocal |----> inverse
 *                +------------+
 *
 *   Alternate inverse path:
 *                      +-----+
 *   one, deviation --->| Div |----> inverse
 *                      +-----+
 *
 *                +-----+
 *   z, inverse ->| Mul |----> normalized
 *                +-----+
 *
 *   Optional output conversion:
 *                +------+
 *   normalized ->| Cast |----> y
 *                +------+
 *
 * After:
 *          +-------+       +-----------------+
 *   x ---->| Shape |------>| ConstantOfShape |----> scale
 *          +-------+       +-----------------+
 *
 *               +------------------+
 *   x, scale -->| RMSNormalization |----> y
 *               +------------------+
 * @endcode
 *
 * The optional Cast pair must convert to the reduction stash type and restore
 * the original type. The rewrite requires opset 23, suffix reduction axes,
 * and unshared intermediates. The scale is a static initializer when its shape
 * is known; otherwise the shown Shape branch creates an all-ones tensor.
 */
class RMSNormalizationPattern final : public core::builder::PatternOptimization {
public:
  explicit RMSNormalizationPattern(int priority = 1)
      : PatternOptimization(priority, "RMSNormalization") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Folds a constant post-scale into RMSNormalization.
 *
 * @code
 * Before:
 *                +------------------+
 *   x, scale1 -->| RMSNormalization |----> normalized
 *                +------------------+
 *                              +-----+
 *   normalized, scale2 ------->| Mul |----> y
 *                              +-----+
 *
 * After:
 *                       +------------------+
 *   x, combined scale ->| RMSNormalization |----> y
 *                       +------------------+
 * @endcode
 *
 * Both scales must be constants with identical shapes and a representable
 * elementwise product. The RMSNormalization output must be unshared.
 */
class RMSNormalizationMulPattern final : public core::builder::PatternOptimization {
public:
  explicit RMSNormalizationMulPattern(int priority = 1)
      : PatternOptimization(priority, "RMSNormalizationMul") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
