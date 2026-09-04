// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Removes a null (all-zero) bias input from a Conv node.
 *
 * @code
 * Before:
 *                  ┌──────┐
 *   X, W, B=0 ────→│ Conv │────→ Y
 *                  └──────┘
 *
 * After:
 *                ┌──────┐
 *   X, W ───────→│ Conv │────→ Y
 *                └──────┘
 * @endcode
 *
 * The Conv node is rebuilt without its known all-zero constant bias while all
 * attributes and output names are retained.
 */
class ConvBiasNullPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConvBiasNullPattern(int priority = 0) : PatternOptimization(priority, "ConvBiasNull") {}

  /// Returns ``Conv`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Conv node whose bias input is a known all-zero constant or a safe expansion of one.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Rebuilds the Conv node without its bias input.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a Pad node followed by a Conv node into a single Conv node whose
 * ``pads`` attribute absorbs the spatial padding.
 *
 * @code
 * Before:
 *                         ┌─────┐
 *   X, pads, value=0 ────→│ Pad │────→ p
 *                         └─────┘
 *                            │
 *                            ↓
 *                     ┌──────┐
 *                     │ Conv │←──── W, B
 *                     └──────┘
 *                            │
 *                            ↓
 *                            Y
 *
 * After:
 *                ┌────────────────────┐
 *   X, W, B ────→│ Conv with new pads │────→ Y
 *                └────────────────────┘
 * @endcode
 *
 * The fusion applies when the Pad uses the default ``constant`` mode with a
 * zero constant value, pads only the spatial dimensions, and the Conv does not
 * use ``auto_pad``. The optional Conv bias, other attributes, and every Conv
 * output name are retained.
 */
class PadConvPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit PadConvPattern(int priority = 0) : PatternOptimization(priority, "PadConv") {}

  /// Returns ``Conv`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Pad node feeding a Conv node with foldable spatial padding.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched Pad-Conv sequence with one padded Conv node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a channel-wise Add following a Conv into the Conv bias.
 *
 * The Conv output must have no other consumer or graph use. The Conv weights,
 * optional bias, and Add constant must be FLOAT or DOUBLE constants with
 * matching element types and valid channel-broadcast shapes. The replacement
 * remains a standard-domain Conv and keeps any following activation unchanged.
 *
 * @code
 * Before: X, W, B -> Conv -> Add(channel constant) -> Y
 * After:  X, W, folded B -> Conv --------------------> Y
 * @endcode
 */
class ConvAddFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConvAddFusionPattern(int priority = 0)
      : PatternOptimization(priority, "ConvAddFusion") {}

  /// Returns ``Conv`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Conv followed by a foldable channel-wise Add.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the Conv-Add pair with a Conv carrying the folded bias.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses a scalar or channel-wise Mul following a Conv into its weights and
 * optional bias.
 *
 * The Conv output must have no other consumer or graph use. All folded
 * constants must be FLOAT or DOUBLE tensors of one matching type.
 *
 * @code
 * Before: X, W, B -> Conv -> Mul(channel/scalar constant) -> Y
 * After:  X, folded W, folded B -> Conv ------------------> Y
 * @endcode
 */
class ConvMulFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConvMulFusionPattern(int priority = 0)
      : PatternOptimization(priority, "ConvMulFusion") {}

  /// Returns ``Conv`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Conv followed by a foldable scalar or channel-wise Mul.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the Conv-Mul pair with a Conv carrying folded constants.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Fuses inference BatchNormalization following a Conv into the Conv weights
 * and bias.
 *
 * Scale, bias, mean, variance, Conv weights, and an optional Conv bias must be
 * one-dimensional/channel-compatible FLOAT or DOUBLE constants of one
 * matching type. Optional BatchNormalization outputs must be absent or unused.
 *
 * @code
 * Before: X, W, B -> Conv -> BatchNormalization -> Y
 * After:  X, folded W, folded B -> Conv -----------> Y
 * @endcode
 */
class ConvBatchNormalizationFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit ConvBatchNormalizationFusionPattern(int priority = 0)
      : PatternOptimization(priority, "ConvBatchNormalizationFusion") {}

  /// Returns ``Conv`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a Conv followed by a foldable inference BatchNormalization.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the Conv-BatchNormalization pair with one folded Conv.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
