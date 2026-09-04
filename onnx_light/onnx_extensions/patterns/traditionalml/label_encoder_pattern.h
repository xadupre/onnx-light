// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Composes two consecutive ``ai.onnx.ml::LabelEncoder`` mappings, including
 * propagation of the first encoder's default through the second encoder.
 *
 * The portable subset supports the standard string and int64 list attributes
 * used by LabelEncoder versions 2 and 4.
 *
 * @code
 * Before:
 *        +------------------------------+     +------------------------------+
 *   x -->| LabelEncoder(keys1, values1) |---->| LabelEncoder(keys2, values2) |---> y
 *        +------------------------------+     +------------------------------+
 *
 * After:
 *        +-----------------------------------------+
 *   x -->| LabelEncoder(composed keys and values) |---> y
 *        +-----------------------------------------+
 * @endcode
 */
class LabelEncoderFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit LabelEncoderFusionPattern(int priority = 0)
      : PatternOptimization(priority, "LabelEncoderFusion") {}

  /// Returns LabelEncoder as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a single-use pair with compatible list mappings and defaults.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the pair with the composed LabelEncoder.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
