// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses a canonical convolution-based DFT into a standard ONNX STFT.
 *
 * @code
 * Before:
 *             +-----------+     +------------+     +-----------+
 *   signal -->| Transpose |--+->| Conv(real) |---->| Unsqueeze |--+
 *             +-----------+  |  +------------+     +-----------+  |
 *                            |  +------------+     +-----------+  |  +--------+
 *                            +->| Conv(imag) |---->| Unsqueeze |--+->| Concat |
 *                               +------------+     +-----------+     +--------+
 *                                                                         |
 *                                                               +-----------+
 *                                                               | Transpose |--> output
 *                                                               +-----------+
 *
 * After:
 *                                              +------+
 *   signal, frame_step, window, frame_length -->| STFT |--> output
 *                                              +------+
 * @endcode
 *
 * The Conv weights must be matching FLOAT or DOUBLE tensors shaped
 * ``[bins, 1, frame_length]`` and must equal the real and imaginary DFT bases
 * multiplied by one common window. The convolution stride becomes
 * ``frame_step``. A half spectrum selects ``onesided=1`` and a full spectrum
 * selects ``onesided=0``. For frame lengths one and two, the coincident
 * half/full bin counts are represented as ``onesided=1``.
 */
class STFTFusionPattern final : public core::builder::PatternOptimization {
public:
  /// Creates the pattern with the given optimization priority.
  explicit STFTFusionPattern(int priority = 0) : PatternOptimization(priority, "STFTFusion") {}

  /// Returns ``Transpose`` as the only possible root operator.
  std::set<std::string> FastOpType() const override;

  /// Finds a private canonical convolution-based DFT subgraph.
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;

  /// Replaces the matched subgraph with one standard-domain STFT node.
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
