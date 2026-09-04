// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Decomposes a real-valued STFT into portable ONNX operators.
 *
 * @code
 * Before:
 *   signal ───────┐
 *   frame_step ───┤
 *   window ───────┤
 *   frame_length ─┴─→ ┌──────┐
 *                     │ STFT │────→ output
 *                     └──────┘
 *
 * After:
 *                  ┌───────────┐   ┌───────────┐
 *   signal ───────→│ Transpose │──→│ Conv(real)│──→ real ───┐
 *                  └───────────┘   └───────────┘             │
 *                         │        ┌───────────┐              ↓
 *                         └───────→│ Conv(imag)│──→ imag ─→ Unsqueeze
 *                                  └───────────┘              │
 *   real ───────────────────────────────────────────────────→ Concat
 *                                                             │
 *                                                             ↓
 *                                                        ┌───────────┐
 *                                                        │ Transpose │────→ output
 *                                                        └───────────┘
 * @endcode
 *
 * The generated Conv weights contain the DFT basis, with an optional constant
 * window folded into them. The rewrite supports FLOAT and DOUBLE real signals
 * of shape ``[batch, signal_length, 1]``, constant positive scalar INT64
 * ``frame_step`` and ``frame_length``, and an omitted or constant same-typed
 * one-dimensional window whose length equals ``frame_length``. Both values of
 * ``onesided`` are supported. To bound optimizer memory use, each generated
 * DFT weight tensor is limited to 16,777,216 elements.
 */
class STFTDecompositionPattern final : public core::builder::PatternOptimization {
public:
  explicit STFTDecompositionPattern(int priority = 0)
      : PatternOptimization(priority, "STFTDecomposition") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
