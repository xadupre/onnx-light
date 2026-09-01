// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Fuses the tanh approximation of Gelu.
 *
 * @code
 * Before:
 *               ┌─────┐
 *   x, three ──▶│ Pow │────▶ powered
 *               └─────┘
 *                                ┌─────┐
 *   powered, cubic coefficient ─▶│ Mul │────▶ cubic
 *                                └─────┘
 *               ┌─────┐
 *   x, cubic ──▶│ Add │────▶ polynomial
 *               └─────┘
 *                            ┌─────┐
 *   polynomial, tanh scale ─▶│ Mul │────▶ scaled polynomial
 *                            └─────┘
 *                         ┌──────┐
 *   scaled polynomial ───▶│ Tanh │────▶ tanh value
 *                         └──────┘
 *                     ┌─────┐
 *   tanh value, one ─▶│ Add │────▶ tanh term
 *                     └─────┘
 *                   ┌─────┐
 *   x, one half ───▶│ Mul │────▶ half input
 *                   └─────┘
 *                           ┌─────┐
 *   half input, tanh term ─▶│ Mul │────▶ y
 *                           └─────┘
 *
 * After:
 *          ┌──────┐
 *   x ────▶│ Gelu │────▶ y
 *          └──────┘
 * @endcode
 *
 * The replacement uses ``approximate="tanh"``. The cubic coefficient may also
 * be the float16 value ``0.044708251953125``; its usual value is ``0.044715``.
 * The exponent, tanh scale, final offset, and input scale are respectively
 * ``3``, ``0.7978515625``, ``1``, and ``0.5``. All decomposition
 * intermediates must be unshared, and the configured default-domain opset
 * minimum must be met.
 */
class GeluPattern final : public core::builder::PatternOptimization {
public:
  explicit GeluPattern(int priority = 0, int min_opset = 20, std::string domain = "")
      : PatternOptimization(priority, "Gelu"), min_opset_(min_opset), domain_(std::move(domain)) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
  std::string domain_;
};

/**
 * Fuses a Where-based LeakyRelu decomposition.
 *
 * @code
 * Before:
 *              ┌─────────┐
 *   x, zero ──▶│ Greater │────▶ condition
 *              └─────────┘
 *
 *               ┌─────┐
 *   x, alpha ──▶│ Mul │────▶ scaled x
 *               └─────┘
 *
 *                             ┌───────┐
 *   condition, x, scaled x ──▶│ Where │────▶ y
 *                             └───────┘
 *
 * After:
 *          ┌───────────┐
 *   x ────▶│ LeakyRelu │────▶ y
 *          └───────────┘
 * @endcode
 *
 * The comparison threshold must be exactly zero, the slope must be scalar,
 * and the Greater and Mul outputs must have no other consumers.
 */
class LeakyReluPattern final : public core::builder::PatternOptimization {
public:
  explicit LeakyReluPattern(int priority = 0, int min_opset = 6)
      : PatternOptimization(priority, "LeakyRelu"), min_opset_(min_opset) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
};

/**
 * Fuses a float16 mean cross-entropy loss decomposition.
 *
 * @code
 * Before:
 *                    ┌───────┐       ┌─────┐
 *   labels, -100 ───▶│ Equal │──────▶│ Not │────▶ valid
 *                    └───────┘       └─────┘
 *
 *                               ┌───────┐       ┌───────────┐
 *   valid, labels, zero_i64 ───▶│ Where │──────▶│ Unsqueeze │────▶ gather indices
 *                               └───────┘       └───────────┘
 *
 *              ┌────────────┐
 *   scores ───▶│ LogSoftmax │────▶ log probabilities
 *              └────────────┘
 *
 *                                       ┌────────────────┐       ┌─────────┐
 *   log probabilities, gather indices ─▶│ GatherElements │──────▶│ Squeeze │
 *                                       └────────────────┘       └─────────┘
 *                                                                    │
 *                                                                    ▼
 *                                                                 ┌─────┐
 *                                                                 │ Neg │───▶ losses
 *                                                                 └─────┘
 *
 *                             ┌───────┐
 *   valid, losses, zero_f16 ─▶│ Where │────▶ masked losses
 *                             └───────┘
 *
 *                     ┌──────┐       ┌───────────┐       ┌──────┐
 *   masked losses ───▶│ Cast │──────▶│ ReduceSum │──────▶│ Cast │────▶ numerator
 *                     └──────┘       └───────────┘       └──────┘
 *
 *              ┌──────┐       ┌───────────┐       ┌──────┐
 *   valid ────▶│ Cast │──────▶│ ReduceSum │──────▶│ Cast │────▶ denominator
 *              └──────┘       └───────────┘       └──────┘
 *
 *                            ┌─────┐
 *   numerator, denominator ─▶│ Div │────▶ loss
 *                            └─────┘
 *
 * After:
 *                     ┌─────────────────────────┐
 *   scores, labels ──▶│ SoftmaxCrossEntropyLoss │────▶ loss
 *                     └─────────────────────────┘
 * @endcode
 *
 * LogSoftmax, GatherElements, Unsqueeze, and Squeeze use axis one. Scores,
 * masked losses, and the result must be float16; labels and the index zero
 * must be int64. The reduction inputs are cast to float, both reductions use
 * ``keepdims=0``, and their results are cast back to float16. The fused loss
 * uses ``ignore_index=-100`` and ``reduction="mean"``.
 */
class SoftmaxCrossEntropyLossCastPattern final : public core::builder::PatternOptimization {
public:
  explicit SoftmaxCrossEntropyLossCastPattern(int priority = 0, int min_opset = 14,
                                              std::string domain = "")
      : PatternOptimization(priority, "SoftmaxCrossEntropyLossCast"), min_opset_(min_opset),
        domain_(std::move(domain)) {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

private:
  int min_opset_;
  std::string domain_;
};

/**
 * Replaces a maximum with scalar zero by Relu.
 *
 * @code
 * Before:
 *              ┌─────┐
 *   x, zero ──▶│ Max │────▶ y
 *              └─────┘
 *
 * After:
 *          ┌──────┐
 *   x ────▶│ Relu │────▶ y
 *          └──────┘
 * @endcode
 *
 * Exactly one Max input must be scalar zero. Supported element types are
 * float, float16, int16, and int32; integer inputs require opset 14 or newer.
 */
class MaxReluPattern final : public core::builder::PatternOptimization {
public:
  explicit MaxReluPattern(int priority = 1) : PatternOptimization(priority, "MaxRelu") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
