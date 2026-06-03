// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeSoftmaxCrossEntropyLoss(ShapesContext &ctx, const NodeProto &node,
                                         const char *scores, const char *labels,
                                         const char *weights) {
  CheckNodeOpAndOutput(node, "SoftmaxCrossEntropyLoss", "ComputeShapeSoftmaxCrossEntropyLoss");

  const OptimTensor &scores_tensor = ctx.Get(scores);
  const OptimShape &scores_shape = scores_tensor.Shape();
  const int64_t scores_rank = static_cast<int64_t>(scores_shape.Rank());
  EXT_ENFORCE_INVALID(scores_rank >= 2,
                      "ComputeShapeSoftmaxCrossEntropyLoss: scores rank must be >= 2 (got " +
                          std::to_string(scores_rank) + ").");

  const OptimTensor &labels_tensor = ctx.Get(labels);
  const OptimShape &labels_shape = labels_tensor.Shape();
  const int64_t labels_rank = static_cast<int64_t>(labels_shape.Rank());
  EXT_ENFORCE_INVALID(
      labels_rank == scores_rank - 1,
      "ComputeShapeSoftmaxCrossEntropyLoss: labels rank must equal scores rank - 1 (got "
      "scores_rank=" +
          std::to_string(scores_rank) + ", labels_rank=" + std::to_string(labels_rank) + ").");

  if (weights != nullptr && weights[0] != '\0') {
    const OptimTensor &weights_tensor = ctx.Get(weights);
    EXT_ENFORCE_INVALID(weights_tensor.Shape().Rank() == 1u,
                        "ComputeShapeSoftmaxCrossEntropyLoss: weights rank must be 1 (got " +
                            std::to_string(weights_tensor.Shape().Rank()) + ").");
  }

  const std::string reduction = GetAttributeOr<std::string>(node, "reduction", std::string("mean"));

  // First output: loss tensor.
  if (reduction == "none") {
    // Shape matches labels: (N, D1, D2, ..., Dk).
    ctx.Set(node.output(0), OptimTensor(nullptr, scores_tensor.Dtype(), labels_shape));
  } else {
    // Scalar output (rank 0).
    ctx.Set(node.output(0), OptimTensor(nullptr, scores_tensor.Dtype(), OptimShape()));
  }

  // Optional second output: log_prob has same shape and dtype as scores.
  if (node.output_size() >= 2) {
    const std::string log_prob_name = node.output(1).as_string();
    if (!log_prob_name.empty()) {
      ctx.Set(log_prob_name, OptimTensor(nullptr, scores_tensor.Dtype(), scores_shape));
    }
  }
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
