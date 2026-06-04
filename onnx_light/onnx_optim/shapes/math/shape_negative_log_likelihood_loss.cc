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

void ComputeShapeNegativeLogLikelihoodLoss(ShapesContext &ctx, const NodeProto &node,
                                           const char *input, const char *target,
                                           const char *weight) {
  CheckNodeOpAndOutput(node, "NegativeLogLikelihoodLoss", "ComputeShapeNegativeLogLikelihoodLoss");

  const OptimTensor &input_tensor = ctx.Get(input);
  const OptimShape &input_shape = input_tensor.Shape();
  const int64_t input_rank = static_cast<int64_t>(input_shape.Rank());
  EXT_ENFORCE_INVALID(input_rank >= 2,
                      "ComputeShapeNegativeLogLikelihoodLoss: input rank must be >= 2 (got " +
                          std::to_string(input_rank) + ").");

  const OptimTensor &target_tensor = ctx.Get(target);
  const OptimShape &target_shape = target_tensor.Shape();
  const int64_t target_rank = static_cast<int64_t>(target_shape.Rank());
  EXT_ENFORCE_INVALID(
      target_rank == input_rank - 1,
      "ComputeShapeNegativeLogLikelihoodLoss: target rank must equal input rank - 1 (got "
      "input_rank=" +
          std::to_string(input_rank) + ", target_rank=" + std::to_string(target_rank) + ").");

  if (weight != nullptr && weight[0] != '\0') {
    const OptimTensor &weight_tensor = ctx.Get(weight);
    EXT_ENFORCE_INVALID(weight_tensor.Shape().Rank() == 1u,
                        "ComputeShapeNegativeLogLikelihoodLoss: weight rank must be 1 (got " +
                            std::to_string(weight_tensor.Shape().Rank()) + ").");
  }

  const std::string reduction = GetAttributeOr<std::string>(node, "reduction", std::string("mean"));

  if (reduction == "none") {
    // Shape matches target: (N, D1, D2, ..., Dk).
    ctx.Set(node.output(0), OptimTensor(nullptr, input_tensor.Dtype(), target_shape));
  } else {
    // Scalar output (rank 0).
    ctx.Set(node.output(0), OptimTensor(nullptr, input_tensor.Dtype(), OptimShape()));
  }
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
