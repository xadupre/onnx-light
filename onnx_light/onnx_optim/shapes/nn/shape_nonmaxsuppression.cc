// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeNonMaxSuppression(ShapesContext &ctx, const NodeProto &node, const char *boxes,
                                   const char *scores) {
  CheckNodeOpAndOutput(node, "NonMaxSuppression", "ComputeShapeNonMaxSuppression");

  const OptimShape &boxes_shape = ctx.Get(boxes).Shape();
  EXT_ENFORCE_INVALID(boxes_shape.Rank() == 3, "ComputeShapeNonMaxSuppression: input '", boxes,
                      "' must have rank 3 (num_batches, spatial_dimension, 4).");
  EXT_ENFORCE_INVALID(!(boxes_shape[2].IsInt() && boxes_shape[2].AsInt() != 4),
                      "ComputeShapeNonMaxSuppression: last dim of '", boxes, "' must be 4, got ",
                      boxes_shape[2].AsInt(), ".");

  const OptimShape &scores_shape = ctx.Get(scores).Shape();
  EXT_ENFORCE_INVALID(scores_shape.Rank() == 3, "ComputeShapeNonMaxSuppression: input '", scores,
                      "' must have rank 3 (num_batches, num_classes, spatial_dimension).");

  // The output is rank 2 with dtype INT64. The first dim depends on the
  // runtime values of the inputs and is therefore symbolic.
  OptimShape out_shape;
  const auto &out_name = node.output(0);
  out_shape.PushBack(OptimDim(std::string("NonMaxSuppression.num_selected_indices(") +
                              std::string(out_name.data(), out_name.size()) + ")"));
  out_shape.PushBack(OptimDim(static_cast<int64_t>(3)));
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
