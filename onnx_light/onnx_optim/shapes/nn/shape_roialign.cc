// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Returns dim 0 of ``rois`` if it is static, otherwise dim 0 of
// ``batch_indices`` if static, otherwise the symbolic rois dim. The
// inputs are required to have ranks 2 and 1 respectively.
OptimDim PickNumRois(const OptimShape &rois_shape, const OptimShape &batch_indices_shape) {
  const OptimDim &from_rois = rois_shape[0];
  if (from_rois.IsInt()) {
    return from_rois;
  }
  const OptimDim &from_batch = batch_indices_shape[0];
  if (from_batch.IsInt()) {
    return from_batch;
  }
  return from_rois;
}

} // namespace

void ComputeShapeRoiAlign(ShapesContext &ctx, const NodeProto &node, const char *x,
                          const char *rois, const char *batch_indices) {
  CheckNodeOpAndOutput(node, "RoiAlign", "ComputeShapeRoiAlign");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  if (in_shape.Rank() != 4) {
    throw std::invalid_argument("ComputeShapeRoiAlign: input '" + std::string(x) +
                                "' must have rank 4 (N, C, H, W).");
  }

  const OptimShape &rois_shape = ctx.Get(rois).Shape();
  if (rois_shape.Rank() != 2) {
    throw std::invalid_argument("ComputeShapeRoiAlign: input '" + std::string(rois) +
                                "' must have rank 2 (num_rois, 4).");
  }

  const OptimShape &batch_indices_shape = ctx.Get(batch_indices).Shape();
  if (batch_indices_shape.Rank() != 1) {
    throw std::invalid_argument("ComputeShapeRoiAlign: input '" + std::string(batch_indices) +
                                "' must have rank 1 (num_rois,).");
  }

  const int64_t output_height = GetAttributeOr<int64_t>(node, "output_height", 1);
  const int64_t output_width = GetAttributeOr<int64_t>(node, "output_width", 1);
  if (output_height <= 0 || output_width <= 0) {
    throw std::invalid_argument(
        "ComputeShapeRoiAlign: attributes 'output_height' and 'output_width' must be positive.");
  }

  OptimShape out_shape;
  out_shape.PushBack(PickNumRois(rois_shape, batch_indices_shape));
  out_shape.PushBack(in_shape[1]);
  out_shape.PushBack(OptimDim(output_height));
  out_shape.PushBack(OptimDim(output_width));

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
