// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeGlobalPool(ShapesContext &ctx, const NodeProto &node, const char *x) {
  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 2,
                      "ComputeShapeGlobalPool: input must have rank >= 2 (N, C, D1, ...).");

  // Output shape: (N, C, 1, 1, ..., 1) — same rank as input.
  const size_t n_spatial = in_shape.Rank() - 2;
  OptimShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_spatial; ++i) {
    out_shape.PushBack(OptimDim(static_cast<int64_t>(1)));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
