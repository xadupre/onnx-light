// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeLpNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LpNormalization", "ComputeShapeLpNormalization");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 1,
                      "ComputeShapeLpNormalization: input must have rank >= 1.");

  // Output has the same dtype and shape as the input.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), in_shape));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
