// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeRotaryEmbedding(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "RotaryEmbedding", "ComputeShapeRotaryEmbedding");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  if (in_shape.Rank() != 3 && in_shape.Rank() != 4) {
    throw std::invalid_argument(std::string("ComputeShapeRotaryEmbedding: input '") + x +
                                "' must have rank 3 or 4, got " + std::to_string(in_shape.Rank()) +
                                ".");
  }

  // Output has the same dtype and shape as the input ``X``.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), in_shape));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
