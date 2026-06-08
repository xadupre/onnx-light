// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeCausalConvWithState(ShapesContext &ctx, const NodeProto &node, const char *input,
                                     const char *weight) {
  CheckNodeOpAndOutput(node, "CausalConvWithState", "ComputeShapeCausalConvWithState");
  if (node.output_size() < 2) {
    throw std::invalid_argument("ComputeShapeCausalConvWithState: node must declare two outputs.");
  }

  const OptimTensor &in = ctx.Get(input);
  const OptimTensor &w = ctx.Get(weight);
  const OptimShape &in_shape = in.Shape();
  const OptimShape &w_shape = w.Shape();

  if (in_shape.Rank() != 3) {
    throw std::invalid_argument(std::string("ComputeShapeCausalConvWithState: input '") + input +
                                "' must have rank 3, got " + std::to_string(in_shape.Rank()) + ".");
  }
  if (w_shape.Rank() != 3) {
    throw std::invalid_argument(std::string("ComputeShapeCausalConvWithState: weight '") + weight +
                                "' must have rank 3, got " + std::to_string(w_shape.Rank()) + ".");
  }

  // Output 0 (``output``) has the same shape and dtype as ``input``.
  ctx.Set(node.output(0), OptimTensor(nullptr, in.Dtype(), in_shape));

  // Output 1 (``present_state``) has shape (B, C, K - 1). When the kernel
  // dimension is concrete we compute K-1; otherwise we leave the trailing
  // dim symbolic.
  OptimDim km1;
  if (w_shape[2].IsInt()) {
    km1 = OptimDim(w_shape[2].AsInt() - 1);
  } else {
    km1 = OptimDim(w_shape[2].AsExpr() + "-1");
  }
  OptimShape present_shape{in_shape[0], in_shape[1], km1};
  ctx.Set(node.output(1), OptimTensor(nullptr, in.Dtype(), present_shape));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
