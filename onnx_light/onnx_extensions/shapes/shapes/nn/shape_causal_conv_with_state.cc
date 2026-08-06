// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeCausalConvWithState(ShapesContext &ctx, const NodeProto &node, const char *input,
                                     const char *weight) {
  CheckNodeOpAndOutput(node, "CausalConvWithState", "ComputeShapeCausalConvWithState");
  EXT_ENFORCE_INVALID(!(node.output_size() < 2),
                      "ComputeShapeCausalConvWithState: node must declare two outputs.");

  const SymTensor &in = ctx.Get(input);
  const SymTensor &w = ctx.Get(weight);
  const SymShape &in_shape = in.Shape();
  const SymShape &w_shape = w.Shape();

  EXT_ENFORCE_INVALID(in_shape.Rank() == 3, "ComputeShapeCausalConvWithState: input '", input,
                      "' must have rank 3, got ", in_shape.Rank(), ".");
  EXT_ENFORCE_INVALID(w_shape.Rank() == 3, "ComputeShapeCausalConvWithState: weight '", weight,
                      "' must have rank 3, got ", w_shape.Rank(), ".");

  // Output 0 (``output``) has the same shape and dtype as ``input``.
  ctx.Set(node.output(0), SymTensor(nullptr, in.Dtype(), in_shape));

  // Output 1 (``present_state``) has shape (B, C, K - 1). When the kernel
  // dimension is concrete we compute K-1; otherwise we leave the trailing
  // dim symbolic.
  SymDim km1;
  if (w_shape[2].IsInt()) {
    km1 = SymDim(w_shape[2].AsInt() - 1);
  } else {
    km1 = SymDim(w_shape[2].AsExpr() + "-1");
  }
  SymShape present_shape{in_shape[0], in_shape[1], km1};
  ctx.Set(node.output(1), SymTensor(nullptr, in.Dtype(), present_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
