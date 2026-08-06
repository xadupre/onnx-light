// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeLpNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LpNormalization", "ComputeShapeLpNormalization");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 1,
                      "ComputeShapeLpNormalization: input must have rank >= 1.");

  // Output has the same dtype and shape as the input.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), in_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
