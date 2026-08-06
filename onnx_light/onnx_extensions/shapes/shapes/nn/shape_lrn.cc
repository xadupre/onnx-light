// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeLRN(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LRN", "ComputeShapeLRN");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 2,
                      "ComputeShapeLRN: input must have rank >= 2 (N, C, D1, ...).");

  // Output has the same dtype and shape as the input.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), in_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
