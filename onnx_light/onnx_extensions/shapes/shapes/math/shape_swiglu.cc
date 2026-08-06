// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeSwiGLU(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  CheckNodeOpAndOutput(node, "SwiGLU", "ComputeShapeSwiGLU");
  // SwiGLU requires A and B to have identical shapes: no broadcasting. The
  // output dtype and shape both match the gate input A (which equals B).
  (void)b;
  const SymTensor &input = ctx.Get(a);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
