// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeAsinh(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Asinh", "ComputeShapeAsinh");
  const SymTensor &input = ctx.Get(x);
  // Asinh is element-wise in every supported opset revision: the output
  // dtype and shape match the input.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
