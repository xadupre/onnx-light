// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical {

void ComputeShapeIsNaN(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "IsNaN", "ComputeShapeIsNaN");
  const SymTensor &input = ctx.Get(x);
  // IsNaN is element-wise on a floating-point tensor: the output dtype
  // is always BOOL and the output shape matches the input shape.
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kBool, input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical
