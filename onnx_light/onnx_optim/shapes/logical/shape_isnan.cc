// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/logical/shape_logical.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace logical {

void ComputeShapeIsNaN(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "IsNaN", "ComputeShapeIsNaN");
  const OptimTensor &input = ctx.Get(x);
  // IsNaN is element-wise on a floating-point tensor: the output dtype
  // is always BOOL and the output shape matches the input shape.
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kBool, input.Shape()));
}

} // namespace logical
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
