// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeTanh(ShapesContext &ctx, const NodeProto &node, const char *x_name) {
  CheckNodeOpAndOutput(node, "Tanh", "ComputeShapeTanh");
  const OptimTensor &input = ctx.Get(x_name);
  // Tanh is element-wise in every supported opset revision: the output
  // dtype and shape match the input.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
