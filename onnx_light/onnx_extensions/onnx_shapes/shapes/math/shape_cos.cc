// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_shapes {
namespace shapes {
namespace math {

void ComputeShapeCos(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Cos", "ComputeShapeCos");
  const SymTensor &input = ctx.Get(x);
  // Cos is element-wise in every supported opset revision: the output
  // dtype and shape match the input.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_shapes
} // namespace ONNX_LIGHT_NAMESPACE
