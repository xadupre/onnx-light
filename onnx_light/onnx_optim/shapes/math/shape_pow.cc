// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shape_broadcast.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapePow(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  // Pow is element-wise with numpy-style broadcasting in every
  // currently-supported opset revision; the output dtype follows the
  // first input dtype.
  const TensorType out_dtype = ctx.Get(a).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "Pow", out_dtype);
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
