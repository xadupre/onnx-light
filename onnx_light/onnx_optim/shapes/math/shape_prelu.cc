// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shape_broadcast.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapePRelu(ShapesContext &ctx, const NodeProto &node, const char *x,
                       const char *slope) {
  // PRelu is element-wise binary; the output shape matches ``x``'s shape
  // (``slope`` is unidirectionally broadcastable to ``x``) and the output
  // dtype matches the shared input dtype (type constraint ``T``).
  // Multidirectional broadcasting (used here) is a strict superset of the
  // unidirectional broadcasting permitted by the ONNX schema, so it is safe
  // to reuse the same helper.
  const TensorType out_dtype = ctx.Get(x).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, x, slope, "PRelu", out_dtype);
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
