// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_extensions/shapes/shapes/math/shape_math.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapePRelu(ShapesContext &ctx, const NodeProto &node, const char *x,
                       const char *slope) {
  // PRelu is element-wise binary; the output shape matches ``x``'s shape
  // (``slope`` is unidirectionally broadcastable to ``x``) and the output
  // dtype matches the shared input dtype (type constraint ``T``).
  // Multidirectional broadcasting is a strict superset of the
  // unidirectional broadcasting permitted by the ONNX schema, so the
  // same helper is reused.
  const TensorType out_dtype = ctx.Get(x).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, x, slope, "PRelu", out_dtype);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
