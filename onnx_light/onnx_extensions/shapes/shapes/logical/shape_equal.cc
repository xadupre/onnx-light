// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical {

void ComputeShapeEqual(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  // Equal is element-wise with numpy-style broadcasting in every
  // currently-supported opset revision; the output dtype is bool.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "Equal", TensorType::kBool);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical
