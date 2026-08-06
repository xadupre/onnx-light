// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical {

void ComputeShapeGreater(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  // Greater is element-wise with numpy-style broadcasting in every
  // currently-supported opset revision; the output dtype is bool.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "Greater", TensorType::kBool);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical
