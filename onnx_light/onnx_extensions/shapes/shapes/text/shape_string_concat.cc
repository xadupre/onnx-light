// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_extensions/shapes/shapes/text/shape_text.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text {

void ComputeShapeStringConcat(ShapesContext &ctx, const NodeProto &node, const char *a,
                              const char *b) {
  // StringConcat is element-wise with numpy-style broadcasting (since
  // opset 20); the output dtype is string.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "StringConcat", TensorType::kString);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text
