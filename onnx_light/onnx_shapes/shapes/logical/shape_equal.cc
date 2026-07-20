// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_shapes/shapes/logical/shape_logical.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_shapes {
namespace shapes {
namespace logical {

void ComputeShapeEqual(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  // Equal is element-wise with numpy-style broadcasting in every
  // currently-supported opset revision; the output dtype is bool.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "Equal", TensorType::kBool);
}

} // namespace logical
} // namespace shapes
} // namespace onnx_shapes
} // namespace ONNX_LIGHT_NAMESPACE
