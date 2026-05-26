// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_broadcast.h"
#include "onnx_optim/shapes/text/shape_text.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace text {

void ComputeShapeStringConcat(ShapesContext &ctx, const NodeProto &node, const char *a,
                              const char *b) {
  // StringConcat is element-wise with numpy-style broadcasting (since
  // opset 20); the output dtype is string.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "StringConcat", TensorType::kString);
}

} // namespace text
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
