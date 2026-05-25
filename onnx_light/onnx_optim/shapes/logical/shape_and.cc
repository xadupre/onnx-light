// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/shape_broadcast.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace logical {

void ComputeShapeAnd(ShapesContext &ctx, const NodeProto &node, const std::string &a,
                     const std::string &b) {
  // And is element-wise with numpy-style broadcasting in every
  // currently-supported opset revision; the output dtype is bool.
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "And", TensorType::kBool);
}

} // namespace logical
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
