// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical {

void ComputeShapeBitwiseAnd(ShapesContext &ctx, const NodeProto &node, const char *a,
                            const char *b) {
  // BitwiseAnd (opset 18) is element-wise with numpy-style broadcasting;
  // the output dtype matches the shared integer input dtype.
  const TensorType out_dtype = ctx.Get(a).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "BitwiseAnd", out_dtype);
}

void ComputeShapeBitwiseOr(ShapesContext &ctx, const NodeProto &node, const char *a,
                           const char *b) {
  const TensorType out_dtype = ctx.Get(a).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "BitwiseOr", out_dtype);
}

void ComputeShapeBitwiseXor(ShapesContext &ctx, const NodeProto &node, const char *a,
                            const char *b) {
  const TensorType out_dtype = ctx.Get(a).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "BitwiseXor", out_dtype);
}

void ComputeShapeBitwiseNot(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "BitwiseNot", "ComputeShapeBitwiseNot");
  const SymTensor &input = ctx.Get(x);
  // BitwiseNot (opset 18) is element-wise: the output dtype and shape
  // match the input.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeBitShift(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  // BitShift (opset 11) is element-wise with numpy-style broadcasting;
  // the output dtype matches the shared unsigned integer input dtype.
  const TensorType out_dtype = ctx.Get(a).Dtype();
  ComputeShapeBinaryBroadcast(ctx, node, a, b, "BitShift", out_dtype);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical
