// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_extensions/shapes/shapes/logical/shape_logical.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical {

void ComputeShapeWhere(ShapesContext &ctx, const NodeProto &node, const char *condition,
                       const char *x, const char *y) {
  EXT_ENFORCE_INVALID(node.op_type() == "Where",
                      "ComputeShapeWhere expects node.op_type() == 'Where'.");
  EXT_ENFORCE_INVALID(node.output_size() > 0, "ComputeShapeWhere expects at least one output.");

  const core::symbolic::SymTensor &condition_tensor = ctx.Get(condition);
  const core::symbolic::SymTensor &x_tensor = ctx.Get(x);
  const core::symbolic::SymTensor &y_tensor = ctx.Get(y);

  const core::symbolic::SymShape xy_shape = BroadcastShapes(x_tensor.Shape(), y_tensor.Shape());
  const core::symbolic::SymShape out_shape = BroadcastShapes(condition_tensor.Shape(), xy_shape);

  ctx.Set(node.output(0),
          core::symbolic::SymTensor(nullptr, x_tensor.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::logical
