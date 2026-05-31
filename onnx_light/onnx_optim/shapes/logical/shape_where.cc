// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/shape_broadcast.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace logical {

void ComputeShapeWhere(ShapesContext &ctx, const NodeProto &node, const char *condition,
                       const char *x, const char *y) {
  EXT_ENFORCE_INVALID(node.op_type() == "Where",
                      "ComputeShapeWhere expects node.op_type() == 'Where'.");
  EXT_ENFORCE_INVALID(node.output_size() > 0, "ComputeShapeWhere expects at least one output.");

  const onnx_optim::OptimTensor &condition_tensor = ctx.Get(condition);
  const onnx_optim::OptimTensor &x_tensor = ctx.Get(x);
  const onnx_optim::OptimTensor &y_tensor = ctx.Get(y);

  const onnx_optim::OptimShape xy_shape = BroadcastShapes(x_tensor.Shape(), y_tensor.Shape());
  const onnx_optim::OptimShape out_shape = BroadcastShapes(condition_tensor.Shape(), xy_shape);

  ctx.Set(node.output(0).as_string(),
          onnx_optim::OptimTensor(nullptr, x_tensor.Dtype(), std::move(out_shape)));
}

} // namespace logical
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
