// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_broadcast.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeExpand(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Expand", "ComputeShapeExpand");

  if (node.input_size() < 2) {
    throw std::invalid_argument("ComputeShapeExpand: Expand requires two inputs (input, shape).");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimTensor &shape_input = ctx.Get(node.input(1).as_string());

  const TensorType dtype = input.Dtype();

  // When the target shape values are known via data-propagation, compute
  // the broadcast output shape directly.
  if (shape_input.HasValueAsShape()) {
    const OptimShape &target = shape_input.ValueAsShape();
    OptimShape out_shape = BroadcastShapes(input.Shape(), target);
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Fall back to rank inference using the static shape of the ``shape``
  // input: it is a 1-D tensor whose single static dim gives the output rank.
  OptimShape out_shape;
  if (shape_input.Shape().Rank() == 1 && shape_input.Shape()[0].IsInt()) {
    const int64_t rank = shape_input.Shape()[0].AsInt();
    for (int64_t i = 0; i < rank; ++i) {
      out_shape.PushBack(OptimDim("Expand_dim" + std::to_string(i)));
    }
  } else {
    out_shape.PushBack(OptimDim("Expand_dim0"));
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
