// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <string>

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeExpand(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Expand", "ComputeShapeExpand");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeExpand: Expand requires two inputs (input, shape).");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymTensor &shape_input = ctx.Get(node.input(1));

  const TensorType dtype = input.Dtype();

  // When the target shape values are known via data-propagation, compute
  // the broadcast output shape directly.
  if (shape_input.HasValueAsShape()) {
    const SymShape &target = shape_input.ValueAsShape();
    SymShape out_shape = BroadcastShapes(input.Shape(), target);
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Fall back to rank inference using the static shape of the ``shape``
  // input: it is a 1-D tensor whose single static dim gives the output rank.
  SymShape out_shape;
  if (shape_input.Shape().Rank() == 1 && shape_input.Shape()[0].IsInt()) {
    const int64_t rank = shape_input.Shape()[0].AsInt();
    for (int64_t i = 0; i < rank; ++i) {
      out_shape.PushBack(SymDim("Expand_dim" + std::to_string(i)));
    }
  } else {
    out_shape.PushBack(SymDim("Expand_dim0"));
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
