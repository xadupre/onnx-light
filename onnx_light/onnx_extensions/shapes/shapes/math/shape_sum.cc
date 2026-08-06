// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include <utility>

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeSum(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Sum", "ComputeShapeSum");

  const int n_inputs = node.input_size();
  EXT_ENFORCE_INVALID(!(n_inputs < 1), "ComputeShapeSum: Sum requires at least one input.");

  // Start from the shape and dtype of the first input. Sum's type constraint
  // ``T`` requires every input to share the same float dtype, so the output
  // dtype is the dtype of the first input.
  const SymTensor &first = ctx.Get(node.input(0));
  const TensorType out_dtype = first.Dtype();
  SymShape out_shape = first.Shape();

  // Multidirectional broadcast across all remaining inputs.
  for (int i = 1; i < n_inputs; ++i) {
    const SymTensor &cur = ctx.Get(node.input(i));
    out_shape = BroadcastShapes(out_shape, cur.Shape());
  }

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
