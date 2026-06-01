// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/shapes/shape_broadcast.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeSum(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Sum", "ComputeShapeSum");

  const int n_inputs = node.input_size();
  if (n_inputs < 1) {
    throw std::invalid_argument("ComputeShapeSum: Sum requires at least one input.");
  }

  // Start from the shape and dtype of the first input. Sum's type constraint
  // ``T`` requires every input to share the same float dtype, so the output
  // dtype is the dtype of the first input.
  const OptimTensor &first = ctx.Get(node.input(0).as_string());
  const TensorType out_dtype = first.Dtype();
  OptimShape out_shape = first.Shape();

  // Multidirectional broadcast across all remaining inputs.
  for (int i = 1; i < n_inputs; ++i) {
    const OptimTensor &cur = ctx.Get(node.input(i).as_string());
    out_shape = BroadcastShapes(out_shape, cur.Shape());
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
