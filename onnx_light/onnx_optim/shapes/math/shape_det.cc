// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include "onnx_optim/shapes/shape_check.h"

#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeDet(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Det", "ComputeShapeDet");
  const OptimTensor &input = ctx.Get(x);
  const OptimShape &input_shape = input.Shape();
  if (input_shape.Rank() < 2) {
    throw std::invalid_argument("ComputeShapeDet: input '" + std::string(x) +
                                "' must have rank >= 2.");
  }
  // Validate the inner-most two dimensions match when both are concrete.
  const OptimDim &h = input_shape[input_shape.Rank() - 2];
  const OptimDim &w = input_shape[input_shape.Rank() - 1];
  if (h.IsInt() && w.IsInt() && h.AsInt() != w.AsInt()) {
    throw std::invalid_argument("ComputeShapeDet: inner-most 2 dimensions must be equal (got " +
                                std::to_string(h.AsInt()) + " and " + std::to_string(w.AsInt()) +
                                ").");
  }
  // Drop the trailing two dimensions to build the output shape.
  std::vector<OptimDim> out_dims;
  out_dims.reserve(input_shape.Rank() - 2);
  for (std::size_t i = 0; i + 2 < input_shape.Rank(); ++i) {
    out_dims.push_back(input_shape[i]);
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), OptimShape(out_dims)));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
