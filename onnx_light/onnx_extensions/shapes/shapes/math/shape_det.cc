// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeDet(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Det", "ComputeShapeDet");
  const SymTensor &input = ctx.Get(x);
  const SymShape &input_shape = input.Shape();
  EXT_ENFORCE_INVALID(!(input_shape.Rank() < 2), "ComputeShapeDet: input '", x,
                      "' must have rank >= 2.");
  // Validate the inner-most two dimensions match when both are concrete.
  const SymDim &h = input_shape[input_shape.Rank() - 2];
  const SymDim &w = input_shape[input_shape.Rank() - 1];
  EXT_ENFORCE_INVALID(!(h.IsInt() && w.IsInt() && h.AsInt() != w.AsInt()),
                      "ComputeShapeDet: inner-most 2 dimensions must be equal (got ", h.AsInt(),
                      " and ", w.AsInt(), ").");
  // Drop the trailing two dimensions to build the output shape.
  std::vector<SymDim> out_dims;
  out_dims.reserve(input_shape.Rank() - 2);
  for (std::size_t i = 0; i + 2 < input_shape.Rank(); ++i) {
    out_dims.push_back(input_shape[i]);
  }
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), SymShape(out_dims)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
