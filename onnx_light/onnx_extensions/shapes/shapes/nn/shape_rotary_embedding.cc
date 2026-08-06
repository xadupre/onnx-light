// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeRotaryEmbedding(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "RotaryEmbedding", "ComputeShapeRotaryEmbedding");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(!(in_shape.Rank() != 3 && in_shape.Rank() != 4),
                      "ComputeShapeRotaryEmbedding: input '", x, "' must have rank 3 or 4, got ",
                      in_shape.Rank(), ".");

  // Output has the same dtype and shape as the input ``X``.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), in_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
