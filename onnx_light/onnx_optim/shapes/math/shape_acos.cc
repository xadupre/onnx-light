// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeAcos(ShapesContext &ctx, const NodeProto &node, const char *x) {
  if (node.op_type() != "Acos") {
    throw std::invalid_argument("ComputeShapeAcos expects op_type='Acos', got '" +
                                node.op_type().as_string() + "'.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument("ComputeShapeAcos: node has no output.");
  }
  const OptimTensor &input = ctx.Get(x);
  // Acos is element-wise in every supported opset revision: the output
  // dtype and shape match the input.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
