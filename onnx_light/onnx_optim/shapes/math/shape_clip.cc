// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeClip(ShapesContext &ctx, const NodeProto &node, const char *x_name) {
  CheckNodeOpAndOutput(node, "Clip", "ComputeShapeClip");
  const OptimTensor &input = ctx.Get(x_name);
  // Clip is element-wise in every supported opset revision: the optional
  // ``min`` and ``max`` inputs (since opset 11) — or attributes (v1, v6) —
  // are scalars that do not influence the output dtype or shape.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
