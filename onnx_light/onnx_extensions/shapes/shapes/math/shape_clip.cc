// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeClip(ShapesContext &ctx, const NodeProto &node, const char *x_name) {
  CheckNodeOpAndOutput(node, "Clip", "ComputeShapeClip");
  const SymTensor &input = ctx.Get(x_name);
  // Clip is element-wise in every supported opset revision: the optional
  // ``min`` and ``max`` inputs (since opset 11) — or attributes (v1, v6) —
  // are scalars that do not influence the output dtype or shape.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
