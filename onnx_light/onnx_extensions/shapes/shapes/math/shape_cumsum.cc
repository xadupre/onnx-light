// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeCumSum(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CumSum", "ComputeShapeCumSum");
  const SymTensor &input = ctx.Get(x);
  // CumSum produces an output with the same shape and dtype as ``x``;
  // the ``axis`` input (and the ``exclusive``/``reverse`` attributes) only
  // affects element values.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeCumProd(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CumProd", "ComputeShapeCumProd");
  const SymTensor &input = ctx.Get(x);
  // CumProd produces an output with the same shape and dtype as ``x``.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
