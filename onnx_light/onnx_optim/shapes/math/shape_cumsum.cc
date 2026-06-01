// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeCumSum(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CumSum", "ComputeShapeCumSum");
  const OptimTensor &input = ctx.Get(x);
  // CumSum produces an output with the same shape and dtype as ``x``;
  // the ``axis`` input (and the ``exclusive``/``reverse`` attributes) only
  // affects element values.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeCumProd(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CumProd", "ComputeShapeCumProd");
  const OptimTensor &input = ctx.Get(x);
  // CumProd produces an output with the same shape and dtype as ``x``.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
