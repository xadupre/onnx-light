// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_shapes {
namespace shapes {
namespace math {

void ComputeShapeHardSigmoid(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "HardSigmoid", "ComputeShapeHardSigmoid");
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_shapes
} // namespace ONNX_LIGHT_NAMESPACE
