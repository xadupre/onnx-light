// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_shapes/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_shapes {
namespace shapes {
namespace traditionalml {

void ComputeShapeImputer(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Imputer", "ComputeShapeImputer");

  // Imputer is element-wise: the output has the same dtype and shape as the input.
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_shapes
} // namespace ONNX_LIGHT_NAMESPACE
