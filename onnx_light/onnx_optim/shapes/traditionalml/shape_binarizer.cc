// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

void ComputeShapeBinarizer(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Binarizer", "ComputeShapeBinarizer");

  // Binarizer is element-wise: the output has the same dtype and shape as the input.
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
