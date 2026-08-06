// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

void ComputeShapeScaler(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Scaler", "ComputeShapeScaler");

  // Scaler is element-wise: the output has the same shape as the input.
  // The output dtype is always float per the ONNX schema, regardless of
  // the input dtype.
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kFloat, input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
