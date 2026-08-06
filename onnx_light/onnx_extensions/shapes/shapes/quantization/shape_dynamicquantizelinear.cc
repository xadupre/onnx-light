// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/quantization/shape_quantization.h"

#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization {

void ComputeShapeDynamicQuantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "DynamicQuantizeLinear", "ComputeShapeDynamicQuantizeLinear");

  const SymTensor &input = ctx.Get(x);
  SymShape out_shape = input.Shape();

  // y: same shape as x, dtype uint8.
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kUint8, std::move(out_shape)));

  // y_scale: scalar float (rank 0).
  if (node.output_size() >= 2 && !node.output(1).empty()) {
    ctx.Set(node.output(1), SymTensor(nullptr, TensorType::kFloat, SymShape{}));
  }
  // y_zero_point: scalar uint8 (rank 0).
  if (node.output_size() >= 3 && !node.output(2).empty()) {
    ctx.Set(node.output(2), SymTensor(nullptr, TensorType::kUint8, SymShape{}));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization
