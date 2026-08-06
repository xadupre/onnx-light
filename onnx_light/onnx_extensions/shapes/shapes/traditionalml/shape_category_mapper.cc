// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

void ComputeShapeCategoryMapper(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CategoryMapper", "ComputeShapeCategoryMapper");

  // CategoryMapper is a one-to-one mapping between strings and integers.
  // The output shape matches the input shape; the dtype flips between
  // STRING and INT64 based on the input dtype.
  const SymTensor &input = ctx.Get(x);
  TensorType output_dtype;
  if (input.Dtype() == TensorType::kString) {
    output_dtype = TensorType::kInt64;
  } else if (input.Dtype() == TensorType::kInt64) {
    output_dtype = TensorType::kString;
  } else {
    EXT_THROW_INVALID(
        "ComputeShapeCategoryMapper: CategoryMapper input must be a tensor of strings or "
        "int64s.");
  }
  ctx.Set(node.output(0), SymTensor(nullptr, output_dtype, input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
