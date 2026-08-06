// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

namespace {

SymDim BatchDimFromInput(const SymTensor &input) {
  EXT_ENFORCE_INVALID(!(input.Shape().Empty()),
                      "ComputeShapeLinearRegressor: input rank must be 1 or 2 when known.");
  if (input.Shape().Rank() == 1) {
    return SymDim(1);
  }
  if (input.Shape().Rank() == 2) {
    return input.Shape()[0];
  }
  EXT_THROW_INVALID("ComputeShapeLinearRegressor: input rank must be 1 or 2 when known.");
}

} // namespace

void ComputeShapeLinearRegressor(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LinearRegressor", "ComputeShapeLinearRegressor");

  const SymTensor &input = ctx.Get(x);
  SymShape output_shape;
  output_shape.PushBack(BatchDimFromInput(input));

  const int64_t targets = GetAttributeOr(node, "targets", static_cast<int64_t>(1));
  EXT_ENFORCE_INVALID(targets >= 1,
                      "ComputeShapeLinearRegressor: 'targets' attribute must be >= 1.");
  output_shape.PushBack(SymDim(targets));

  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
