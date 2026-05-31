// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

OptimDim BatchDimFromInput(const OptimTensor &input) {
  if (input.Shape().Empty()) {
    throw std::invalid_argument("ComputeShapeSVMRegressor: input rank must be 1 or 2 when known.");
  }
  if (input.Shape().Rank() == 1) {
    return OptimDim(1);
  }
  if (input.Shape().Rank() == 2) {
    return input.Shape()[0];
  }
  throw std::invalid_argument("ComputeShapeSVMRegressor: input rank must be 1 or 2 when known.");
}

} // namespace

void ComputeShapeSVMRegressor(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "SVMRegressor", "ComputeShapeSVMRegressor");

  const OptimTensor &input = ctx.Get(x);
  OptimShape output_shape;
  output_shape.PushBack(BatchDimFromInput(input));
  output_shape.PushBack(OptimDim(1));
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
