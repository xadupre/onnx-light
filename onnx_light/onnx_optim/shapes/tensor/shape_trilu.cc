// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <stdexcept>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeTrilu(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Trilu", "ComputeShapeTrilu");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeTrilu: Trilu requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  if (input.Shape().Rank() < 2) {
    throw std::invalid_argument("ComputeShapeTrilu: input rank must be >= 2.");
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
