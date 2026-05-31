// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeNonZero(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "NonZero", "ComputeShapeNonZero");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeNonZero: NonZero requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());

  // The output is always a 2-D INT64 tensor whose first dimension is the
  // input rank (concrete integer) and whose second dimension is the number
  // of non-zero elements (a runtime value, kept symbolic).
  OptimShape out_shape;
  out_shape.PushBack(OptimDim(static_cast<int64_t>(input.Shape().Rank())));
  out_shape.PushBack(OptimDim("NonZero_" + node.output(0).as_string() + "_nnz"));

  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
