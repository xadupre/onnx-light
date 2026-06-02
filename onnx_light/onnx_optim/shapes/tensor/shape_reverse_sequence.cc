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

void ComputeShapeReverseSequence(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ReverseSequence", "ComputeShapeReverseSequence");
  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeReverseSequence: ReverseSequence requires two inputs.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  if (input.Shape().Rank() < 2) {
    throw std::invalid_argument("ComputeShapeReverseSequence: input rank must be >= 2.");
  }
  const OptimTensor &sequence_lens = ctx.Get(node.input(1).as_string());
  if (sequence_lens.Shape().Rank() != 1) {
    throw std::invalid_argument(
        "ComputeShapeReverseSequence: 'sequence_lens' must have rank of 1.");
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
