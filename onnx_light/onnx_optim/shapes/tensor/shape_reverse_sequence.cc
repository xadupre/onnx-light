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
  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeReverseSequence: ReverseSequence requires two inputs.");

  const OptimTensor &input = ctx.Get(node.input(0));
  EXT_ENFORCE_INVALID(!(input.Shape().Rank() < 2),
                      "ComputeShapeReverseSequence: input rank must be >= 2.");
  const OptimTensor &sequence_lens = ctx.Get(node.input(1));
  EXT_ENFORCE_INVALID(sequence_lens.Shape().Rank() == 1,
                      "ComputeShapeReverseSequence: 'sequence_lens' must have rank of 1.");

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
