// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeIdentity(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Identity", "ComputeShapeIdentity");
  EXT_ENFORCE_INVALID(node.input_size() >= 1, "ComputeShapeIdentity: Identity requires one input.");

  const std::string input_name = std::string(node.input(0));

  if (ctx.HasSequence(input_name)) {
    // Sequence input: propagate the sequence descriptor to the output.
    ctx.SetSequence(node.output(0), OptimSequence(ctx.GetSequence(input_name)));
    return;
  }

  const OptimTensor &input = ctx.Get(input_name);
  // Identity simply propagates the input dtype and shape (including any
  // symbolic dims) to the output unchanged.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
