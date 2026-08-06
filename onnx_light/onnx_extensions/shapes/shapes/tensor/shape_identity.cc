// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeIdentity(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Identity", "ComputeShapeIdentity");
  EXT_ENFORCE_INVALID(node.input_size() >= 1, "ComputeShapeIdentity: Identity requires one input.");

  const std::string input_name = node.input(0);

  if (ctx.HasSequence(input_name)) {
    // Sequence input: propagate the sequence descriptor to the output.
    ctx.SetSequence(node.output(0), SymSequence(ctx.GetSequence(input_name)));
    return;
  }

  const SymTensor &input = ctx.Get(input_name);
  // Identity simply propagates the input dtype and shape (including any
  // symbolic dims) to the output unchanged.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
