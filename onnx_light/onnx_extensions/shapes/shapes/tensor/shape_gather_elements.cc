// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeGatherElements(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GatherElements", "ComputeShapeGatherElements");

  EXT_ENFORCE_INVALID(
      !(node.input_size() < 2),
      "ComputeShapeGatherElements: GatherElements requires two inputs (data, indices).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &indices = ctx.Get(node.input(1));

  // Output has the same shape as indices and the same dtype as data.
  ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), indices.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
