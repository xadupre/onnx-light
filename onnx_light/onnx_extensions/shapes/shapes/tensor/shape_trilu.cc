// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeTrilu(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Trilu", "ComputeShapeTrilu");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1), "ComputeShapeTrilu: Trilu requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  EXT_ENFORCE_INVALID(!(input.Shape().Rank() < 2), "ComputeShapeTrilu: input rank must be >= 2.");

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
