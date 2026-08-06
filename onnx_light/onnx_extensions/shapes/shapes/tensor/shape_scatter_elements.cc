// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeScatterElements(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ScatterElements", "ComputeShapeScatterElements");

  EXT_ENFORCE_INVALID(!(node.input_size() < 3),
                      "ComputeShapeScatterElements: ScatterElements requires three "
                      "inputs (data, indices, updates).");

  const SymTensor &data = ctx.Get(node.input(0));

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
