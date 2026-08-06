// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeScatterND(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ScatterND", "ComputeShapeScatterND");

  EXT_ENFORCE_INVALID(
      !(node.input_size() < 3),
      "ComputeShapeScatterND: ScatterND requires three inputs (data, indices, updates).");

  const SymTensor &data = ctx.Get(node.input(0));

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
