// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeScatterElements(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ScatterElements", "ComputeShapeScatterElements");

  EXT_ENFORCE_INVALID(!(node.input_size() < 3),
                      "ComputeShapeScatterElements: ScatterElements requires three "
                      "inputs (data, indices, updates).");

  const OptimTensor &data = ctx.Get(node.input(0));

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
