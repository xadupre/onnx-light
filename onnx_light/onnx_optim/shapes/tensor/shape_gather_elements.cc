// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeGatherElements(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GatherElements", "ComputeShapeGatherElements");

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeGatherElements: GatherElements requires two inputs (data, indices).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());
  const OptimTensor &indices = ctx.Get(node.input(1).as_string());

  // Output has the same shape as indices and the same dtype as data.
  ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), indices.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
