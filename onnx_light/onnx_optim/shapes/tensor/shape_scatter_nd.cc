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

void ComputeShapeScatterND(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ScatterND", "ComputeShapeScatterND");

  if (node.input_size() < 3) {
    throw std::invalid_argument(
        "ComputeShapeScatterND: ScatterND requires three inputs (data, indices, updates).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
