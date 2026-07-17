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

  EXT_ENFORCE_INVALID(
      !(node.input_size() < 3),
      "ComputeShapeScatterND: ScatterND requires three inputs (data, indices, updates).");

  const OptimTensor &data = ctx.Get(std::string(node.input(0)));

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
