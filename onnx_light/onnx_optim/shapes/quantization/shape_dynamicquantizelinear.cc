// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/quantization/shape_quantization.h"

#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace quantization {

void ComputeShapeDynamicQuantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "DynamicQuantizeLinear", "ComputeShapeDynamicQuantizeLinear");

  const OptimTensor &input = ctx.Get(x);
  OptimShape out_shape = input.Shape();

  // y: same shape as x, dtype uint8.
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kUint8, std::move(out_shape)));

  // y_scale: scalar float (rank 0).
  if (node.output_size() >= 2 && !node.output(1).as_string().empty()) {
    ctx.Set(node.output(1), OptimTensor(nullptr, TensorType::kFloat, OptimShape{}));
  }
  // y_zero_point: scalar uint8 (rank 0).
  if (node.output_size() >= 3 && !node.output(2).as_string().empty()) {
    ctx.Set(node.output(2), OptimTensor(nullptr, TensorType::kUint8, OptimShape{}));
  }
}

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
