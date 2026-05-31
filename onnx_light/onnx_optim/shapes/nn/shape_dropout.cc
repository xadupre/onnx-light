// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

void CheckScalarInput(const ShapesContext &ctx, const char *name, const char *label) {
  if (name == nullptr) {
    return;
  }
  const std::string input_name(name);
  if (input_name.empty()) {
    return;
  }
  const OptimShape &shape = ctx.Get(input_name).Shape();
  if (shape.Rank() != 0u) {
    throw std::invalid_argument(std::string("ComputeShapeDropout: ") + label +
                                " of Dropout must be a scalar.");
  }
}

} // namespace

void ComputeShapeDropout(ShapesContext &ctx, const NodeProto &node, const char *data,
                         const char *ratio, const char *training_mode) {
  CheckNodeOpAndOutput(node, "Dropout", "ComputeShapeDropout");

  const OptimTensor &input = ctx.Get(data);
  const OptimShape &in_shape = input.Shape();
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), in_shape));

  CheckScalarInput(ctx, ratio, "Ratio");
  CheckScalarInput(ctx, training_mode, "training_mode");

  if (node.output_size() >= 2) {
    const std::string mask_name = node.output(1).as_string();
    if (!mask_name.empty()) {
      ctx.Set(mask_name, OptimTensor(nullptr, TensorType::kBool, in_shape));
    }
  }
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
