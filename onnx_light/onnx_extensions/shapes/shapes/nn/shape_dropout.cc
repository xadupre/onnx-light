// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

namespace {

void CheckScalarInput(const ShapesContext &ctx, const char *name, const char *label) {
  if (name == nullptr) {
    return;
  }
  const std::string input_name(name);
  if (input_name.empty()) {
    return;
  }
  const SymShape &shape = ctx.Get(input_name).Shape();
  EXT_ENFORCE_INVALID(shape.Rank() == 0u, "ComputeShapeDropout: ", label,
                      " of Dropout must be a scalar.");
}

} // namespace

void ComputeShapeDropout(ShapesContext &ctx, const NodeProto &node, const char *data,
                         const char *ratio, const char *training_mode) {
  CheckNodeOpAndOutput(node, "Dropout", "ComputeShapeDropout");

  const SymTensor &input = ctx.Get(data);
  const SymShape &in_shape = input.Shape();
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), in_shape));

  CheckScalarInput(ctx, ratio, "Ratio");
  CheckScalarInput(ctx, training_mode, "training_mode");

  if (node.output_size() >= 2) {
    const std::string mask_name = node.output(1);
    if (!mask_name.empty()) {
      ctx.Set(mask_name, SymTensor(nullptr, TensorType::kBool, in_shape));
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
