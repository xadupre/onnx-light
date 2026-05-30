// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_optim/shapes/text/shape_text.h"

#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace text {

void ComputeShapeStringSplit(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "StringSplit", "ComputeShapeStringSplit");
  EXT_ENFORCE_INVALID(node.output_size() >= 2,
                      "ComputeShapeStringSplit: node must declare at least two outputs.");
  const std::string &y_name = node.output(0).as_string();
  const std::string &z_name = node.output(1).as_string();
  EXT_ENFORCE_INVALID(!y_name.empty() && !z_name.empty(),
                      "ComputeShapeStringSplit: both outputs must be named.");

  const OptimTensor &input = ctx.Get(a);
  OptimShape y_shape = input.Shape();
  y_shape.PushBack(OptimDim("StringSplit(" + std::string(a) + ")"));
  ctx.Set(y_name, OptimTensor(nullptr, TensorType::kString, std::move(y_shape)));
  ctx.Set(z_name, OptimTensor(nullptr, TensorType::kInt64, input.Shape()));
}

} // namespace text
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
