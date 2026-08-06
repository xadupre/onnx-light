// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/text/shape_text.h"

#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text {

void ComputeShapeStringSplit(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "StringSplit", "ComputeShapeStringSplit");
  EXT_ENFORCE_INVALID(node.output_size() >= 2,
                      "ComputeShapeStringSplit: node must declare at least two outputs.");
  const std::string &y_name = node.output(0);
  const std::string &z_name = node.output(1);
  EXT_ENFORCE_INVALID(!y_name.empty() && !z_name.empty(),
                      "ComputeShapeStringSplit: both outputs must be named.");

  const SymTensor &input = ctx.Get(a);
  SymShape y_shape = input.Shape();
  y_shape.PushBack(SymDim("StringSplit(" + std::string(a) + ")"));
  ctx.Set(y_name, SymTensor(nullptr, TensorType::kString, std::move(y_shape)));
  ctx.Set(z_name, SymTensor(nullptr, TensorType::kInt64, input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text
