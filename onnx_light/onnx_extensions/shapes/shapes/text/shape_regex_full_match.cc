// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/text/shape_text.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text {

void ComputeShapeRegexFullMatch(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "RegexFullMatch", "ComputeShapeRegexFullMatch");
  // RegexFullMatch is element-wise: output dtype is bool, output
  // shape equals the input shape.
  const SymTensor &input = ctx.Get(a);
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kBool, input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text
