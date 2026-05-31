// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_optim/shapes/text/shape_text.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace text {

void ComputeShapeRegexFullMatch(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "RegexFullMatch", "ComputeShapeRegexFullMatch");
  // RegexFullMatch is element-wise: output dtype is bool, output
  // shape equals the input shape.
  const OptimTensor &input = ctx.Get(a);
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kBool, input.Shape()));
}

} // namespace text
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
