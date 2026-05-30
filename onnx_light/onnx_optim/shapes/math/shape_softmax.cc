// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeSoftmax(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Softmax", "ComputeShapeSoftmax");
  const OptimTensor &input = ctx.Get(x);
  const int64_t rank = static_cast<int64_t>(input.Shape().Rank());
  EXT_ENFORCE_INVALID(rank >= 1, "ComputeShapeSoftmax: input rank must be >= 1.");

  const int opset = ctx.HasOpsetVersion(kOnnxDomain) ? ctx.OpsetVersion(kOnnxDomain) : 13;
  const int64_t default_axis = opset >= 13 ? int64_t{-1} : int64_t{1};
  const int64_t axis = GetAttributeOr<int64_t>(node, "axis", default_axis);
  const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "ComputeShapeSoftmax: axis " + std::to_string(axis) +
                          " is out of range for rank " + std::to_string(rank) + ".");

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
