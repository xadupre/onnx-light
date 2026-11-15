// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeLogCumSumExp(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LogCumSumExp", "ComputeShapeLogCumSumExp");
  const SymTensor &input = ctx.Get(x);
  const int64_t rank = static_cast<int64_t>(input.Shape().Rank());
  EXT_ENFORCE_INVALID(rank >= 1, "ComputeShapeLogCumSumExp: input rank must be >= 1.");
  const int64_t axis = GetAttributeOr<int64_t>(node, "axis", 0);
  EXT_ENFORCE_INVALID(axis >= -rank && axis < rank,
                      "ComputeShapeLogCumSumExp: axis is out of range.");
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
