// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeGroupedMatMul(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GroupedMatMul", "ComputeShapeGroupedMatMul");
  EXT_ENFORCE_INVALID(node.input_size() >= 3 && node.input_size() <= 4,
                      "ComputeShapeGroupedMatMul: expected three or four inputs.");
  const auto &input = ctx.Get(node.input(0));
  const auto &weights = ctx.Get(node.input(1));
  const auto &indices = ctx.Get(node.input(2));
  const auto &x = input.Shape();
  const auto &w = weights.Shape();
  const auto &i = indices.Shape();
  EXT_ENFORCE_INVALID(x.Rank() == 2 && w.Rank() == 3 && i.Rank() == 2,
                      "ComputeShapeGroupedMatMul: expected ranks 2, 3, 2.");
  auto unify = [](SymDim left, const SymDim &right) {
    if (left.IsInt() && right.IsInt()) {
      EXT_ENFORCE_INVALID(left.AsInt() == right.AsInt(),
                          "ComputeShapeGroupedMatMul: incompatible dimensions.");
    }
    return right.IsInt() ? right : left;
  };
  const SymDim m = unify(x[0], i[0]);
  unify(x[1], w[1]);
  SymDim n = w[2];
  if (node.input_size() > 3 && !node.input(3).empty()) {
    const auto &bias = ctx.Get(node.input(3));
    EXT_ENFORCE_INVALID(bias.Shape().Rank() == 2 && bias.Dtype() == input.Dtype(),
                        "ComputeShapeGroupedMatMul: invalid bias rank or dtype.");
    unify(w[0], bias.Shape()[0]);
    n = unify(n, bias.Shape()[1]);
  }
  EXT_ENFORCE_INVALID(input.Dtype() == weights.Dtype(),
                      "ComputeShapeGroupedMatMul: weights must match input dtype.");
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), SymShape{m, i[1], n}));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
