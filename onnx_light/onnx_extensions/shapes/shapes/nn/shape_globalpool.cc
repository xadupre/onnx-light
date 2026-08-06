// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeGlobalPool(ShapesContext &ctx, const NodeProto &node, const char *x) {
  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 2,
                      "ComputeShapeGlobalPool: input must have rank >= 2 (N, C, D1, ...).");

  // Output shape: (N, C, 1, 1, ..., 1) — same rank as input.
  const size_t n_spatial = in_shape.Rank() - 2;
  SymShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_spatial; ++i) {
    out_shape.PushBack(SymDim(static_cast<int64_t>(1)));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
