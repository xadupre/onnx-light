// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeTensorScatter(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "TensorScatter", "ComputeShapeTensorScatter");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeTensorScatter: TensorScatter requires at least two inputs "
                      "(past_cache, update).");

  const SymTensor &past_cache = ctx.Get(node.input(0));
  const SymTensor &update = ctx.Get(node.input(1));

  const SymShape &cache_shape = past_cache.Shape();
  const SymShape &update_shape = update.Shape();

  EXT_ENFORCE_INVALID(!(cache_shape.Rank() < 2),
                      "ComputeShapeTensorScatter: 'past_cache' must have rank >= 2.");
  EXT_ENFORCE_INVALID(update_shape.Rank() == cache_shape.Rank(),
                      "ComputeShapeTensorScatter: 'past_cache' and 'update' must share rank.");

  const int64_t rank = static_cast<int64_t>(cache_shape.Rank());
  int64_t axis = GetAttributeOr<int64_t>(node, "axis", static_cast<int64_t>(-2));
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(!(axis <= 0 || axis >= rank),
                      "ComputeShapeTensorScatter: 'axis' must designate a non-batch dimension in "
                      "[1, rank-1].");

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      continue;
    }
    const SymDim &a = cache_shape[static_cast<std::size_t>(i)];
    const SymDim &b = update_shape[static_cast<std::size_t>(i)];
    EXT_ENFORCE_INVALID(!(a.IsInt() && b.IsInt() && a.AsInt() != b.AsInt()),
                        "ComputeShapeTensorScatter: 'past_cache' and 'update' must agree on every "
                        "dimension other than 'axis'.");
  }

  if (node.input_size() >= 3 && !node.input(2).empty()) {
    const SymTensor &write_indices = ctx.Get(node.input(2));
    EXT_ENFORCE_INVALID(write_indices.Shape().Rank() == 1,
                        "ComputeShapeTensorScatter: 'write_indices' must have rank 1.");
  }

  ctx.Set(node.output(0), SymTensor(nullptr, past_cache.Dtype(), cache_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
