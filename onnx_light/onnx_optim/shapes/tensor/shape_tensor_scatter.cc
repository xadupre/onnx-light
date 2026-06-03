// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeTensorScatter(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "TensorScatter", "ComputeShapeTensorScatter");

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeTensorScatter: TensorScatter requires at least two inputs "
        "(past_cache, update).");
  }

  const OptimTensor &past_cache = ctx.Get(node.input(0).as_string());
  const OptimTensor &update = ctx.Get(node.input(1).as_string());

  const OptimShape &cache_shape = past_cache.Shape();
  const OptimShape &update_shape = update.Shape();

  if (cache_shape.Rank() < 2) {
    throw std::invalid_argument("ComputeShapeTensorScatter: 'past_cache' must have rank >= 2.");
  }
  if (update_shape.Rank() != cache_shape.Rank()) {
    throw std::invalid_argument(
        "ComputeShapeTensorScatter: 'past_cache' and 'update' must share rank.");
  }

  const int64_t rank = static_cast<int64_t>(cache_shape.Rank());
  int64_t axis = GetAttributeOr<int64_t>(node, "axis", static_cast<int64_t>(-2));
  if (axis < 0) {
    axis += rank;
  }
  if (axis <= 0 || axis >= rank) {
    throw std::invalid_argument(
        "ComputeShapeTensorScatter: 'axis' must designate a non-batch dimension in "
        "[1, rank-1].");
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      continue;
    }
    const OptimDim &a = cache_shape[static_cast<std::size_t>(i)];
    const OptimDim &b = update_shape[static_cast<std::size_t>(i)];
    if (a.IsInt() && b.IsInt() && a.AsInt() != b.AsInt()) {
      throw std::invalid_argument(
          "ComputeShapeTensorScatter: 'past_cache' and 'update' must agree on every "
          "dimension other than 'axis'.");
    }
  }

  if (node.input_size() >= 3 && !node.input(2).as_string().empty()) {
    const OptimTensor &write_indices = ctx.Get(node.input(2).as_string());
    if (write_indices.Shape().Rank() != 1) {
      throw std::invalid_argument("ComputeShapeTensorScatter: 'write_indices' must have rank 1.");
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, past_cache.Dtype(), cache_shape));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
