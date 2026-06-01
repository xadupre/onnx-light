// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeGatherND(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GatherND", "ComputeShapeGatherND");

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeGatherND: GatherND requires two inputs (data, indices).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());
  const OptimTensor &indices = ctx.Get(node.input(1).as_string());

  const TensorType dtype = data.Dtype();
  const OptimShape &data_shape = data.Shape();
  const OptimShape &idx_shape = indices.Shape();
  const int64_t r = static_cast<int64_t>(data_shape.Rank());
  const int64_t q = static_cast<int64_t>(idx_shape.Rank());

  const int64_t batch_dims = GetAttributeOr<int64_t>(node, "batch_dims", 0);

  OptimShape out_shape;
  if (q < 1) {
    // Cannot infer; emit fully symbolic output of unknown rank by leaving
    // out_shape empty (consistent with other shape inference fallbacks).
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }
  // Leading q - 1 dims come from indices.
  for (int64_t i = 0; i < q - 1; ++i) {
    out_shape.PushBack(idx_shape[static_cast<std::size_t>(i)]);
  }
  // Trailing dims come from data after batch_dims + indices_shape[-1].
  const OptimDim &k_last_dim = idx_shape[static_cast<std::size_t>(q - 1)];
  if (k_last_dim.IsInt() && r > 0) {
    const int64_t k_last = k_last_dim.AsInt();
    const int64_t start = batch_dims + k_last;
    if (start > r) {
      throw std::invalid_argument(
          "ComputeShapeGatherND: indices last dim + batch_dims exceeds data rank.");
    }
    for (int64_t i = start; i < r; ++i) {
      out_shape.PushBack(data_shape[static_cast<std::size_t>(i)]);
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
