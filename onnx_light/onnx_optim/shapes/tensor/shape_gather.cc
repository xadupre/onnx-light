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

void ComputeShapeGather(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Gather", "ComputeShapeGather");

  if (node.input_size() < 2) {
    throw std::invalid_argument("ComputeShapeGather: Gather requires two inputs (data, indices).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());
  const OptimTensor &indices = ctx.Get(node.input(1).as_string());

  const TensorType dtype = data.Dtype();
  const OptimShape &data_shape = data.Shape();
  const OptimShape &idx_shape = indices.Shape();
  const int64_t r = static_cast<int64_t>(data_shape.Rank());
  const int64_t q = static_cast<int64_t>(idx_shape.Rank());

  int64_t axis = GetAttributeOr<int64_t>(node, "axis", 0);
  if (r > 0) {
    if (axis < 0) {
      axis += r;
    }
    if (axis < 0 || axis >= r) {
      throw std::invalid_argument("ComputeShapeGather: axis=" + std::to_string(axis) +
                                  " out of range for data rank " + std::to_string(r) + ".");
    }
  }

  OptimShape out_shape;
  if (r == 0) {
    // Unknown data rank: produce symbolic shape of rank q (best effort).
    for (int64_t i = 0; i < q; ++i) {
      out_shape.PushBack(OptimDim("Gather_dim" + std::to_string(i)));
    }
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }
  for (int64_t i = 0; i < axis; ++i) {
    out_shape.PushBack(data_shape[static_cast<std::size_t>(i)]);
  }
  for (int64_t i = 0; i < q; ++i) {
    out_shape.PushBack(idx_shape[static_cast<std::size_t>(i)]);
  }
  for (int64_t i = axis + 1; i < r; ++i) {
    out_shape.PushBack(data_shape[static_cast<std::size_t>(i)]);
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
