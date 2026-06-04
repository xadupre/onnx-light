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

void ComputeShapeOneHot(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OneHot", "ComputeShapeOneHot");
  if (node.input_size() < 3) {
    throw std::invalid_argument(
        "ComputeShapeOneHot: OneHot requires three inputs (indices, depth, values).");
  }

  const OptimTensor &indices = ctx.Get(node.input(0).as_string());
  const OptimTensor &depth = ctx.Get(node.input(1).as_string());
  const OptimTensor &values = ctx.Get(node.input(2).as_string());

  const OptimShape &indices_shape = indices.Shape();
  const std::size_t in_rank = indices_shape.Rank();
  const int64_t out_rank = static_cast<int64_t>(in_rank) + 1;

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", -1);
  if (axis_attr < -out_rank || axis_attr >= out_rank) {
    throw std::invalid_argument(
        "ComputeShapeOneHot: 'axis' is out of range [-rank(indices)-1, rank(indices)].");
  }
  const int64_t axis_pos = axis_attr < 0 ? axis_attr + out_rank : axis_attr;

  // Resolve the depth dimension. When the value of the ``depth`` input is
  // known via data-propagation (encoded as a 1-D value-as-shape of length 1)
  // we use it; otherwise the dimension is left symbolic.
  OptimDim depth_dim("OneHot_depth");
  if (depth.HasValueAsShape()) {
    const OptimShape &val = depth.ValueAsShape();
    if (val.Rank() == 1 && val[0].IsInt()) {
      depth_dim = OptimDim(val[0].AsInt());
    }
  }

  OptimShape out_shape;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (i == axis_pos) {
      out_shape.PushBack(depth_dim);
    } else if (i < axis_pos) {
      out_shape.PushBack(indices_shape[static_cast<std::size_t>(i)]);
    } else {
      out_shape.PushBack(indices_shape[static_cast<std::size_t>(i - 1)]);
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, values.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
