// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeOneHot(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "OneHot", "ComputeShapeOneHot");
  EXT_ENFORCE_INVALID(!(node.input_size() < 3),
                      "ComputeShapeOneHot: OneHot requires three inputs (indices, depth, values).");

  const SymTensor &indices = ctx.Get(node.input(0));
  const SymTensor &depth = ctx.Get(node.input(1));
  const SymTensor &values = ctx.Get(node.input(2));

  const SymShape &indices_shape = indices.Shape();
  const std::size_t in_rank = indices_shape.Rank();
  const int64_t out_rank = static_cast<int64_t>(in_rank) + 1;

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", -1);
  EXT_ENFORCE_INVALID(
      !(axis_attr < -out_rank || axis_attr >= out_rank),
      "ComputeShapeOneHot: 'axis' is out of range [-rank(indices)-1, rank(indices)].");
  const int64_t axis_pos = axis_attr < 0 ? axis_attr + out_rank : axis_attr;

  // Resolve the depth dimension. When the value of the ``depth`` input is
  // known via data-propagation (encoded as a 1-D value-as-shape of length 1)
  // we use it; otherwise the dimension is left symbolic.
  SymDim depth_dim("OneHot_depth");
  if (depth.HasValueAsShape()) {
    const SymShape &val = depth.ValueAsShape();
    if (val.Rank() == 1 && val[0].IsInt()) {
      depth_dim = SymDim(val[0].AsInt());
    }
  }

  SymShape out_shape;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (i == axis_pos) {
      out_shape.PushBack(depth_dim);
    } else if (i < axis_pos) {
      out_shape.PushBack(indices_shape[static_cast<std::size_t>(i)]);
    } else {
      out_shape.PushBack(indices_shape[static_cast<std::size_t>(i - 1)]);
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, values.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
