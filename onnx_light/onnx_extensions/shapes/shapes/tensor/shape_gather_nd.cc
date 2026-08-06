// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeGatherND(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GatherND", "ComputeShapeGatherND");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeGatherND: GatherND requires two inputs (data, indices).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &indices = ctx.Get(node.input(1));

  const TensorType dtype = data.Dtype();
  const SymShape &data_shape = data.Shape();
  const SymShape &idx_shape = indices.Shape();
  const int64_t r = static_cast<int64_t>(data_shape.Rank());
  const int64_t q = static_cast<int64_t>(idx_shape.Rank());

  const int64_t batch_dims = GetAttributeOr<int64_t>(node, "batch_dims", 0);

  SymShape out_shape;
  if (q < 1) {
    // Cannot infer; emit fully symbolic output of unknown rank by leaving
    // out_shape empty (consistent with other shape inference fallbacks).
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }
  // Leading q - 1 dims come from indices.
  for (int64_t i = 0; i < q - 1; ++i) {
    out_shape.PushBack(idx_shape[static_cast<std::size_t>(i)]);
  }
  // Trailing dims come from data after batch_dims + indices_shape[-1].
  const SymDim &k_last_dim = idx_shape[static_cast<std::size_t>(q - 1)];
  if (k_last_dim.IsInt() && r > 0) {
    const int64_t k_last = k_last_dim.AsInt();
    const int64_t start = batch_dims + k_last;
    EXT_ENFORCE_INVALID(!(start > r),
                        "ComputeShapeGatherND: indices last dim + batch_dims exceeds data rank.");
    for (int64_t i = start; i < r; ++i) {
      out_shape.PushBack(data_shape[static_cast<std::size_t>(i)]);
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
