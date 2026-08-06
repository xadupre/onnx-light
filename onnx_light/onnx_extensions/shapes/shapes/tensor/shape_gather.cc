// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeGather(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Gather", "ComputeShapeGather");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeGather: Gather requires two inputs (data, indices).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &indices = ctx.Get(node.input(1));

  const TensorType dtype = data.Dtype();
  const SymShape &data_shape = data.Shape();
  const SymShape &idx_shape = indices.Shape();
  const int64_t r = static_cast<int64_t>(data_shape.Rank());
  const int64_t q = static_cast<int64_t>(idx_shape.Rank());

  int64_t axis = GetAttributeOr<int64_t>(node, "axis", 0);
  if (r > 0) {
    if (axis < 0) {
      axis += r;
    }
    EXT_ENFORCE_INVALID(!(axis < 0 || axis >= r), "ComputeShapeGather: axis=", axis,
                        " out of range for data rank ", r, ".");
  }

  SymShape out_shape;
  if (r == 0) {
    // Unknown data rank: produce symbolic shape of rank q (best effort).
    for (int64_t i = 0; i < q; ++i) {
      out_shape.PushBack(SymDim("Gather_dim" + std::to_string(i)));
    }
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
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

  SymTensor out_tensor(nullptr, dtype, std::move(out_shape));

  // Propagate ValueAsShape when data is 1-D with a value-as-shape annotation
  // and the indices are known constants (axis must be 0 for a 1-D tensor).
  if (r == 1 && axis == 0 && data.HasValueAsShape() && indices.HasValueAsShape()) {
    const SymShape &data_vas = data.ValueAsShape();
    const SymShape &idx_vas = indices.ValueAsShape();
    const std::size_t data_len = data_vas.Rank();
    SymShape out_vas;
    bool valid = true;
    for (std::size_t i = 0; i < idx_vas.Rank(); ++i) {
      if (!idx_vas[i].IsInt()) {
        valid = false;
        break;
      }
      int64_t idx = idx_vas[i].AsInt();
      if (idx < 0) {
        idx += static_cast<int64_t>(data_len);
      }
      if (idx < 0 || static_cast<std::size_t>(idx) >= data_len) {
        valid = false;
        break;
      }
      out_vas.PushBack(data_vas[static_cast<std::size_t>(idx)]);
    }
    if (valid) {
      out_tensor.SetValueAsShape(std::move(out_vas));
    }
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
