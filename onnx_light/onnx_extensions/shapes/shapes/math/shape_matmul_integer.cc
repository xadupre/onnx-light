// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include <string>
#include <vector>

#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeMatMulInteger(ShapesContext &ctx, const NodeProto &node, const char *a,
                               const char *b) {
  CheckNodeOpAndOutput(node, "MatMulInteger", "ComputeShapeMatMulInteger");

  const SymTensor &tensor_a = ctx.Get(a);
  const SymTensor &tensor_b = ctx.Get(b);
  const SymShape &shape_a = tensor_a.Shape();
  const SymShape &shape_b = tensor_b.Shape();

  EXT_ENFORCE_INVALID(shape_a.Rank() != 0,
                      "ComputeShapeMatMulInteger: input A must have rank >= 1, got rank 0.");
  EXT_ENFORCE_INVALID(shape_b.Rank() != 0,
                      "ComputeShapeMatMulInteger: input B must have rank >= 1, got rank 0.");

  auto promoted = [](const SymShape &s, bool left) {
    std::vector<SymDim> dims;
    if (s.Rank() == 1) {
      if (left) {
        dims = {SymDim(1), s[0]};
      } else {
        dims = {s[0], SymDim(1)};
      }
    } else {
      dims = s.Dims();
    }
    return SymShape(dims);
  };

  const SymShape a2 = promoted(shape_a, true);
  const SymShape b2 = promoted(shape_b, false);
  const SymDim k_left = a2[a2.Rank() - 1];
  const SymDim k_right = b2[b2.Rank() - 2];

  if (k_left.IsInt() && k_right.IsInt()) {
    EXT_ENFORCE_INVALID(k_left.AsInt() == k_right.AsInt(),
                        "ComputeShapeMatMulInteger: incompatible inner dimensions ",
                        std::to_string(k_left.AsInt()), " and ", std::to_string(k_right.AsInt()),
                        ".");
  }

  std::vector<SymDim> a_prefix_dims;
  std::vector<SymDim> b_prefix_dims;
  for (size_t i = 0; i + 2 < a2.Rank(); ++i) {
    a_prefix_dims.push_back(a2[i]);
  }
  for (size_t i = 0; i + 2 < b2.Rank(); ++i) {
    b_prefix_dims.push_back(b2[i]);
  }
  SymShape out_shape = BroadcastShapes(SymShape(a_prefix_dims), SymShape(b_prefix_dims));

  if (shape_a.Rank() != 1) {
    out_shape.PushBack(a2[a2.Rank() - 2]);
  }
  if (shape_b.Rank() != 1) {
    out_shape.PushBack(b2[b2.Rank() - 1]);
  }

  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kInt32, out_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
