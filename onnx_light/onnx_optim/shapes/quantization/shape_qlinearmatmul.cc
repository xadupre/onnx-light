// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/quantization/shape_quantization.h"

#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_broadcast.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace quantization {

void ComputeShapeQLinearMatMul(ShapesContext &ctx, const NodeProto &node, const char *a,
                               const char *b, const char *y_zero_point) {
  CheckNodeOpAndOutput(node, "QLinearMatMul", "ComputeShapeQLinearMatMul");

  const OptimTensor &tensor_a = ctx.Get(a);
  const OptimTensor &tensor_b = ctx.Get(b);
  const OptimTensor &tensor_yzp = ctx.Get(y_zero_point);
  const OptimShape &shape_a = tensor_a.Shape();
  const OptimShape &shape_b = tensor_b.Shape();

  EXT_ENFORCE_INVALID(shape_a.Rank() != 0,
                      "ComputeShapeQLinearMatMul: input a must have rank >= 1, got rank 0.");
  EXT_ENFORCE_INVALID(shape_b.Rank() != 0,
                      "ComputeShapeQLinearMatMul: input b must have rank >= 1, got rank 0.");

  auto promoted = [](const OptimShape &s, bool left) {
    std::vector<OptimDim> dims;
    if (s.Rank() == 1) {
      if (left) {
        dims = {OptimDim(1), s[0]};
      } else {
        dims = {s[0], OptimDim(1)};
      }
    } else {
      dims = s.Dims();
    }
    return OptimShape(dims);
  };

  const OptimShape a2 = promoted(shape_a, true);
  const OptimShape b2 = promoted(shape_b, false);
  const OptimDim k_left = a2[a2.Rank() - 1];
  const OptimDim k_right = b2[b2.Rank() - 2];

  if (k_left.IsInt() && k_right.IsInt()) {
    EXT_ENFORCE_INVALID(k_left.AsInt() == k_right.AsInt(),
                        "ComputeShapeQLinearMatMul: incompatible inner dimensions " +
                            std::to_string(k_left.AsInt()) + " and " +
                            std::to_string(k_right.AsInt()) + ".");
  }

  std::vector<OptimDim> a_prefix_dims;
  std::vector<OptimDim> b_prefix_dims;
  for (size_t i = 0; i + 2 < a2.Rank(); ++i) {
    a_prefix_dims.push_back(a2[i]);
  }
  for (size_t i = 0; i + 2 < b2.Rank(); ++i) {
    b_prefix_dims.push_back(b2[i]);
  }
  OptimShape out_shape = BroadcastShapes(OptimShape(a_prefix_dims), OptimShape(b_prefix_dims));

  if (shape_a.Rank() != 1) {
    out_shape.PushBack(a2[a2.Rank() - 2]);
  }
  if (shape_b.Rank() != 1) {
    out_shape.PushBack(b2[b2.Rank() - 1]);
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, tensor_yzp.Dtype(), out_shape));
}

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
