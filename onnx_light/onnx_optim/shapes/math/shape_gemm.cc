// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeGemm(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  CheckNodeOpAndOutput(node, "Gemm", "ComputeShapeGemm");

  const OptimTensor &tensor_a = ctx.Get(a);
  const OptimTensor &tensor_b = ctx.Get(b);

  EXT_ENFORCE_INVALID(tensor_a.Shape().Rank() == 2,
                      "ComputeShapeGemm: input A must have rank 2, got rank " +
                          std::to_string(tensor_a.Shape().Rank()) + ".");
  EXT_ENFORCE_INVALID(tensor_b.Shape().Rank() == 2,
                      "ComputeShapeGemm: input B must have rank 2, got rank " +
                          std::to_string(tensor_b.Shape().Rank()) + ".");

  const int64_t transA = GetAttributeOr<int64_t>(node, "transA", int64_t{0});
  const int64_t transB = GetAttributeOr<int64_t>(node, "transB", int64_t{0});

  // A: (M, K) if transA=0, (K, M) if transA=1 → output row count = dim(transA?1:0).
  // B: (K, N) if transB=0, (N, K) if transB=1 → output col count = dim(transB?0:1).
  const OptimDim dim_m = tensor_a.Shape()[transA ? 1 : 0];
  const OptimDim dim_n = tensor_b.Shape()[transB ? 0 : 1];

  const TensorType out_dtype = tensor_a.Dtype();
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, OptimShape{dim_m, dim_n}));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
