// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include <algorithm>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeGemm(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  CheckNodeOpAndOutput(node, "Gemm", "ComputeShapeGemm");

  const SymTensor &tensor_a = ctx.Get(a);
  const SymTensor &tensor_b = ctx.Get(b);

  const bool extended = ctx.OpsetVersion(node.domain()) >= 29;
  EXT_ENFORCE_INVALID(extended ? tensor_a.Shape().Rank() >= 1 : tensor_a.Shape().Rank() == 2,
                      "ComputeShapeGemm: invalid input A rank ",
                      std::to_string(tensor_a.Shape().Rank()), ".");
  EXT_ENFORCE_INVALID(tensor_b.Shape().Rank() == 2,
                      "ComputeShapeGemm: input B must have rank 2, got rank ",
                      std::to_string(tensor_b.Shape().Rank()), ".");

  const int64_t transA = GetAttributeOr<int64_t>(node, "transA", int64_t{0});
  const int64_t transB = GetAttributeOr<int64_t>(node, "transB", int64_t{0});

  std::vector<SymDim> dims = tensor_a.Shape().Dims();
  if (transA && dims.size() > 1) {
    std::swap(dims[dims.size() - 2], dims.back());
  }
  const SymDim dim_n = tensor_b.Shape()[transB ? 0 : 1];
  if (extended) {
    const SymDim dim_k = tensor_b.Shape()[transB ? 1 : 0];
    EXT_ENFORCE_INVALID(!dims.back().IsInt() || !dim_k.IsInt() ||
                            dims.back().AsInt() == dim_k.AsInt(),
                        "ComputeShapeGemm: incompatible contraction dimensions.");
  }
  dims.back() = dim_n;
  if (extended && node.input_size() > 2 && !node.input(2).empty()) {
    const auto &bias = ctx.Get(node.input(2)).Shape();
    EXT_ENFORCE_INVALID(bias.Rank() <= dims.size(),
                        "ComputeShapeGemm: C rank exceeds output rank.");
    for (size_t i = 0; i < bias.Rank(); ++i) {
      const auto &out_dim = dims[dims.size() - bias.Rank() + i];
      EXT_ENFORCE_INVALID(!bias[i].IsInt() || !out_dim.IsInt() || bias[i].AsInt() == 1 ||
                              bias[i].AsInt() == out_dim.AsInt(),
                          "ComputeShapeGemm: C cannot broadcast to output.");
    }
  }
  const TensorType out_dtype = tensor_a.Dtype();
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, SymShape(dims)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
