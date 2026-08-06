// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/text/shape_text.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text {

void ComputeShapeTfIdfVectorizer(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "TfIdfVectorizer", "ComputeShapeTfIdfVectorizer");

  std::vector<int64_t> ngram_indexes;
  EXT_ENFORCE_INVALID(
      !(!GetAttributeInts(node, "ngram_indexes", ngram_indexes) || ngram_indexes.empty() ||
        !std::all_of(ngram_indexes.cbegin(), ngram_indexes.cend(),
                     [](int64_t i) { return i >= 0; })),
      "ComputeShapeTfIdfVectorizer: ngram_indexes must be non-empty with no negative values.");
  const int64_t max_last_axis = *std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend()) + 1;

  const SymTensor &input = ctx.Get(a);
  const SymShape &in_shape = input.Shape();
  const std::size_t rank = in_shape.Rank();
  if (rank == 1) {
    SymShape out_shape{SymDim(max_last_axis)};
    ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kFloat, std::move(out_shape)));
    return;
  }
  if (rank == 2) {
    SymShape out_shape{in_shape[0], SymDim(max_last_axis)};
    ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kFloat, std::move(out_shape)));
    return;
  }
  EXT_THROW_INVALID("ComputeShapeTfIdfVectorizer: input tensor must have rank 1 or 2; got rank ",
                    rank, ".");
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text
