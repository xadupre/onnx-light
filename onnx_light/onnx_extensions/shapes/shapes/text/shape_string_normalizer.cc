// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/text/shape_text.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text {

void ComputeShapeStringNormalizer(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "StringNormalizer", "ComputeShapeStringNormalizer");
  const SymTensor &input = ctx.Get(a);
  const SymShape &in_shape = input.Shape();

  // StringNormalizer only accepts [C] or [1, C] string tensors.
  const std::size_t rank = in_shape.Rank();
  if (rank == 1) {
    // The output rank matches the input rank; the only dimension is
    // symbolic because we cannot know how many stopwords will be
    // dropped at runtime.
    SymShape out_shape{SymDim("StringNormalizer(" + std::string(a) + ")")};
    ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kString, std::move(out_shape)));
    return;
  }
  if (rank == 2) {
    const SymDim &b_dim = in_shape[0];
    EXT_ENFORCE_INVALID(!(b_dim.IsInt() && b_dim.AsInt() != 1),
                        "ComputeShapeStringNormalizer: input shape must be [C] or [1, C]; "
                        "got a 2-D shape with leading dimension ",
                        b_dim.AsInt(), ".");
    SymShape out_shape{SymDim(static_cast<int64_t>(1)),
                       SymDim("StringNormalizer(" + std::string(a) + ")")};
    ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kString, std::move(out_shape)));
    return;
  }
  EXT_THROW_INVALID("ComputeShapeStringNormalizer: input shape must be [C] or [1, C]; got rank ",
                    rank, ".");
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::text
