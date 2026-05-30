// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_optim/shapes/text/shape_text.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace text {

void ComputeShapeStringNormalizer(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "StringNormalizer", "ComputeShapeStringNormalizer");
  const OptimTensor &input = ctx.Get(a);
  const OptimShape &in_shape = input.Shape();

  // StringNormalizer only accepts [C] or [1, C] string tensors.
  const std::size_t rank = in_shape.Rank();
  if (rank == 1) {
    // The output rank matches the input rank; the only dimension is
    // symbolic because we cannot know how many stopwords will be
    // dropped at runtime.
    OptimShape out_shape{OptimDim("StringNormalizer(" + std::string(a) + ")")};
    ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kString, std::move(out_shape)));
    return;
  }
  if (rank == 2) {
    const OptimDim &b_dim = in_shape[0];
    if (b_dim.IsInt() && b_dim.AsInt() != 1) {
      throw std::invalid_argument(
          "ComputeShapeStringNormalizer: input shape must be [C] or [1, C]; "
          "got a 2-D shape with leading dimension " +
          std::to_string(b_dim.AsInt()) + ".");
    }
    OptimShape out_shape{OptimDim(static_cast<int64_t>(1)),
                         OptimDim("StringNormalizer(" + std::string(a) + ")")};
    ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kString, std::move(out_shape)));
    return;
  }
  throw std::invalid_argument(
      "ComputeShapeStringNormalizer: input shape must be [C] or [1, C]; got rank " +
      std::to_string(rank) + ".");
}

} // namespace text
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
