// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeTile(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Tile", "ComputeShapeTile");

  if (node.input_size() < 2) {
    throw std::invalid_argument("ComputeShapeTile: Tile requires two inputs (input, repeats).");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimTensor &repeats_input = ctx.Get(node.input(1).as_string());

  const TensorType dtype = input.Dtype();
  const OptimShape &in_shape = input.Shape();
  const int64_t input_rank = static_cast<int64_t>(in_shape.Rank());

  // When the repeats values are known via data-propagation, compute the
  // output shape entry-by-entry as input.shape[i] * repeats[i].
  if (repeats_input.HasValueAsShape()) {
    const OptimShape &repeats = repeats_input.ValueAsShape();
    if (static_cast<int64_t>(repeats.Rank()) != input_rank) {
      throw std::invalid_argument(
          "ComputeShapeTile: 'repeats' length (" + std::to_string(repeats.Rank()) +
          ") must equal the rank of 'input' (" + std::to_string(input_rank) + ").");
    }
    OptimShape out_shape;
    for (int64_t i = 0; i < input_rank; ++i) {
      const OptimDim &in_dim = in_shape[static_cast<std::size_t>(i)];
      const OptimDim &rep_dim = repeats[static_cast<std::size_t>(i)];
      if (in_dim.IsInt() && rep_dim.IsInt()) {
        out_shape.PushBack(OptimDim(in_dim.AsInt() * rep_dim.AsInt()));
      } else {
        out_shape.PushBack(OptimDim("Tile_dim" + std::to_string(i)));
      }
    }
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Fall back to rank inference using the static shape of the ``repeats``
  // input: it is a 1-D tensor whose single static dim gives the rank of
  // the output (which equals the rank of the input).
  OptimShape out_shape;
  if (repeats_input.Shape().Rank() == 1 && repeats_input.Shape()[0].IsInt()) {
    const int64_t rank = repeats_input.Shape()[0].AsInt();
    for (int64_t i = 0; i < rank; ++i) {
      out_shape.PushBack(OptimDim("Tile_dim" + std::to_string(i)));
    }
  } else {
    for (int64_t i = 0; i < input_rank; ++i) {
      out_shape.PushBack(OptimDim("Tile_dim" + std::to_string(i)));
    }
    if (out_shape.Rank() == 0) {
      out_shape.PushBack(OptimDim("Tile_dim0"));
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
