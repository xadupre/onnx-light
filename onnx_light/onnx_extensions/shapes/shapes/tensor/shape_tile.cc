// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::tensor {

namespace {

// Converts a ``DimType`` back to an ``SymDim``.
SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

} // namespace

void ComputeShapeTile(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Tile", "ComputeShapeTile");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeTile: Tile requires two inputs (input, repeats).");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymTensor &repeats_input = ctx.Get(node.input(1));

  const TensorType dtype = input.Dtype();
  const SymShape &in_shape = input.Shape();
  const int64_t input_rank = static_cast<int64_t>(in_shape.Rank());

  // When the repeats values are known via data-propagation, compute the
  // output shape entry-by-entry as input.shape[i] * repeats[i].
  if (repeats_input.HasValueAsShape()) {
    const SymShape &repeats = repeats_input.ValueAsShape();
    EXT_ENFORCE_INVALID(!(static_cast<int64_t>(repeats.Rank()) != input_rank),
                        "ComputeShapeTile: 'repeats' length (", repeats.Rank(),
                        ") must equal the rank of 'input' (", input_rank, ").");
    SymShape out_shape;
    for (int64_t i = 0; i < input_rank; ++i) {
      const SymDim &in_dim = in_shape[static_cast<std::size_t>(i)];
      const SymDim &rep_dim = repeats[static_cast<std::size_t>(i)];
      if (in_dim.IsInt() && rep_dim.IsInt()) {
        out_shape.PushBack(SymDim(in_dim.AsInt() * rep_dim.AsInt()));
      } else if (rep_dim.IsInt()) {
        // The repeat count is known but the input dim is symbolic: compute
        // the product symbolically using the expressions library so that
        // downstream shape inference can further simplify the result.
        out_shape.PushBack(FromDimType(
            expressions::dim_mul(ToDimType(in_dim), expressions::DimType{rep_dim.AsInt()})));
      } else {
        out_shape.PushBack(SymDim("Tile_dim" + std::to_string(i)));
      }
    }
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Fall back to rank inference using the static shape of the ``repeats``
  // input: it is a 1-D tensor whose single static dim gives the rank of
  // the output (which equals the rank of the input).
  SymShape out_shape;
  if (repeats_input.Shape().Rank() == 1 && repeats_input.Shape()[0].IsInt()) {
    const int64_t rank = repeats_input.Shape()[0].AsInt();
    for (int64_t i = 0; i < rank; ++i) {
      out_shape.PushBack(SymDim("Tile_dim" + std::to_string(i)));
    }
  } else {
    for (int64_t i = 0; i < input_rank; ++i) {
      out_shape.PushBack(SymDim("Tile_dim" + std::to_string(i)));
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
