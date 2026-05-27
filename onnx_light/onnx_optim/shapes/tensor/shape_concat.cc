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
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

namespace {

// Merges ``in`` into ``out``. Two concrete integer dimensions must be
// equal. A concrete integer wins over a symbolic dimension. Two
// symbolic dimensions are merged into ``out`` only when they share
// the same expression; otherwise the previously-merged value in
// ``out`` is preserved.
void MergeDim(OptimDim &out, const OptimDim &in, int axis, int input_index) {
  if (out.IsInt() && in.IsInt()) {
    if (out.AsInt() != in.AsInt()) {
      throw std::invalid_argument(
          "ComputeShapeConcat: input " + std::to_string(input_index) + " dimension " +
          std::to_string(axis) + " (" + std::to_string(in.AsInt()) +
          ") differs from the previously-merged value (" + std::to_string(out.AsInt()) + ").");
    }
    return;
  }
  if (in.IsInt()) {
    // Concrete value overrides a previously-symbolic dimension.
    out = in;
    return;
  }
  // ``in`` is symbolic; keep ``out`` as is.
}

} // namespace

void ComputeShapeConcat(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Concat", "ComputeShapeConcat");

  const int n_inputs = node.input_size();
  if (n_inputs < 1) {
    throw std::invalid_argument("ComputeShapeConcat: Concat requires at least one input.");
  }

  const OptimTensor &first = ctx.Get(node.input(0).as_string());
  const TensorType common_dtype = first.Dtype();
  const OptimShape &first_shape = first.Shape();
  const int rank = static_cast<int>(first_shape.Rank());

  // Resolve ``axis``. The default of ``1`` mirrors the opset 1
  // default; in later opsets ``axis`` is required so the default is
  // only used when the attribute was accidentally omitted.
  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 1);
  const int64_t resolved_axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  if (resolved_axis < 0 || resolved_axis >= rank) {
    throw std::invalid_argument("ComputeShapeConcat: axis " + std::to_string(axis_attr) +
                                " is out of range for rank " + std::to_string(rank) + ".");
  }
  const std::size_t axis = static_cast<std::size_t>(resolved_axis);

  // Start the output shape from the first input.
  OptimShape out_shape = first_shape;

  // Track running sum of the concat-axis dimension. Only meaningful
  // when every input's concat-axis dimension is a concrete integer.
  bool axis_dim_known = first_shape[axis].IsInt();
  int64_t axis_dim_total = axis_dim_known ? first_shape[axis].AsInt() : 0;

  for (int i = 1; i < n_inputs; ++i) {
    const OptimTensor &t = ctx.Get(node.input(i).as_string());
    if (t.Dtype() != common_dtype) {
      throw std::invalid_argument("ComputeShapeConcat: input '" + node.input(i).as_string() +
                                  "' has a dtype that differs from input '" +
                                  node.input(0).as_string() + "'.");
    }
    const OptimShape &shape = t.Shape();
    if (static_cast<int>(shape.Rank()) != rank) {
      throw std::invalid_argument("ComputeShapeConcat: input " + std::to_string(i) + " has rank " +
                                  std::to_string(shape.Rank()) + " != " + std::to_string(rank) +
                                  " (first input).");
    }
    for (std::size_t d = 0; d < shape.Rank(); ++d) {
      if (d == axis) {
        if (axis_dim_known && shape[d].IsInt()) {
          axis_dim_total += shape[d].AsInt();
        } else {
          axis_dim_known = false;
        }
      } else {
        MergeDim(out_shape[d], shape[d], static_cast<int>(d), i);
      }
    }
  }

  if (axis_dim_known) {
    out_shape[axis] = OptimDim(axis_dim_total);
  } else {
    out_shape[axis] = OptimDim("Concat_axis" + std::to_string(resolved_axis));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, common_dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
