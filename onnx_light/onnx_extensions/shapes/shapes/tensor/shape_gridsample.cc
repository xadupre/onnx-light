// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

// Merges two dims: if both concrete and equal, keep; concrete wins over
// symbolic; symbolic + symbolic keeps the first one.
SymDim MergeDim(const SymDim &a, const SymDim &b, const char *axis_name) {
  if (a.IsInt() && b.IsInt()) {
    EXT_ENFORCE_INVALID(a.AsInt() == b.AsInt(), "ComputeShapeGridSample: ", axis_name,
                        " dimensions disagree (", a.AsInt(), " vs ", b.AsInt(), ").");
    return a;
  }
  if (a.IsInt()) {
    return a;
  }
  if (b.IsInt()) {
    return b;
  }
  return a;
}

} // namespace

void ComputeShapeGridSample(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "GridSample", "ComputeShapeGridSample");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeGridSample: GridSample requires two inputs (X, grid).");

  const SymTensor &x = ctx.Get(node.input(0));
  const SymTensor &grid = ctx.Get(node.input(1));

  const SymShape &x_shape = x.Shape();
  const SymShape &grid_shape = grid.Shape();

  // If either rank is unknown (Rank() == 0 in this codebase represents an
  // unranked tensor in our shape lattice), produce a best-effort output of
  // unknown rank.
  if (x_shape.Rank() == 0 || grid_shape.Rank() == 0) {
    SymShape out;
    out.PushBack(SymDim("GridSample_dim0"));
    ctx.Set(node.output(0), SymTensor(nullptr, x.Dtype(), std::move(out)));
    return;
  }

  EXT_ENFORCE_INVALID(x_shape.Rank() == grid_shape.Rank(),
                      "ComputeShapeGridSample: X and grid must have the same rank. Got X rank ",
                      x_shape.Rank(), " vs grid rank ", grid_shape.Rank(), ".");

  EXT_ENFORCE_INVALID(!(x_shape.Rank() < 3),
                      "ComputeShapeGridSample: X and grid ranks must be >= 3. Got ", x_shape.Rank(),
                      ".");

  const size_t rank = x_shape.Rank();

  // Validate that the trailing dim of grid (when known) equals rank - 2.
  const SymDim &grid_last = grid_shape[rank - 1];
  EXT_ENFORCE_INVALID(
      !(grid_last.IsInt() && grid_last.AsInt() != static_cast<int64_t>(rank) - 2),
      "ComputeShapeGridSample: the last dimension of grid must equal the number of spatial "
      "dimensions (rank - 2 = ",
      rank - 2, "). Got ", grid_last.AsInt(), ".");

  SymShape out;
  // N: merged dim between X[0] and grid[0].
  out.PushBack(MergeDim(x_shape[0], grid_shape[0], "N"));
  // C: taken from X[1].
  out.PushBack(x_shape[1]);
  // Spatial dims taken from grid[1 .. rank-2].
  for (size_t i = 1; i + 1 < rank; ++i) {
    out.PushBack(grid_shape[i]);
  }

  ctx.Set(node.output(0), SymTensor(nullptr, x.Dtype(), std::move(out)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
