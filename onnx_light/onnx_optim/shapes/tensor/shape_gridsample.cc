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

// Merges two dims: if both concrete and equal, keep; concrete wins over
// symbolic; symbolic + symbolic keeps the first one.
OptimDim MergeDim(const OptimDim &a, const OptimDim &b, const char *axis_name) {
  if (a.IsInt() && b.IsInt()) {
    if (a.AsInt() != b.AsInt()) {
      throw std::invalid_argument(std::string("ComputeShapeGridSample: ") + axis_name +
                                  " dimensions disagree (" + std::to_string(a.AsInt()) + " vs " +
                                  std::to_string(b.AsInt()) + ").");
    }
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

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeGridSample: GridSample requires two inputs (X, grid).");
  }

  const OptimTensor &x = ctx.Get(node.input(0).as_string());
  const OptimTensor &grid = ctx.Get(node.input(1).as_string());

  const OptimShape &x_shape = x.Shape();
  const OptimShape &grid_shape = grid.Shape();

  // If either rank is unknown (Rank() == 0 in this codebase represents an
  // unranked tensor in our shape lattice), produce a best-effort output of
  // unknown rank.
  if (x_shape.Rank() == 0 || grid_shape.Rank() == 0) {
    OptimShape out;
    out.PushBack(OptimDim("GridSample_dim0"));
    ctx.Set(node.output(0), OptimTensor(nullptr, x.Dtype(), std::move(out)));
    return;
  }

  if (x_shape.Rank() != grid_shape.Rank()) {
    throw std::invalid_argument(
        "ComputeShapeGridSample: X and grid must have the same rank. Got X rank " +
        std::to_string(x_shape.Rank()) + " vs grid rank " + std::to_string(grid_shape.Rank()) +
        ".");
  }

  if (x_shape.Rank() < 3) {
    throw std::invalid_argument("ComputeShapeGridSample: X and grid ranks must be >= 3. Got " +
                                std::to_string(x_shape.Rank()) + ".");
  }

  const size_t rank = x_shape.Rank();

  // Validate that the trailing dim of grid (when known) equals rank - 2.
  const OptimDim &grid_last = grid_shape[rank - 1];
  if (grid_last.IsInt() && grid_last.AsInt() != static_cast<int64_t>(rank) - 2) {
    throw std::invalid_argument(
        "ComputeShapeGridSample: the last dimension of grid must equal the number of spatial "
        "dimensions (rank - 2 = " +
        std::to_string(rank - 2) + "). Got " + std::to_string(grid_last.AsInt()) + ".");
  }

  OptimShape out;
  // N: merged dim between X[0] and grid[0].
  out.PushBack(MergeDim(x_shape[0], grid_shape[0], "N"));
  // C: taken from X[1].
  out.PushBack(x_shape[1]);
  // Spatial dims taken from grid[1 .. rank-2].
  for (size_t i = 1; i + 1 < rank; ++i) {
    out.PushBack(grid_shape[i]);
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, x.Dtype(), std::move(out)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
