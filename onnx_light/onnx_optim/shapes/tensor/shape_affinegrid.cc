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

void ComputeShapeAffineGrid(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "AffineGrid", "ComputeShapeAffineGrid");

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeAffineGrid: AffineGrid requires two inputs (theta, size).");
  }

  const OptimTensor &theta = ctx.Get(node.input(0).as_string());
  const OptimTensor &size_tensor = ctx.Get(node.input(1).as_string());

  const OptimShape &theta_shape = theta.Shape();
  if (theta_shape.Rank() != 3) {
    throw std::invalid_argument(
        "ComputeShapeAffineGrid: theta must have rank 3 ((N, 2, 3) for 2D or (N, 3, 4) for 3D).");
  }

  // Determine 2-D vs 3-D from theta when both inner dims are static; fall
  // back to the length of ``size`` (its single dim) otherwise.
  enum class Mode { kUnknown, k2D, k3D };
  Mode mode = Mode::kUnknown;
  if (theta_shape[1].IsInt() && theta_shape[2].IsInt()) {
    const int64_t r = theta_shape[1].AsInt();
    const int64_t c = theta_shape[2].AsInt();
    if (r == 2 && c == 3) {
      mode = Mode::k2D;
    } else if (r == 3 && c == 4) {
      mode = Mode::k3D;
    } else {
      throw std::invalid_argument(
          "ComputeShapeAffineGrid: theta inner dims must be (2, 3) for 2D or (3, 4) for 3D.");
    }
  }

  // ``size`` is a 1-D INT64 tensor. Cross-check its single dim with ``mode``
  // when known.
  const OptimShape &size_shape = size_tensor.Shape();
  if (size_shape.Rank() != 1) {
    // Be lenient: when the size input shape is unknown we still produce a
    // best-effort output. But if the rank is wrong, bail out.
    if (size_shape.Rank() != 0) {
      throw std::invalid_argument("ComputeShapeAffineGrid: size must be a 1-D tensor.");
    }
  }
  if (size_shape.Rank() == 1 && size_shape[0].IsInt()) {
    const int64_t len = size_shape[0].AsInt();
    if (len != 4 && len != 5) {
      throw std::invalid_argument(
          "ComputeShapeAffineGrid: size must have 4 (2D) or 5 (3D) entries.");
    }
    const Mode from_size = (len == 4) ? Mode::k2D : Mode::k3D;
    if (mode == Mode::kUnknown) {
      mode = from_size;
    } else if (mode != from_size) {
      throw std::invalid_argument(
          "ComputeShapeAffineGrid: theta rank and size length disagree on 2D vs 3D.");
    }
  }

  // If we still don't know the mode, leave the output fully symbolic.
  OptimShape out_shape;
  out_shape.PushBack(theta_shape[0]); // N

  // Try to read the spatial dims from ``size``'s constant value, if known.
  const bool size_known = size_tensor.HasValueAsShape();
  const OptimShape *size_values = size_known ? &size_tensor.ValueAsShape() : nullptr;
  if (size_known) {
    const int64_t len = static_cast<int64_t>(size_values->Rank());
    if (len != 4 && len != 5) {
      throw std::invalid_argument(
          "ComputeShapeAffineGrid: size must have 4 (2D) or 5 (3D) entries.");
    }
    const Mode from_vals = (len == 4) ? Mode::k2D : Mode::k3D;
    if (mode == Mode::kUnknown) {
      mode = from_vals;
    } else if (mode != from_vals) {
      throw std::invalid_argument(
          "ComputeShapeAffineGrid: theta rank and size value length disagree on 2D vs 3D.");
    }
  }

  if (mode == Mode::kUnknown) {
    // Best effort: rank-unknown output. Mirror the convention used by other
    // shape-inference functions (single symbolic dim) so downstream nodes
    // can still consume the result.
    out_shape.PushBack(OptimDim("AffineGrid_dim0"));
    ctx.Set(node.output(0), OptimTensor(nullptr, theta.Dtype(), std::move(out_shape)));
    return;
  }

  if (mode == Mode::k2D) {
    if (size_known) {
      out_shape.PushBack((*size_values)[2]); // H
      out_shape.PushBack((*size_values)[3]); // W
    } else {
      out_shape.PushBack(OptimDim("AffineGrid_H"));
      out_shape.PushBack(OptimDim("AffineGrid_W"));
    }
    out_shape.PushBack(OptimDim(static_cast<int64_t>(2)));
  } else { // k3D
    if (size_known) {
      out_shape.PushBack((*size_values)[2]); // D
      out_shape.PushBack((*size_values)[3]); // H
      out_shape.PushBack((*size_values)[4]); // W
    } else {
      out_shape.PushBack(OptimDim("AffineGrid_D"));
      out_shape.PushBack(OptimDim("AffineGrid_H"));
      out_shape.PushBack(OptimDim("AffineGrid_W"));
    }
    out_shape.PushBack(OptimDim(static_cast<int64_t>(3)));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, theta.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
