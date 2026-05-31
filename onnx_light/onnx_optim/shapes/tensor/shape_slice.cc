// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

namespace {

std::optional<std::vector<int64_t>> TryReadIntVector(const OptimTensor &t) {
  if (!t.HasValueAsShape()) {
    return std::nullopt;
  }
  std::vector<int64_t> values;
  const OptimShape &shape = t.ValueAsShape();
  values.reserve(shape.Rank());
  for (size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt()) {
      return std::nullopt;
    }
    values.push_back(shape[i].AsInt());
  }
  return values;
}

void ProcessSliceInputs(const int64_t dim, int64_t &start, int64_t &end, int64_t step) {
  if (step == 0) {
    throw std::invalid_argument("ComputeShapeSlice: 'steps' entries cannot be 0.");
  }
  if (dim == 0) {
    start = 0;
    end = 0;
    return;
  }
  if (start < 0) {
    start += dim;
  }
  if (end < 0) {
    end += dim;
  }
  if (step < 0) {
    start = std::clamp(start, static_cast<int64_t>(0), dim - 1);
    end = std::clamp(end, static_cast<int64_t>(-1), dim - 1);
  } else {
    start = std::clamp(start, static_cast<int64_t>(0), dim);
    end = std::clamp(end, static_cast<int64_t>(0), dim);
  }
}

int64_t SliceLength(int64_t start, int64_t end, int64_t step) {
  if (step > 0) {
    if (end <= start) {
      return 0;
    }
    return 1 + (end - start - 1) / step;
  }
  if (end >= start) {
    return 0;
  }
  return 1 + (start - end - 1) / (-step);
}

} // namespace

void ComputeShapeSlice(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Slice", "ComputeShapeSlice");
  if (node.input_size() < 3) {
    throw std::invalid_argument(
        "ComputeShapeSlice: Slice requires at least three inputs (data, starts, ends).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());
  const OptimTensor &starts_t = ctx.Get(node.input(1).as_string());
  const OptimTensor &ends_t = ctx.Get(node.input(2).as_string());
  const OptimShape &data_shape = data.Shape();
  const int64_t rank = static_cast<int64_t>(data_shape.Rank());

  OptimShape out_shape;
  for (int64_t i = 0; i < rank; ++i) {
    out_shape.PushBack(OptimDim("Slice_dim" + std::to_string(i)));
  }

  const std::optional<std::vector<int64_t>> starts_opt = TryReadIntVector(starts_t);
  const std::optional<std::vector<int64_t>> ends_opt = TryReadIntVector(ends_t);
  if (!starts_opt.has_value() || !ends_opt.has_value()) {
    ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), std::move(out_shape)));
    return;
  }
  const std::vector<int64_t> &starts = *starts_opt;
  const std::vector<int64_t> &ends = *ends_opt;
  if (starts.size() != ends.size()) {
    throw std::invalid_argument("ComputeShapeSlice: starts and ends lengths must match.");
  }

  std::vector<int64_t> axes;
  if (node.input_size() >= 4 && !node.input(3).empty()) {
    const std::optional<std::vector<int64_t>> axes_opt =
        TryReadIntVector(ctx.Get(node.input(3).as_string()));
    if (!axes_opt.has_value()) {
      ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), std::move(out_shape)));
      return;
    }
    axes = *axes_opt;
  } else {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
  }
  if (axes.size() != starts.size()) {
    throw std::invalid_argument("ComputeShapeSlice: axes length must match starts length.");
  }

  std::vector<int64_t> steps;
  if (node.input_size() >= 5 && !node.input(4).empty()) {
    const std::optional<std::vector<int64_t>> steps_opt =
        TryReadIntVector(ctx.Get(node.input(4).as_string()));
    if (!steps_opt.has_value()) {
      ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), std::move(out_shape)));
      return;
    }
    steps = *steps_opt;
  } else {
    steps.assign(starts.size(), static_cast<int64_t>(1));
  }
  if (steps.size() != starts.size()) {
    throw std::invalid_argument("ComputeShapeSlice: steps length must match starts length.");
  }

  for (size_t i = 0; i < starts.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument("ComputeShapeSlice: axis out of range.");
    }
    if (!data_shape[static_cast<size_t>(axis)].IsInt()) {
      out_shape[static_cast<size_t>(axis)] = OptimDim("Slice_dim" + std::to_string(axis));
      continue;
    }
    int64_t start = starts[i];
    int64_t end = ends[i];
    const int64_t step = steps[i];
    ProcessSliceInputs(data_shape[static_cast<size_t>(axis)].AsInt(), start, end, step);
    out_shape[static_cast<size_t>(axis)] = OptimDim(SliceLength(start, end, step));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, data.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
