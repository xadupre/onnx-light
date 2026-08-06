// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

std::optional<std::vector<int64_t>> TryReadIntVector(const SymTensor &t) {
  if (!t.HasValueAsShape()) {
    return std::nullopt;
  }
  std::vector<int64_t> values;
  const SymShape &shape = t.ValueAsShape();
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
  EXT_ENFORCE_INVALID(step != 0, "ComputeShapeSlice: 'steps' entries cannot be 0.");
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

SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

expressions::DimType ResolveSliceIndex(const expressions::DimType &dim, int64_t index) {
  if (index >= 0) {
    return expressions::DimType{index};
  }
  return expressions::dim_add(dim, expressions::DimType{index});
}

SymDim SymbolicSliceLength(const SymDim &dim, int64_t start, int64_t end, int64_t step) {
  const expressions::DimType d = ToDimType(dim);
  const expressions::DimType start_expr = ResolveSliceIndex(d, start);
  const expressions::DimType end_expr = ResolveSliceIndex(d, end);

  const bool reverse = step < 0;
  const int64_t abs_step = reverse ? -step : step;
  expressions::DimType extent = reverse ? expressions::dim_sub(start_expr, end_expr)
                                        : expressions::dim_sub(end_expr, start_expr);
  if (abs_step == 1) {
    return FromDimType(extent);
  }
  extent = expressions::dim_add(extent, expressions::DimType{abs_step - 1});
  return FromDimType(expressions::dim_div(extent, expressions::DimType{abs_step}));
}

} // namespace

void ComputeShapeSlice(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Slice", "ComputeShapeSlice");
  EXT_ENFORCE_INVALID(
      !(node.input_size() < 3),
      "ComputeShapeSlice: Slice requires at least three inputs (data, starts, ends).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &starts_t = ctx.Get(node.input(1));
  const SymTensor &ends_t = ctx.Get(node.input(2));
  const SymShape &data_shape = data.Shape();
  const int64_t rank = static_cast<int64_t>(data_shape.Rank());

  SymShape out_shape = data_shape;

  const std::optional<std::vector<int64_t>> starts_opt = TryReadIntVector(starts_t);
  const std::optional<std::vector<int64_t>> ends_opt = TryReadIntVector(ends_t);
  if (!starts_opt.has_value() || !ends_opt.has_value()) {
    ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), std::move(out_shape)));
    return;
  }
  const std::vector<int64_t> &starts = *starts_opt;
  const std::vector<int64_t> &ends = *ends_opt;
  EXT_ENFORCE_INVALID(starts.size() == ends.size(),
                      "ComputeShapeSlice: starts and ends lengths must match.");

  std::vector<int64_t> axes;
  if (node.input_size() >= 4 && !node.input(3).empty()) {
    const std::optional<std::vector<int64_t>> axes_opt = TryReadIntVector(ctx.Get(node.input(3)));
    if (!axes_opt.has_value()) {
      ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), std::move(out_shape)));
      return;
    }
    axes = *axes_opt;
  } else {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) {
      axes[i] = static_cast<int64_t>(i);
    }
  }
  EXT_ENFORCE_INVALID(axes.size() == starts.size(),
                      "ComputeShapeSlice: axes length must match starts length.");

  std::vector<int64_t> steps;
  if (node.input_size() >= 5 && !node.input(4).empty()) {
    const std::optional<std::vector<int64_t>> steps_opt = TryReadIntVector(ctx.Get(node.input(4)));
    if (!steps_opt.has_value()) {
      ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), std::move(out_shape)));
      return;
    }
    steps = *steps_opt;
  } else {
    steps.assign(starts.size(), static_cast<int64_t>(1));
  }
  EXT_ENFORCE_INVALID(steps.size() == starts.size(),
                      "ComputeShapeSlice: steps length must match starts length.");
  for (size_t i = 0; i < steps.size(); ++i) {
    EXT_ENFORCE_INVALID(steps[i] != 0, "ComputeShapeSlice: 'steps' entries cannot be 0.");
  }

  for (size_t i = 0; i < starts.size(); ++i) {
    int64_t axis = axes[i];
    const int64_t step = steps[i];
    if (axis < 0) {
      axis += rank;
    }
    EXT_ENFORCE_INVALID(!(axis < 0 || axis >= rank), "ComputeShapeSlice: axis out of range.");
    if (!data_shape[static_cast<size_t>(axis)].IsInt()) {
      out_shape[static_cast<size_t>(axis)] =
          SymbolicSliceLength(data_shape[static_cast<size_t>(axis)], starts[i], ends[i], step);
      continue;
    }
    int64_t start = starts[i];
    int64_t end = ends[i];
    ProcessSliceInputs(data_shape[static_cast<size_t>(axis)].AsInt(), start, end, step);
    out_shape[static_cast<size_t>(axis)] = SymDim(SliceLength(start, end, step));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), std::move(out_shape)));
}

} // namespace shapes::tensor

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
