// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

std::vector<int64_t> NormalizeAxes(const std::vector<int64_t> &axes, int64_t output_rank) {
  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t axis : axes) {
    const int64_t adjusted = axis < 0 ? axis + output_rank : axis;
    EXT_ENFORCE_INVALID(adjusted >= 0 && adjusted < output_rank,
                        "ComputeShapeUnsqueeze: axis out of range.");
    normalized.push_back(adjusted);
  }
  std::sort(normalized.begin(), normalized.end());
  EXT_ENFORCE_INVALID(std::adjacent_find(normalized.begin(), normalized.end()) == normalized.end(),
                      "ComputeShapeUnsqueeze: duplicate axes are not allowed.");
  return normalized;
}

} // namespace

void ComputeShapeUnsqueeze(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Unsqueeze", "ComputeShapeUnsqueeze");
  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeUnsqueeze: Unsqueeze requires two inputs (data, axes).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &axes_tensor = ctx.Get(node.input(1));
  const int64_t input_rank = static_cast<int64_t>(data.Shape().Rank());

  SymShape out_shape;
  if (axes_tensor.HasValueAsShape()) {
    const SymShape &axes = axes_tensor.ValueAsShape();
    std::vector<int64_t> raw_axes;
    raw_axes.reserve(axes.Rank());
    for (size_t i = 0; i < axes.Rank(); ++i) {
      EXT_ENFORCE_INVALID(axes[i].IsInt(),
                          "ComputeShapeUnsqueeze: axes values must be concrete integers.");
      raw_axes.push_back(axes[i].AsInt());
    }

    const int64_t output_rank = input_rank + static_cast<int64_t>(raw_axes.size());
    const std::vector<int64_t> normalized_axes = NormalizeAxes(raw_axes, output_rank);

    size_t axis_index = 0;
    size_t input_index = 0;
    for (int64_t out_i = 0; out_i < output_rank; ++out_i) {
      if (axis_index < normalized_axes.size() && normalized_axes[axis_index] == out_i) {
        out_shape.PushBack(SymDim(1));
        ++axis_index;
      } else {
        out_shape.PushBack(data.Shape()[input_index]);
        ++input_index;
      }
    }
  } else if (axes_tensor.Shape().Rank() == 1 && axes_tensor.Shape()[0].IsInt()) {
    const int64_t output_rank = input_rank + axes_tensor.Shape()[0].AsInt();
    EXT_ENFORCE_INVALID(output_rank >= 0,
                        "ComputeShapeUnsqueeze: inferred output rank must be non-negative.");
    for (int64_t i = 0; i < output_rank; ++i) {
      out_shape.PushBack(SymDim("Unsqueeze_dim" + std::to_string(i)));
    }
  } else {
    out_shape.PushBack(SymDim("Unsqueeze_dim0"));
  }

  SymTensor out_tensor(nullptr, data.Dtype(), std::move(out_shape));

  // Unsqueeze only inserts size-1 dimensions into the shape; it does not
  // change the tensor values.  Forward the ValueAsShape annotation so that
  // downstream ops (e.g. Concat → Reshape) can still reconstruct the
  // symbolic target shape from these values.
  if (data.HasValueAsShape()) {
    out_tensor.SetValueAsShape(data.ValueAsShape());
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
