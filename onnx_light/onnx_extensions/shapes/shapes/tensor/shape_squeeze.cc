// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

std::vector<int64_t> NormalizeAxes(const std::vector<int64_t> &axes, int64_t rank) {
  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t axis : axes) {
    const int64_t adjusted = axis < 0 ? axis + rank : axis;
    EXT_ENFORCE_INVALID(adjusted >= 0 && adjusted < rank,
                        "ComputeShapeSqueeze: axis out of range.");
    normalized.push_back(adjusted);
  }
  std::sort(normalized.begin(), normalized.end());
  EXT_ENFORCE_INVALID(std::adjacent_find(normalized.begin(), normalized.end()) == normalized.end(),
                      "ComputeShapeSqueeze: duplicate axes are not allowed.");
  return normalized;
}

} // namespace

void ComputeShapeSqueeze(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Squeeze", "ComputeShapeSqueeze");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSqueeze: Squeeze requires at least one input.");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymShape &input_shape = data.Shape();
  const int64_t rank = static_cast<int64_t>(input_shape.Rank());

  bool axes_specified = false;
  bool axes_known = false;
  bool axes_count_known = false;
  int64_t axes_count = 0;
  std::vector<int64_t> normalized_axes;

  if (node.input_size() >= 2 && !node.input(1).empty()) {
    axes_specified = true;
    const SymTensor &axes_tensor = ctx.Get(node.input(1));
    if (axes_tensor.HasValueAsShape()) {
      const SymShape &axes = axes_tensor.ValueAsShape();
      std::vector<int64_t> raw_axes;
      raw_axes.reserve(axes.Rank());
      for (size_t i = 0; i < axes.Rank(); ++i) {
        EXT_ENFORCE_INVALID(axes[i].IsInt(),
                            "ComputeShapeSqueeze: axes values must be concrete integers.");
        raw_axes.push_back(axes[i].AsInt());
      }
      normalized_axes = NormalizeAxes(raw_axes, rank);
      axes_known = true;
      axes_count_known = true;
      axes_count = static_cast<int64_t>(normalized_axes.size());
    } else if (axes_tensor.Shape().Rank() == 1 && axes_tensor.Shape()[0].IsInt()) {
      axes_count_known = true;
      axes_count = axes_tensor.Shape()[0].AsInt();
    }
  }

  SymShape out_shape;
  if (axes_known) {
    size_t axis_index = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (axis_index < normalized_axes.size() && normalized_axes[axis_index] == i) {
        if (input_shape[static_cast<size_t>(i)].IsInt()) {
          EXT_ENFORCE_INVALID(input_shape[static_cast<size_t>(i)].AsInt() == 1,
                              "ComputeShapeSqueeze: selected axis dimension must be 1.");
        }
        ++axis_index;
        continue;
      }
      out_shape.PushBack(input_shape[static_cast<size_t>(i)]);
    }
  } else if (!axes_specified) {
    for (size_t i = 0; i < input_shape.Rank(); ++i) {
      if (input_shape[i].IsInt() && input_shape[i].AsInt() == 1) {
        continue;
      }
      out_shape.PushBack(input_shape[i]);
    }
  } else if (axes_count_known) {
    EXT_ENFORCE_INVALID(axes_count >= 0 && axes_count <= rank,
                        "ComputeShapeSqueeze: number of axes exceeds input rank.");
    for (int64_t i = 0; i < rank - axes_count; ++i) {
      out_shape.PushBack(SymDim("Squeeze_dim" + std::to_string(i)));
    }
  } else {
    out_shape.PushBack(SymDim("Squeeze_dim0"));
  }

  SymTensor out_tensor(nullptr, data.Dtype(), std::move(out_shape));

  // Squeeze only removes size-1 dimensions from the shape; it does not
  // change the tensor values.  Forward the ValueAsShape annotation so that
  // downstream ops can still use the propagated values.
  if (data.HasValueAsShape()) {
    out_tensor.SetValueAsShape(data.ValueAsShape());
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
