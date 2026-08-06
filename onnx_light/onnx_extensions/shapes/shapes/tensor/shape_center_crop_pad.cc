// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeCenterCropPad(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "CenterCropPad", "ComputeShapeCenterCropPad");
  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeCenterCropPad: CenterCropPad requires two inputs.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  // Resolve the optional ``axes`` attribute (defaulting to ``[0..rank-1]``).
  std::vector<int64_t> axes;
  if (!GetAttributeInts(node, "axes", axes)) {
    axes.resize(static_cast<std::size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
      axes[static_cast<std::size_t>(i)] = i;
    }
  } else {
    for (auto &a : axes) {
      const int64_t na = a < 0 ? a + rank : a;
      EXT_ENFORCE_INVALID(!(na < 0 || na >= rank), "ComputeShapeCenterCropPad: axis ", a,
                          " is out of range for input rank ", rank, ".");
      a = na;
    }
  }

  // Try to resolve the ``shape`` input as a known 1-D value-as-shape.
  std::vector<int64_t> shape_values;
  bool shape_known = false;
  const SymTensor &shape_input = ctx.Get(node.input(1));
  if (shape_input.HasValueAsShape()) {
    const SymShape &shape_as = shape_input.ValueAsShape();
    bool all_int = true;
    for (std::size_t i = 0; i < shape_as.Rank(); ++i) {
      const SymDim &d = shape_as[i];
      if (!d.IsInt()) {
        all_int = false;
        break;
      }
      shape_values.push_back(d.AsInt());
    }
    if (all_int) {
      shape_known = true;
    }
  }
  EXT_ENFORCE_INVALID(!(shape_known && shape_values.size() != axes.size()),
                      "ComputeShapeCenterCropPad: number of elements of input 'shape' (",
                      shape_values.size(), ") does not match the number of axes (", axes.size(),
                      ").");

  // Build the output shape: start from the input, overwrite axes selected by
  // ``axes`` with values from ``shape`` (or a fresh symbolic dim when
  // unknown).
  SymShape out_shape;
  for (int64_t i = 0; i < rank; ++i) {
    out_shape.PushBack(in_shape[static_cast<std::size_t>(i)]);
  }
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t a = static_cast<std::size_t>(axes[i]);
    if (shape_known) {
      out_shape[a] = SymDim(shape_values[i]);
    } else {
      out_shape[a] = SymDim("CenterCropPad_dim" + std::to_string(a));
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
