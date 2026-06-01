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

void ComputeShapeCompress(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Compress", "ComputeShapeCompress");

  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeCompress: Compress requires two inputs (input, condition).");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const TensorType dtype = input.Dtype();
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  // The symbol used for the unknown output count (number of selected slices).
  const std::string sym = "Compress_" + node.output(0).as_string() + "_count";

  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  if (axis_attr == nullptr) {
    // No axis: input is flattened and individual elements are selected.
    // Output is 1-D with a symbolic (unknown) length.
    OptimShape out_shape;
    out_shape.PushBack(OptimDim(sym));
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Axis mode: output has same rank as input, but the axis dimension is
  // replaced by a symbolic dimension.
  int64_t axis = axis_attr->ref_i();
  if (rank > 0) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument("ComputeShapeCompress: axis=" + std::to_string(axis) +
                                  " out of range for input rank " + std::to_string(rank) + ".");
    }
  }

  OptimShape out_shape;
  for (int64_t d = 0; d < rank; ++d) {
    if (d == axis) {
      out_shape.PushBack(OptimDim(sym));
    } else {
      out_shape.PushBack(in_shape[static_cast<std::size_t>(d)]);
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
