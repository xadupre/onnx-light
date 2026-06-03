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

void ComputeShapeUnique(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Unique", "ComputeShapeUnique");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeUnique: Unique requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const TensorType dtype = input.Dtype();
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  const std::string sym = "Unique_" + node.output(0).as_string() + "_n";

  // Output 0: Y has the dtype of the input.
  OptimShape y_shape;
  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  if (axis_attr == nullptr) {
    // No axis: flattened, Y is 1-D with a symbolic length.
    y_shape.PushBack(OptimDim(sym));
  } else {
    int64_t axis = axis_attr->ref_i();
    if (rank > 0) {
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("ComputeShapeUnique: axis=" + std::to_string(axis) +
                                    " out of range for input rank " + std::to_string(rank) + ".");
      }
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis) {
        y_shape.PushBack(OptimDim(sym));
      } else {
        y_shape.PushBack(in_shape[static_cast<std::size_t>(d)]);
      }
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(y_shape)));

  // Optional INT64 1-D outputs: indices (length n_unique), inverse_indices
  // (length count == axis_dim if axis is provided, or total element count
  // otherwise; left symbolic when not deducible), counts (length n_unique).
  // Skip outputs declared as the empty string (missing optional).
  const int n_out = node.output_size();
  for (int i = 1; i < n_out && i < 4; ++i) {
    const std::string out_name = node.output(i).as_string();
    if (out_name.empty()) {
      continue;
    }
    OptimShape one_d;
    if (i == 2) {
      // inverse_indices: length equals the number of items being grouped.
      if (axis_attr != nullptr) {
        // axis-mode: length == size of axis dimension when it is concrete.
        int64_t axis = axis_attr->ref_i();
        if (axis < 0) {
          axis += rank;
        }
        const OptimDim &axis_d = in_shape[static_cast<std::size_t>(axis)];
        one_d.PushBack(axis_d);
      } else {
        // flattened: leave symbolic (the input's total element count is not
        // always concretely known and OptimShape does not expose a product).
        one_d.PushBack(OptimDim("Unique_" + out_name + "_count"));
      }
    } else {
      // indices, counts: length == number of unique groups (symbolic).
      one_d.PushBack(OptimDim(sym));
    }
    ctx.Set(node.output(i), OptimTensor(nullptr, TensorType::kInt64, std::move(one_d)));
  }
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
