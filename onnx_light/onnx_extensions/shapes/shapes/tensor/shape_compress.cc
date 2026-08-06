// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::tensor {

void ComputeShapeCompress(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Compress", "ComputeShapeCompress");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeCompress: Compress requires two inputs (input, condition).");

  const SymTensor &input = ctx.Get(node.input(0));
  const TensorType dtype = input.Dtype();
  const SymShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  // The symbol used for the unknown output count (number of selected slices).
  const std::string sym = "Compress_" + node.output(0) + "_count";

  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  if (axis_attr == nullptr) {
    // No axis: input is flattened and individual elements are selected.
    // Output is 1-D with a symbolic (unknown) length. The count is
    // bounded above by the number of elements in the input.
    SymShape out_shape;
    out_shape.PushBack(SymDim(sym));
    if (rank > 0) {
      std::vector<expressions::DimType> dims;
      dims.reserve(in_shape.Rank());
      for (std::size_t d = 0; d < in_shape.Rank(); ++d) {
        dims.push_back(ToDimType(in_shape[d]));
      }
      ctx.AddLessEqualConstraint(sym, expressions::dim_to_string(expressions::dim_multi_mul(dims)));
    }
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  // Axis mode: output has same rank as input, but the axis dimension is
  // replaced by a symbolic dimension.
  int64_t axis = axis_attr->ref_i();
  if (rank > 0) {
    if (axis < 0) {
      axis += rank;
    }
    EXT_ENFORCE_INVALID(!(axis < 0 || axis >= rank), "ComputeShapeCompress: axis=", axis,
                        " out of range for input rank ", rank, ".");
  }

  SymShape out_shape;
  for (int64_t d = 0; d < rank; ++d) {
    if (d == axis) {
      out_shape.PushBack(SymDim(sym));
    } else {
      out_shape.PushBack(in_shape[static_cast<std::size_t>(d)]);
    }
  }
  // The selected count along ``axis`` is bounded above by the input
  // dimension at ``axis``: record ``count <= input.shape[axis]``.
  if (rank > 0) {
    ctx.AddLessEqualConstraint(
        sym, expressions::dim_to_string(ToDimType(in_shape[static_cast<std::size_t>(axis)])));
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
