// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <string>
#include <utility>
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

void ComputeShapeNonZero(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "NonZero", "ComputeShapeNonZero");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1), "ComputeShapeNonZero: NonZero requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));

  // The output is always a 2-D INT64 tensor whose first dimension is the
  // input rank (concrete integer) and whose second dimension is the number
  // of non-zero elements (a runtime value, kept symbolic).
  const std::string nnz_sym = "NonZero_" + node.output(0) + "_nnz";
  SymShape out_shape;
  out_shape.PushBack(SymDim(static_cast<int64_t>(input.Shape().Rank())));
  out_shape.PushBack(SymDim(nnz_sym));

  // The number of non-zero elements is bounded above by the total
  // number of elements in the input: record ``nnz <= prod(input.shape)``
  // so downstream passes can leverage this inequality.
  if (input.Shape().Rank() > 0) {
    std::vector<expressions::DimType> dims;
    dims.reserve(input.Shape().Rank());
    for (std::size_t d = 0; d < input.Shape().Rank(); ++d) {
      dims.push_back(ToDimType(input.Shape()[d]));
    }
    ctx.AddLessEqualConstraint(nnz_sym,
                               expressions::dim_to_string(expressions::dim_multi_mul(dims)));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
