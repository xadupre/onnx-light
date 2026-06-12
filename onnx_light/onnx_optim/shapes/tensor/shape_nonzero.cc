// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_optim/expressions.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

namespace {

expressions::DimType ToDimType(const OptimDim &d) {
  if (d.IsInt()) {
    return expressions::DimType{d.AsInt()};
  }
  return expressions::DimType{d.AsExpr()};
}

} // namespace

void ComputeShapeNonZero(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "NonZero", "ComputeShapeNonZero");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeNonZero: NonZero requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());

  // The output is always a 2-D INT64 tensor whose first dimension is the
  // input rank (concrete integer) and whose second dimension is the number
  // of non-zero elements (a runtime value, kept symbolic).
  const std::string nnz_sym = "NonZero_" + node.output(0).as_string() + "_nnz";
  OptimShape out_shape;
  out_shape.PushBack(OptimDim(static_cast<int64_t>(input.Shape().Rank())));
  out_shape.PushBack(OptimDim(nnz_sym));

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

  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
