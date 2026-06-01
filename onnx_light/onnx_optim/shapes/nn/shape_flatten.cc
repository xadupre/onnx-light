// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Multiplies a contiguous range of dimensions ``shape[begin..end)`` into a
// single :cpp:class:`OptimDim`. When every dim is a concrete int they are
// multiplied to a concrete int (the empty range produces ``1``). When any dim
// is symbolic the result is a symbolic expression
// ``"(<dim0>)*(<dim1>)*..."`` (a single symbolic dim with no concrete
// neighbours is returned verbatim).
OptimDim MultiplyDimRange(const OptimShape &shape, size_t begin, size_t end) {
  // Empty range -> 1.
  if (begin >= end) {
    return OptimDim(static_cast<int64_t>(1));
  }
  // Single dim -> return as-is.
  if (begin + 1 == end) {
    return shape[begin];
  }
  // Try concrete product first.
  bool all_int = true;
  int64_t product = 1;
  for (size_t i = begin; i < end; ++i) {
    if (!shape[i].IsInt()) {
      all_int = false;
      break;
    }
    product *= shape[i].AsInt();
  }
  if (all_int) {
    return OptimDim(product);
  }
  // Build a symbolic expression "(d0)*(d1)*...".
  std::string expr;
  for (size_t i = begin; i < end; ++i) {
    if (!expr.empty()) {
      expr += "*";
    }
    expr += "(";
    if (shape[i].IsInt()) {
      expr += std::to_string(shape[i].AsInt());
    } else {
      expr += shape[i].AsExpr();
    }
    expr += ")";
  }
  return OptimDim(std::move(expr));
}

} // namespace

void ComputeShapeFlatten(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "Flatten", "ComputeShapeFlatten");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  int64_t axis = GetAttributeOr<int64_t>(node, "axis", 1);
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis > rank) {
    throw std::invalid_argument("ComputeShapeFlatten: 'axis' out of range for input rank " +
                                std::to_string(rank) + ".");
  }

  OptimShape out_shape;
  out_shape.PushBack(MultiplyDimRange(in_shape, 0, static_cast<size_t>(axis)));
  out_shape.PushBack(
      MultiplyDimRange(in_shape, static_cast<size_t>(axis), static_cast<size_t>(rank)));

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
