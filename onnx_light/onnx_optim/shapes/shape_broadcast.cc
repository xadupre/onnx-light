// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_broadcast.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Returns a textual representation of ``dim`` suitable for embedding in
// a synthesised symbolic expression.
std::string DimToString(const OptimDim &dim) {
  if (dim.IsInt()) {
    return std::to_string(dim.AsInt());
  }
  return dim.AsExpr();
}

// Pairs the trailing dimensions of ``a`` and ``b`` (right-aligned) and
// computes the resulting dimension under numpy-style broadcasting.
OptimDim BroadcastDim(const OptimDim &a, const OptimDim &b) {
  // Fast path: both integers.
  if (a.IsInt() && b.IsInt()) {
    const int64_t ai = a.AsInt();
    const int64_t bi = b.AsInt();
    if (ai == bi) {
      return a;
    }
    if (ai == 1) {
      return b;
    }
    if (bi == 1) {
      return a;
    }
    throw std::invalid_argument("BroadcastShapes: incompatible integer dimensions " +
                                std::to_string(ai) + " and " + std::to_string(bi) + ".");
  }
  // Either operand is the integer 1 → result is the other operand.
  if (a.IsInt() && a.AsInt() == 1) {
    return b;
  }
  if (b.IsInt() && b.AsInt() == 1) {
    return a;
  }
  // Equal symbolic (or equal integer) dimensions: result equals input.
  if (a == b) {
    return a;
  }
  // One side is a concrete integer (not 1) and the other is symbolic:
  // the only value compatible with broadcasting is that concrete
  // integer itself.
  if (a.IsInt()) {
    return a;
  }
  if (b.IsInt()) {
    return b;
  }
  // Two different symbolic dimensions: produce a synthesised symbolic
  // expression that records the broadcast.
  return OptimDim("broadcast(" + DimToString(a) + ", " + DimToString(b) + ")");
}

} // namespace

OptimShape BroadcastShapes(const OptimShape &a, const OptimShape &b) {
  const std::size_t ra = a.Rank();
  const std::size_t rb = b.Rank();
  const std::size_t r = std::max(ra, rb);
  std::vector<OptimDim> dims;
  dims.reserve(r);
  for (std::size_t i = 0; i < r; ++i) {
    // Right-align: missing leading dimensions are treated as 1.
    const bool has_a = i + ra >= r;
    const bool has_b = i + rb >= r;
    if (has_a && has_b) {
      dims.push_back(BroadcastDim(a[i - (r - ra)], b[i - (r - rb)]));
    } else if (has_a) {
      dims.push_back(a[i - (r - ra)]);
    } else {
      dims.push_back(b[i - (r - rb)]);
    }
  }
  return OptimShape(dims);
}

void ComputeShapeBinaryBroadcast(ShapesContext &ctx, const NodeProto &node, const char *input_a,
                                 const char *input_b, const char *expected_op_type,
                                 TensorType output_dtype) {
  if (node.op_type() != expected_op_type) {
    throw std::invalid_argument(std::string("ComputeShapeBinaryBroadcast expects op_type='") +
                                expected_op_type + "', got '" + node.op_type().as_string() + "'.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument(std::string("ComputeShapeBinaryBroadcast: node '") +
                                expected_op_type + "' has no output.");
  }
  const OptimTensor &lhs = ctx.Get(input_a);
  const OptimTensor &rhs = ctx.Get(input_b);
  OptimShape out_shape = BroadcastShapes(lhs.Shape(), rhs.Shape());
  ctx.Set(node.output(0).as_string(), OptimTensor(nullptr, output_dtype, std::move(out_shape)));
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
