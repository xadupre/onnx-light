// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_broadcast.h"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

namespace {

// Returns a textual representation of ``dim`` suitable for embedding in
// a synthesised symbolic expression.
std::string DimToString(const SymDim &dim) {
  if (dim.IsInt()) {
    return std::to_string(dim.AsInt());
  }
  return dim.AsExpr();
}

// Pairs the trailing dimensions of ``a`` and ``b`` (right-aligned) and
// computes the resulting dimension under numpy-style broadcasting.
SymDim BroadcastDim(const SymDim &a, const SymDim &b) {
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
    EXT_THROW_INVALID("BroadcastShapes: incompatible integer dimensions ", ai, " and ", bi, ".");
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
  return SymDim("broadcast(" + DimToString(a) + ", " + DimToString(b) + ")");
}

} // namespace

SymShape BroadcastShapes(const SymShape &a, const SymShape &b) {
  const std::size_t ra = a.Rank();
  const std::size_t rb = b.Rank();
  const std::size_t r = std::max(ra, rb);
  std::vector<SymDim> dims;
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
  return SymShape(dims);
}

void ComputeShapeBinaryBroadcast(ShapesContext &ctx, const NodeProto &node, const char *input_a,
                                 const char *input_b, const char *expected_op_type,
                                 TensorType output_dtype) {
  CheckNodeOpAndOutput(node, expected_op_type, "ComputeShapeBinaryBroadcast");
  const SymTensor &lhs = ctx.Get(input_a);
  const SymTensor &rhs = ctx.Get(input_b);
  SymShape out_shape = BroadcastShapes(lhs.Shape(), rhs.Shape());
  ctx.Set(node.output(0), SymTensor(nullptr, output_dtype, std::move(out_shape)));
}

namespace {

SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

} // namespace

void PropagateValueAsShapeArithmetic(ShapesContext &ctx, const NodeProto &node, const char *input_a,
                                     const char *input_b, BroadcastDimOp op) {
  if (node.output_size() < 1) {
    return;
  }
  const std::string out_name = node.output(0);
  if (!ctx.Has(out_name)) {
    return;
  }
  const SymTensor &lhs = ctx.Get(input_a);
  const SymTensor &rhs = ctx.Get(input_b);
  if (!lhs.HasValueAsShape() || !rhs.HasValueAsShape()) {
    return;
  }
  const SymShape &av = lhs.ValueAsShape();
  const SymShape &bv = rhs.ValueAsShape();
  const std::size_t ra = av.Rank();
  const std::size_t rb = bv.Rank();
  const std::size_t r = std::max(ra, rb);
  if (r > kMaxOptimRank) {
    return;
  }
  const SymDim kOne(static_cast<int64_t>(1));
  SymShape out_value_as_shape;
  for (std::size_t i = 0; i < r; ++i) {
    const bool has_a = i + ra >= r;
    const bool has_b = i + rb >= r;
    const SymDim &da = has_a ? av[i - (r - ra)] : kOne;
    const SymDim &db = has_b ? bv[i - (r - rb)] : kOne;
    expressions::DimType result;
    switch (op) {
    case BroadcastDimOp::kAdd:
      result = expressions::dim_add(ToDimType(da), ToDimType(db));
      break;
    case BroadcastDimOp::kSub:
      result = expressions::dim_sub(ToDimType(da), ToDimType(db));
      break;
    }
    out_value_as_shape.PushBack(FromDimType(result));
  }
  SymTensor updated = ctx.Get(out_name);
  updated.SetValueAsShape(std::move(out_value_as_shape));
  ctx.Set(out_name, std::move(updated));
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
