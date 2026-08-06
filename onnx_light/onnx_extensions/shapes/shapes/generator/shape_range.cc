// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/generator/shape_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <variant>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::generator {

namespace {

// Returns the single integer scalar carried by a ``ValueAsShape``
// annotation, if any. ``Range``'s inputs are 0-D scalar tensors; in the
// data-propagated shape representation a known scalar constant is encoded
// as a rank-1 shape with a single integer dim holding the value. Rank-0
// shapes carry no extractable integer value here and are rejected.
bool TryReadKnownIntScalar(const SymTensor &input, int64_t *out) {
  if (!input.HasValueAsShape()) {
    return false;
  }
  const SymShape &v = input.ValueAsShape();
  if (v.Rank() == 0) {
    // A scalar value has no shape dim, so its integer value is unknown
    // from ``ValueAsShape`` alone.
    return false;
  }
  if (v.Rank() == 1 && v[0].IsInt()) {
    *out = v[0].AsInt();
    return true;
  }
  return false;
}

// Returns the single dim (integer or symbolic) carried by a ``ValueAsShape``
// annotation of rank 1, if any. Unlike ``TryReadKnownIntScalar``, succeeds
// when the dim is a symbolic expression string, allowing extraction of the
// symbolic bound of a Range input even when the concrete integer value is
// not statically known.
bool TryReadOptimDimScalar(const SymTensor &input, SymDim *out) {
  if (!input.HasValueAsShape()) {
    return false;
  }
  const SymShape &v = input.ValueAsShape();
  if (v.Rank() == 1) {
    *out = v[0];
    return true;
  }
  return false;
}

// Converts an ``SymDim`` to the ``expressions::DimType`` variant used by
// the symbolic dimension arithmetic helpers (``dim_add``, ``dim_sub``, …).
expressions::DimType DimToDimType(const SymDim &d) {
  if (d.IsInt()) {
    return expressions::DimType{d.AsInt()};
  }
  return expressions::DimType{d.AsExpr()};
}

// Converts an ``expressions::DimType`` back to an ``SymDim``.
SymDim DimTypeToOptimDim(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

} // namespace

void ComputeShapeRange(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Range", "ComputeShapeRange");
  EXT_ENFORCE_INVALID(node.input_size() >= 3,
                      "ComputeShapeRange: Range requires three inputs (start, limit, delta).");

  const SymTensor &start = ctx.Get(node.input(0));
  const SymTensor &limit = ctx.Get(node.input(1));
  const SymTensor &delta = ctx.Get(node.input(2));

  const TensorType out_dtype = start.Dtype();

  SymShape out_shape;
  int64_t s = 0;
  int64_t l = 0;
  int64_t d = 0;
  const bool all_known = IsIntegerTensorType(out_dtype) && TryReadKnownIntScalar(start, &s) &&
                         TryReadKnownIntScalar(limit, &l) && TryReadKnownIntScalar(delta, &d) &&
                         d != 0;
  if (all_known) {
    int64_t n = static_cast<int64_t>(
        std::ceil((static_cast<double>(l) - static_cast<double>(s)) / static_cast<double>(d)));
    n = std::max<int64_t>(n, 0);
    out_shape.PushBack(SymDim(n));
  } else {
    // Try to build a symbolic expression from the inputs when at least one
    // dim is unknown. Reading the dim (integer or symbolic) from each
    // input's ``ValueAsShape`` annotation lets us compose a meaningful
    // expression like ``limit_sym - start_sym`` instead of introducing a
    // fresh ``"Range_dim0"`` token that is disconnected from all existing
    // symbolic dimensions.
    SymDim start_dim, limit_dim, delta_dim;
    const bool has_dims =
        IsIntegerTensorType(out_dtype) && TryReadOptimDimScalar(start, &start_dim) &&
        TryReadOptimDimScalar(limit, &limit_dim) && TryReadOptimDimScalar(delta, &delta_dim) &&
        !(delta_dim.IsInt() && delta_dim.AsInt() == 0);
    if (has_dims) {
      // Compute (limit - start) / delta symbolically. For the very common
      // case of delta == 1 the division is omitted entirely; for other
      // integer deltas we use floor division as an approximation (Range
      // semantics require ceiling division, but for integer deltas the
      // error is at most 1 and shape-tracking only needs a consistent
      // symbolic expression). When delta is itself symbolic we still
      // produce a compound expression.
      expressions::DimType l_expr =
          expressions::dim_sub(DimToDimType(limit_dim), DimToDimType(start_dim));
      if (!(delta_dim.IsInt() && delta_dim.AsInt() == 1)) {
        l_expr = expressions::dim_div(l_expr, DimToDimType(delta_dim));
      }
      out_shape.PushBack(DimTypeToOptimDim(l_expr));
    } else {
      // Unknown output length: produce a single symbolic dim.
      out_shape.PushBack(SymDim("Range_dim0"));
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace shapes::generator
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
