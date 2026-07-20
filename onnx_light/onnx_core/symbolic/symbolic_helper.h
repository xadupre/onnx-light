// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_optim`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace symbolic {

/// Converts an SymDim into the expressions::DimType variant used by the
/// symbolic dimension expression helpers (``dim_add``, ``dim_mul``,
/// ``dim_max``, ...). Integer dims become ``int64_t`` and symbolic dims
/// become their underlying string expression.
inline expressions::DimType ToDimType(const SymDim &d) {
  if (d.IsInt()) {
    return expressions::DimType{d.AsInt()};
  }
  return expressions::DimType{d.AsExpr()};
}

} // namespace symbolic
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
