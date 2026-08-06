// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::core::symbolic {

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

} // namespace ONNX_LIGHT_NAMESPACE::core::symbolic
