// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

using core::shapes::PeakMemoryEstimate;
using core::symbolic::Device;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

namespace peak_memory_detail {

inline bool PeakMemoryShapesAvailable(const std::vector<SymShape> &inputs, size_t minimum,
                                      size_t maximum) {
  bool available = inputs.size() >= minimum && inputs.size() <= maximum;
  for (const auto &shape : inputs) {
    for (const auto &dim : shape.Dims()) {
      EXT_ENFORCE_INVALID(!dim.IsInt() || dim.AsInt() >= 0,
                          "Negative kernel peak-memory dimension.");
      available &= !dim.IsExpr() || (!dim.AsExpr().empty() && dim.AsExpr() != "?");
    }
  }
  return available;
}

inline SymDim PeakMemoryAdd(const SymDim &a, const SymDim &b) {
  if (a.IsInt() && b.IsInt())
    return SymDim(core::shapes::CheckedDimensionAdd(a.AsInt(), b.AsInt()));
  return core::symbolic::FromDimType(
      core::expressions::dim_add(core::symbolic::ToDimType(a), core::symbolic::ToDimType(b)));
}

inline SymDim PeakMemoryMultiply(const SymDim &a, const SymDim &b) {
  if (a.IsInt() && b.IsInt())
    return SymDim(core::shapes::CheckedDimensionMultiply(a.AsInt(), b.AsInt()));
  return core::symbolic::FromDimType(
      core::expressions::dim_mul(core::symbolic::ToDimType(a), core::symbolic::ToDimType(b)));
}

inline SymDim PeakMemoryElements(const SymShape &shape, size_t begin = 0) {
  for (size_t i = begin; i < shape.Rank(); ++i)
    if (shape[i].IsInt() && shape[i].AsInt() == 0)
      return SymDim(int64_t{0});
  SymDim count(int64_t{1});
  for (size_t i = begin; i < shape.Rank(); ++i)
    count = PeakMemoryMultiply(count, shape[i]);
  return count;
}

} // namespace peak_memory_detail
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
