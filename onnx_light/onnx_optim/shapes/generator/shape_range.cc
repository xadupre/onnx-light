// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

namespace {

// Returns the single integer scalar carried by a ``ValueAsShape``
// annotation, if any. ``Range``'s inputs are 0-D scalar tensors; in the
// data-propagated shape representation a known scalar constant is encoded
// as a rank-1 shape with a single integer dim holding the value. Rank-0
// shapes carry no extractable integer value here and are rejected.
bool TryReadKnownIntScalar(const OptimTensor &input, int64_t *out) {
  if (!input.HasValueAsShape()) {
    return false;
  }
  const OptimShape &v = input.ValueAsShape();
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

} // namespace

void ComputeShapeRange(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Range", "ComputeShapeRange");
  EXT_ENFORCE_INVALID(node.input_size() >= 3,
                      "ComputeShapeRange: Range requires three inputs (start, limit, delta).");

  const OptimTensor &start = ctx.Get(node.input(0).as_string());
  const OptimTensor &limit = ctx.Get(node.input(1).as_string());
  const OptimTensor &delta = ctx.Get(node.input(2).as_string());

  const TensorType out_dtype = start.Dtype();

  OptimShape out_shape;
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
    out_shape.PushBack(OptimDim(n));
  } else {
    // Unknown output length: produce a single symbolic dim.
    out_shape.PushBack(OptimDim("Range_dim0"));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
