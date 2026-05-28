// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/reduction/shape_reduction.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace reduction {

namespace {

// Resolves a possibly-negative axis (ONNX semantics: ``axis`` in
// ``[-rank, rank - 1]``) to a non-negative axis. Throws on out-of-range.
int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "ComputeShapeArgReduce: axis " + std::to_string(axis) +
                          " is out of range for rank " + std::to_string(rank) + ".");
  return resolved;
}

} // namespace

void ComputeShapeArgReduce(ShapesContext &ctx, const NodeProto &node, const char *data) {
  // Accept either ArgMax or ArgMin: same shape semantics for both.
  const std::string op_type = node.op_type().as_string();
  EXT_ENFORCE_INVALID(op_type == "ArgMax" || op_type == "ArgMin",
                      "ComputeShapeArgReduce: expected ArgMax or ArgMin, got '" + op_type + "'.");
  EXT_ENFORCE_INVALID(node.output_size() >= 1,
                      "ComputeShapeArgReduce: node must declare at least one output.");

  const OptimTensor &input = ctx.Get(data);
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());
  EXT_ENFORCE_INVALID(rank > 0, "ComputeShapeArgReduce: input '" + std::string(data) +
                                    "' must have at least one dimension.");

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 0);
  const int64_t axis = ResolveAxis(axis_attr, rank);
  const bool keepdims = GetAttributeOr<int64_t>(node, "keepdims", 1) != 0;

  OptimShape out_shape;
  for (std::size_t d = 0; d < in_shape.Rank(); ++d) {
    if (static_cast<int64_t>(d) == axis) {
      if (keepdims) {
        out_shape.PushBack(OptimDim(static_cast<int64_t>(1)));
      }
    } else {
      out_shape.PushBack(in_shape[d]);
    }
  }

  // Output dtype is always int64, regardless of the input dtype.
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
}

} // namespace reduction
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
