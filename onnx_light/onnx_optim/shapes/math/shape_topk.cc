// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

#include <cstdint>
#include <string>
#include <utility>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

void ComputeShapeTopK(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "TopK", "ComputeShapeTopK");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());
  EXT_ENFORCE_INVALID(rank >= 1, "ComputeShapeTopK: input must have rank >= 1.");

  int64_t axis = GetAttributeOr<int64_t>(node, "axis", int64_t{-1});
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank,
                      "ComputeShapeTopK: axis is out of range for the input rank.");

  // ``k`` was an integer attribute in opset 1 and became a 1-D tensor input
  // in opset 10. When the attribute is set we know ``k``; otherwise we emit
  // a symbolic dimension because OptimTensor does not always carry data.
  OptimDim axis_dim;
  const AttributeProto *k_attr = FindAttribute(node, "k");
  if (k_attr != nullptr) {
    const int64_t k = k_attr->ref_i();
    EXT_ENFORCE_INVALID(k > 0, "ComputeShapeTopK: attribute k must be positive.");
    axis_dim = OptimDim(k);
  } else {
    axis_dim = OptimDim("TopK_" + node.output(0).as_string() + "_k");
  }

  OptimShape out_shape = in_shape;
  out_shape[static_cast<std::size_t>(axis)] = axis_dim;

  const TensorType out_dtype = input.Dtype();
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, out_shape));
  if (node.output_size() >= 2 && !node.output(1).as_string().empty()) {
    ctx.Set(node.output(1), OptimTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
  }
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
