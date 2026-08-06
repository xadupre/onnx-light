// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include <cstdint>
#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeTopK(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "TopK", "ComputeShapeTopK");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
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
  // a symbolic dimension because SymTensor does not always carry data.
  SymDim axis_dim;
  const AttributeProto *k_attr = FindAttribute(node, "k");
  if (k_attr != nullptr) {
    const int64_t k = k_attr->ref_i();
    EXT_ENFORCE_INVALID(k > 0, "ComputeShapeTopK: attribute k must be positive.");
    axis_dim = SymDim(k);
  } else {
    // Opset >= 10: ``k`` is a 1-D tensor input at index 1 (mandatory).
    // An empty string is a safe fallback key for malformed nodes only.
    const std::string k_input_name =
        node.input_size() >= 2 ? std::string(node.input(1)) : std::string();
    axis_dim = SymDim(ctx.TopKKDimName(k_input_name));
  }

  SymShape out_shape = in_shape;
  out_shape[static_cast<std::size_t>(axis)] = axis_dim;

  const TensorType out_dtype = input.Dtype();
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, out_shape));
  if (node.output_size() >= 2 && !node.output(1).empty()) {
    ctx.Set(node.output(1), SymTensor(nullptr, TensorType::kInt64, std::move(out_shape)));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
