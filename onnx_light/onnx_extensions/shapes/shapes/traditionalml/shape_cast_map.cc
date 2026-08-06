// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

namespace {

TensorType ResolveOutputDtype(const NodeProto &node) {
  const std::string cast_to = GetAttributeOr<std::string>(node, "cast_to", std::string("TO_FLOAT"));
  if (cast_to == "TO_FLOAT") {
    return TensorType::kFloat;
  }
  if (cast_to == "TO_INT64") {
    return TensorType::kInt64;
  }
  if (cast_to == "TO_STRING") {
    return TensorType::kString;
  }
  EXT_THROW_INVALID(
      "ComputeShapeCastMap: 'cast_to' must be one of 'TO_FLOAT', 'TO_INT64', 'TO_STRING'.");
}

} // namespace

void ComputeShapeCastMap(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "CastMap", "ComputeShapeCastMap");
  (void)x; // The input map's dtype/shape does not influence the output.

  const TensorType output_dtype = ResolveOutputDtype(node);

  const std::string map_form = GetAttributeOr<std::string>(node, "map_form", std::string("DENSE"));
  SymShape output_shape;
  if (map_form == "SPARSE") {
    const int64_t max_map = GetAttributeOr<int64_t>(node, "max_map", static_cast<int64_t>(1));
    output_shape.PushBack(SymDim(max_map));
  } else {
    // DENSE: output length equals the number of keys in the input map, which is
    // a runtime quantity. Represent it as a symbolic dimension.
    output_shape.PushBack(SymDim(std::string("CastMap_dense_n")));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, output_dtype, std::move(output_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
