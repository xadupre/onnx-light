// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include <cstdint>
#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

namespace {

bool TryComputeFlattenedLength(const SymShape &shape, SymDim &last_dim) {
  if (shape.Empty()) {
    return false;
  }
  int64_t product = 1;
  std::string symbolic_dim;
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    const SymDim &d = shape[i];
    if (d.IsInt()) {
      product *= d.AsInt();
      continue;
    }
    if (!symbolic_dim.empty()) {
      return false;
    }
    symbolic_dim = d.AsExpr();
  }
  if (symbolic_dim.empty()) {
    last_dim = SymDim(product);
    return true;
  }
  if (product == 1) {
    last_dim = SymDim(symbolic_dim);
    return true;
  }
  return false;
}

} // namespace

void ComputeShapeArrayFeatureExtractor(ShapesContext &ctx, const NodeProto &node, const char *x,
                                       const char *y) {
  CheckNodeOpAndOutput(node, "ArrayFeatureExtractor", "ComputeShapeArrayFeatureExtractor");
  const SymTensor &input = ctx.Get(x);
  const SymTensor &indices = ctx.Get(y);

  EXT_ENFORCE_INVALID(indices.Dtype() == TensorType::kInt64,
                      "ComputeShapeArrayFeatureExtractor: indices input must be int64.");

  SymShape output_shape;
  const SymShape &input_shape = input.Shape();
  for (std::size_t i = 0; i + 1 < input_shape.Rank(); ++i) {
    output_shape.PushBack(input_shape[i]);
  }

  SymDim last_dim("ArrayFeatureExtractor_dim");
  const bool has_known_last_dim = TryComputeFlattenedLength(indices.Shape(), last_dim);
  if (has_known_last_dim) {
    output_shape.PushBack(std::move(last_dim));
  } else {
    output_shape.PushBack(SymDim("ArrayFeatureExtractor_last_dim"));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(output_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
