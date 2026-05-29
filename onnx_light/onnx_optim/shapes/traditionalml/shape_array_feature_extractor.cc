// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

bool TryComputeFlattenedLength(const OptimShape &shape, OptimDim &last_dim) {
  if (shape.Empty()) {
    return false;
  }
  int64_t product = 1;
  std::string symbolic_dim;
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    const OptimDim &d = shape[i];
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
    last_dim = OptimDim(product);
    return true;
  }
  if (product == 1) {
    last_dim = OptimDim(symbolic_dim);
    return true;
  }
  return false;
}

} // namespace

void ComputeShapeArrayFeatureExtractor(ShapesContext &ctx, const NodeProto &node, const char *x,
                                       const char *y) {
  CheckNodeOpAndOutput(node, "ArrayFeatureExtractor", "ComputeShapeArrayFeatureExtractor");
  const OptimTensor &input = ctx.Get(x);
  const OptimTensor &indices = ctx.Get(y);

  EXT_ENFORCE_INVALID(indices.Dtype() == TensorType::kInt64,
                      "ComputeShapeArrayFeatureExtractor: indices input must be int64.");

  OptimShape output_shape;
  const OptimShape &input_shape = input.Shape();
  for (std::size_t i = 0; i + 1 < input_shape.Rank(); ++i) {
    output_shape.PushBack(input_shape[i]);
  }

  OptimDim last_dim("ArrayFeatureExtractor_dim");
  const bool has_known_last_dim = TryComputeFlattenedLength(indices.Shape(), last_dim);
  if (has_known_last_dim) {
    output_shape.PushBack(std::move(last_dim));
  } else {
    output_shape.PushBack(OptimDim("ArrayFeatureExtractor_last_dim"));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
