// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <cstdint>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

int64_t Int64CategoryCount(const NodeProto &node) {
  const AttributeProto *attr = FindAttribute(node, "cats_int64s");
  return attr == nullptr ? 0 : static_cast<int64_t>(attr->ints_size());
}

int64_t StringCategoryCount(const NodeProto &node) {
  const AttributeProto *attr = FindAttribute(node, "cats_strings");
  return attr == nullptr ? 0 : static_cast<int64_t>(attr->strings_size());
}

} // namespace

void ComputeShapeOneHotEncoder(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "OneHotEncoder", "ComputeShapeOneHotEncoder");

  const int64_t num_int64_cats = Int64CategoryCount(node);
  const int64_t num_string_cats = StringCategoryCount(node);
  EXT_ENFORCE_INVALID((num_int64_cats > 0) != (num_string_cats > 0),
                      "ComputeShapeOneHotEncoder: exactly one of 'cats_int64s' or 'cats_strings' "
                      "must be specified and non-empty.");
  const int64_t num_cats = num_int64_cats > 0 ? num_int64_cats : num_string_cats;

  // OneHotEncoder emits a float tensor whose shape is the input shape extended
  // by a trailing dimension equal to the category count.
  const OptimTensor &input = ctx.Get(x);
  OptimShape output_shape;
  for (std::size_t i = 0; i < input.Shape().Rank(); ++i) {
    output_shape.PushBack(input.Shape()[i]);
  }
  output_shape.PushBack(OptimDim(num_cats));

  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
