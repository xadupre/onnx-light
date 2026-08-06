// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

namespace {

SymDim BatchDimFromInput(const SymTensor &input, const char *caller) {
  EXT_ENFORCE_INVALID(!(input.Shape().Empty()), caller, ": input rank must be 1 or 2 when known.");
  if (input.Shape().Rank() == 1) {
    return SymDim(1);
  }
  if (input.Shape().Rank() == 2) {
    return input.Shape()[0];
  }
  EXT_THROW_INVALID(caller, ": input rank must be 1 or 2 when known.");
}

int64_t ResolveClassCount(const NodeProto &node, bool &using_strings) {
  const AttributeProto *labels_strings = FindAttribute(node, "classlabels_strings");
  const AttributeProto *labels_ints = FindAttribute(node, "classlabels_ints");
  using_strings = labels_strings != nullptr && labels_strings->strings_size() > 0;
  const bool using_ints = labels_ints != nullptr && labels_ints->ints_size() > 0;
  EXT_ENFORCE_INVALID(
      !(using_strings && using_ints),
      "ComputeShapeLinearClassifier: only one of 'classlabels_strings' or 'classlabels_ints' may "
      "be specified as non-empty.");

  std::vector<float> intercepts;
  const AttributeProto *intercept_attr = FindAttribute(node, "intercepts");
  if (intercept_attr != nullptr) {
    for (float v : intercept_attr->ref_floats()) {
      intercepts.push_back(v);
    }
  }
  int64_t class_count = static_cast<int64_t>(intercepts.size());

  const int64_t label_count =
      using_strings ? static_cast<int64_t>(labels_strings->strings_size())
                    : (labels_ints != nullptr ? static_cast<int64_t>(labels_ints->ints_size()) : 0);
  if (class_count == 1 && label_count == 2) {
    class_count = 2;
  }
  return class_count;
}

} // namespace

void ComputeShapeLinearClassifier(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LinearClassifier", "ComputeShapeLinearClassifier");

  const SymTensor &input = ctx.Get(x);
  SymDim batch_dim = BatchDimFromInput(input, "ComputeShapeLinearClassifier");

  bool using_strings = false;
  const int64_t class_count = ResolveClassCount(node, using_strings);
  const TensorType label_type = using_strings ? TensorType::kString : TensorType::kInt64;

  SymShape y_shape;
  y_shape.PushBack(batch_dim);
  ctx.Set(node.output(0), SymTensor(nullptr, label_type, std::move(y_shape)));

  if (node.output_size() >= 2 && !node.output(1).empty()) {
    SymShape z_shape;
    z_shape.PushBack(batch_dim);
    if (class_count > 0) {
      z_shape.PushBack(SymDim(class_count));
    } else {
      z_shape.PushBack(SymDim("E"));
    }
    ctx.Set(node.output(1), SymTensor(nullptr, TensorType::kFloat, std::move(z_shape)));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
