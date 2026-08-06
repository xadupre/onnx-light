// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

namespace {

int64_t BatchSizeFromInput(const SymTensor &input, const char *caller) {
  EXT_ENFORCE_INVALID(!(input.Shape().Empty()), caller, ": input rank must be 1 or 2 when known.");
  if (input.Shape().Rank() == 1) {
    return 1;
  }
  if (input.Shape().Rank() == 2) {
    const SymDim &batch = input.Shape()[0];
    if (batch.IsInt()) {
      return batch.AsInt();
    }
    return -1;
  }
  EXT_THROW_INVALID(caller, ": input rank must be 1 or 2 when known.");
}

int64_t ClassCount(const NodeProto &node, bool &using_strings) {
  const AttributeProto *labels_strings = FindAttribute(node, "classlabels_strings");
  const AttributeProto *labels_ints = FindAttribute(node, "classlabels_ints");
  using_strings = labels_strings != nullptr && labels_strings->strings_size() > 0;
  const bool using_ints = labels_ints != nullptr && labels_ints->ints_size() > 0;
  EXT_ENFORCE_INVALID(
      !(using_strings && using_ints),
      "ComputeShapeSVMClassifier: only one of 'classlabels_strings' or 'classlabels_ints' may be "
      "specified as non-empty.");
  if (using_strings) {
    return labels_strings->strings_size();
  }
  if (using_ints) {
    return labels_ints->ints_size();
  }
  return 0;
}

} // namespace

void ComputeShapeSVMClassifier(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "SVMClassifier", "ComputeShapeSVMClassifier");

  const SymTensor &input = ctx.Get(x);
  const int64_t batch_size = BatchSizeFromInput(input, "ComputeShapeSVMClassifier");

  bool using_strings = false;
  const int64_t class_count = ClassCount(node, using_strings);
  const TensorType label_type = using_strings ? TensorType::kString : TensorType::kInt64;

  SymShape y_shape;
  y_shape.PushBack(batch_size >= 0 ? SymDim(batch_size) : SymDim("N"));
  ctx.Set(node.output(0), SymTensor(nullptr, label_type, std::move(y_shape)));

  if (node.output_size() >= 2 && !node.output(1).empty()) {
    SymShape z_shape;
    z_shape.PushBack(batch_size >= 0 ? SymDim(batch_size) : SymDim("N"));
    if (class_count > 0) {
      // One score per class per sample (ONNX spec: "one per class per example").
      z_shape.PushBack(SymDim(class_count));
    } else {
      z_shape.PushBack(SymDim("S"));
    }
    ctx.Set(node.output(1), SymTensor(nullptr, TensorType::kFloat, std::move(z_shape)));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
