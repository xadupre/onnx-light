// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

int64_t BatchSizeFromInput(const OptimTensor &input, const char *caller) {
  if (input.Shape().Empty()) {
    throw std::invalid_argument(std::string(caller) + ": input rank must be 1 or 2 when known.");
  }
  if (input.Shape().Rank() == 1) {
    return 1;
  }
  if (input.Shape().Rank() == 2) {
    const OptimDim &batch = input.Shape()[0];
    if (batch.IsInt()) {
      return batch.AsInt();
    }
    return -1;
  }
  throw std::invalid_argument(std::string(caller) + ": input rank must be 1 or 2 when known.");
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

  const OptimTensor &input = ctx.Get(x);
  const int64_t batch_size = BatchSizeFromInput(input, "ComputeShapeSVMClassifier");

  bool using_strings = false;
  const int64_t class_count = ClassCount(node, using_strings);
  const TensorType label_type = using_strings ? TensorType::kString : TensorType::kInt64;

  OptimShape y_shape;
  y_shape.PushBack(batch_size >= 0 ? OptimDim(batch_size) : OptimDim("N"));
  ctx.Set(node.output(0), OptimTensor(nullptr, label_type, std::move(y_shape)));

  if (node.output_size() >= 2 && !node.output(1).empty()) {
    OptimShape z_shape;
    z_shape.PushBack(batch_size >= 0 ? OptimDim(batch_size) : OptimDim("N"));
    if (class_count > 0) {
      // Binary SVM exposes one decision score per sample; multi-class emits one
      // score per class.
      const int64_t score_count = class_count <= 2 ? 1 : class_count;
      z_shape.PushBack(OptimDim(score_count));
    } else {
      z_shape.PushBack(OptimDim("S"));
    }
    ctx.Set(node.output(1), OptimTensor(nullptr, TensorType::kFloat, std::move(z_shape)));
  }
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
