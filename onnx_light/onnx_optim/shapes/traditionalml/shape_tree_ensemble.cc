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

/// Returns the batch dimension from a [N,F] or [F] input.
/// For rank-1 input (single sample), the batch dimension is 1.
OptimDim BatchDimFromTreeInput(const OptimTensor &input, const char *caller) {
  if (input.Shape().Empty()) {
    throw std::invalid_argument(std::string(caller) + ": input rank must be 1 or 2 when known.");
  }
  if (input.Shape().Rank() == 1) {
    return OptimDim(static_cast<int64_t>(1));
  }
  if (input.Shape().Rank() == 2) {
    return input.Shape()[0];
  }
  throw std::invalid_argument(std::string(caller) + ": input rank must be 1 or 2 when known.");
}

} // namespace

void ComputeShapeTreeEnsembleRegressor(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "TreeEnsembleRegressor", "ComputeShapeTreeEnsembleRegressor");

  const OptimTensor &input = ctx.Get(x);
  OptimDim batch_dim = BatchDimFromTreeInput(input, "ComputeShapeTreeEnsembleRegressor");

  const int64_t n_targets = GetAttributeOr(node, "n_targets", static_cast<int64_t>(1));
  EXT_ENFORCE_INVALID(n_targets >= 1,
                      "ComputeShapeTreeEnsembleRegressor: 'n_targets' attribute must be >= 1.");

  OptimShape output_shape;
  output_shape.PushBack(batch_dim);
  output_shape.PushBack(OptimDim(n_targets));
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

void ComputeShapeTreeEnsembleClassifier(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "TreeEnsembleClassifier", "ComputeShapeTreeEnsembleClassifier");

  const OptimTensor &input = ctx.Get(x);
  OptimDim batch_dim = BatchDimFromTreeInput(input, "ComputeShapeTreeEnsembleClassifier");

  const AttributeProto *labels_strings = FindAttribute(node, "classlabels_strings");
  const AttributeProto *labels_ints = FindAttribute(node, "classlabels_int64s");
  const bool using_strings = labels_strings != nullptr && labels_strings->strings_size() > 0;
  const bool using_ints = labels_ints != nullptr && labels_ints->ints_size() > 0;
  EXT_ENFORCE_INVALID(!(using_strings && using_ints),
                      "ComputeShapeTreeEnsembleClassifier: only one of 'classlabels_strings' or "
                      "'classlabels_int64s' may be specified as non-empty.");

  const TensorType label_type = using_strings ? TensorType::kString : TensorType::kInt64;
  const int64_t class_count =
      using_strings ? static_cast<int64_t>(labels_strings->strings_size())
                    : (using_ints ? static_cast<int64_t>(labels_ints->ints_size()) : 0);

  OptimShape y_shape;
  y_shape.PushBack(batch_dim);
  ctx.Set(node.output(0), OptimTensor(nullptr, label_type, std::move(y_shape)));

  if (node.output_size() >= 2 && !node.output(1).empty()) {
    OptimShape z_shape;
    z_shape.PushBack(batch_dim);
    if (class_count > 0) {
      z_shape.PushBack(OptimDim(class_count));
    } else {
      z_shape.PushBack(OptimDim("E"));
    }
    ctx.Set(node.output(1), OptimTensor(nullptr, TensorType::kFloat, std::move(z_shape)));
  }
}

void ComputeShapeTreeEnsemble(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "TreeEnsemble", "ComputeShapeTreeEnsemble");

  const OptimTensor &input = ctx.Get(x);
  OptimDim batch_dim = BatchDimFromTreeInput(input, "ComputeShapeTreeEnsemble");

  const int64_t n_targets = GetAttributeOr(node, "n_targets", static_cast<int64_t>(1));
  EXT_ENFORCE_INVALID(n_targets >= 1,
                      "ComputeShapeTreeEnsemble: 'n_targets' attribute must be >= 1.");

  OptimShape output_shape;
  output_shape.PushBack(batch_dim);
  output_shape.PushBack(OptimDim(n_targets));
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
