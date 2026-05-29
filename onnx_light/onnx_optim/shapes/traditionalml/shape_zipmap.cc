// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <string>
#include <utility>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

bool HasNonEmptyStringLabels(const NodeProto &node) {
  const AttributeProto *attr = FindAttribute(node, "classlabels_strings");
  return attr != nullptr && attr->strings_size() > 0;
}

bool HasNonEmptyInt64Labels(const NodeProto &node) {
  const AttributeProto *attr = FindAttribute(node, "classlabels_int64s");
  return attr != nullptr && attr->ints_size() > 0;
}

} // namespace

void ComputeShapeZipMap(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "ZipMap", "ComputeShapeZipMap");

  const OptimTensor &input = ctx.Get(x);
  EXT_ENFORCE_INVALID(input.Dtype() == TensorType::kFloat,
                      "ComputeShapeZipMap: input must be a float tensor.");

  const bool has_string_labels = HasNonEmptyStringLabels(node);
  const bool has_int64_labels = HasNonEmptyInt64Labels(node);
  EXT_ENFORCE_INVALID(
      has_string_labels != has_int64_labels,
      "ComputeShapeZipMap: exactly one of 'classlabels_strings' or 'classlabels_int64s' "
      "must be specified and non-empty.");

  const TensorType output_type =
      has_string_labels ? TensorType::kSeqMapStringFloat : TensorType::kSeqMapInt64Float;

  OptimShape output_shape;
  if (!input.Shape().Empty()) {
    EXT_ENFORCE_INVALID(
        input.Shape().Rank() == 1 || input.Shape().Rank() == 2,
        "ComputeShapeZipMap: input shape must be rank 1 or rank 2 when the rank is known.");
  }
  if (input.Shape().Rank() == 1) {
    output_shape.PushBack(OptimDim(1));
  } else if (input.Shape().Rank() == 2) {
    output_shape.PushBack(input.Shape()[0]);
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, output_type, std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
