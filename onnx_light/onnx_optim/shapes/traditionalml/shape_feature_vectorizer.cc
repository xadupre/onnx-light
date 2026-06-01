// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

// Returns the per-input feature widths declared by the ``inputdimensions``
// attribute when set, or, for each input, attempts to recover the width from
// the input's last dimension when its shape is fully known. Unknown widths
// are reported as -1 in the returned vector.
std::vector<int64_t> ResolveInputDims(const NodeProto &node, const ShapesContext &ctx,
                                      const std::vector<std::string> &inputs) {
  std::vector<int64_t> dims(inputs.size(), -1);
  const AttributeProto *attr = FindAttribute(node, "inputdimensions");
  if (attr != nullptr && attr->ints_size() > 0) {
    EXT_ENFORCE_INVALID(
        static_cast<size_t>(attr->ints_size()) == inputs.size(),
        "ComputeShapeFeatureVectorizer: 'inputdimensions' length must match the number of "
        "inputs when provided.");
    for (size_t i = 0; i < inputs.size(); ++i) {
      dims[i] = attr->ints(static_cast<int>(i));
    }
    return dims;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!ctx.Has(inputs[i])) {
      continue;
    }
    const OptimShape &shape = ctx.Get(inputs[i]).Shape();
    if (shape.Empty()) {
      continue;
    }
    const OptimDim &last = shape[shape.Rank() - 1];
    if (last.IsInt()) {
      dims[i] = last.AsInt();
    }
  }
  return dims;
}

// Returns the batch dimension shared by every input, falling back to a
// symbolic ``"N"`` dimension when any input's batch dimension is unknown.
OptimDim ResolveBatchDim(const ShapesContext &ctx, const std::vector<std::string> &inputs) {
  OptimDim resolved(static_cast<int64_t>(1));
  bool seen = false;
  for (const std::string &name : inputs) {
    if (!ctx.Has(name)) {
      return OptimDim(std::string("N"));
    }
    const OptimShape &shape = ctx.Get(name).Shape();
    if (shape.Empty()) {
      return OptimDim(std::string("N"));
    }
    OptimDim batch = shape.Rank() == 1 ? OptimDim(static_cast<int64_t>(1)) : shape[0];
    if (!seen) {
      resolved = batch;
      seen = true;
      continue;
    }
    // If both are integers, keep the maximum (matches the kernel's broadcast).
    if (resolved.IsInt() && batch.IsInt()) {
      if (batch.AsInt() > resolved.AsInt()) {
        resolved = batch;
      }
      continue;
    }
    // Mixed/symbolic: fall back to a symbolic dimension.
    return OptimDim(std::string("N"));
  }
  return resolved;
}

} // namespace

void ComputeShapeFeatureVectorizer(ShapesContext &ctx, const NodeProto &node,
                                   const std::vector<std::string> &inputs) {
  CheckNodeOpAndOutput(node, "FeatureVectorizer", "ComputeShapeFeatureVectorizer");
  EXT_ENFORCE_INVALID(!inputs.empty(),
                      "ComputeShapeFeatureVectorizer: at least one input is required.");

  const std::vector<int64_t> dims = ResolveInputDims(node, ctx, inputs);
  bool all_known = true;
  int64_t total = 0;
  for (int64_t d : dims) {
    if (d < 0) {
      all_known = false;
      break;
    }
    total += d;
  }

  OptimShape output_shape;
  output_shape.PushBack(ResolveBatchDim(ctx, inputs));
  if (all_known) {
    output_shape.PushBack(OptimDim(total));
  } else {
    output_shape.PushBack(OptimDim(std::string("F")));
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
