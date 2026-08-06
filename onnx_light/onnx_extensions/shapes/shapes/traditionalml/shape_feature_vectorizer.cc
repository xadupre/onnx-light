// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"

#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml {

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
    const SymShape &shape = ctx.Get(inputs[i]).Shape();
    if (shape.Empty()) {
      continue;
    }
    const SymDim &last = shape[shape.Rank() - 1];
    if (last.IsInt()) {
      dims[i] = last.AsInt();
    }
  }
  return dims;
}

// Returns the batch dimension shared by every input, falling back to a
// symbolic ``"N"`` dimension when any input's batch dimension is unknown.
SymDim ResolveBatchDim(const ShapesContext &ctx, const std::vector<std::string> &inputs) {
  SymDim resolved(static_cast<int64_t>(1));
  bool seen = false;
  for (const std::string &name : inputs) {
    if (!ctx.Has(name)) {
      return SymDim(std::string("N"));
    }
    const SymShape &shape = ctx.Get(name).Shape();
    if (shape.Empty()) {
      return SymDim(std::string("N"));
    }
    SymDim batch = shape.Rank() == 1 ? SymDim(static_cast<int64_t>(1)) : shape[0];
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
    return SymDim(std::string("N"));
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

  SymShape output_shape;
  output_shape.PushBack(ResolveBatchDim(ctx, inputs));
  if (all_known) {
    output_shape.PushBack(SymDim(total));
  } else {
    output_shape.PushBack(SymDim(std::string("F")));
  }
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kFloat, std::move(output_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::traditionalml
