// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeInstanceNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "InstanceNormalization", "ComputeShapeInstanceNormalization");
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeGroupNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "GroupNormalization", "ComputeShapeGroupNormalization");
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeMeanVarianceNormalization(ShapesContext &ctx, const NodeProto &node,
                                           const char *x) {
  CheckNodeOpAndOutput(node, "MeanVarianceNormalization", "ComputeShapeMeanVarianceNormalization");
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeRMSNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "RMSNormalization", "ComputeShapeRMSNormalization");
  const SymTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeLayerNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LayerNormalization", "ComputeShapeLayerNormalization");
  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();

  // Y has the same dtype and shape as X.
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), in_shape));

  const int n_outputs = node.output_size();
  if (n_outputs <= 1) {
    return;
  }

  // Optional Mean / InvStdDev outputs have shape ``[d[0], ..., d[axis-1],
  // 1, ..., 1]`` and dtype ``stash_type`` (default FLOAT).
  TensorType stash_type = TensorType::kFloat;
  int64_t axis = -1;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "stash_type") {
      stash_type = static_cast<TensorType>(attr.i());
    } else if (attr.name() == "axis") {
      axis = attr.i();
    }
  }

  const int64_t rank = static_cast<int64_t>(in_shape.Rank());
  int64_t resolved_axis = axis;
  if (resolved_axis < 0) {
    resolved_axis += rank;
  }

  SymShape reduced_shape = in_shape;
  if (resolved_axis >= 0 && resolved_axis <= rank) {
    for (int64_t i = resolved_axis; i < rank; ++i) {
      reduced_shape[static_cast<size_t>(i)] = SymDim(static_cast<int64_t>(1));
    }
  }

  for (int i = 1; i < n_outputs; ++i) {
    const std::string &name = node.output(i);
    if (name.empty()) {
      continue;
    }
    ctx.Set(name, SymTensor(nullptr, stash_type, reduced_shape));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
