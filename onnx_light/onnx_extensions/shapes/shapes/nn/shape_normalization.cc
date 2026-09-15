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
  const SymShape &x_shape = input.Shape();
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), input.Shape()));

  const int opset = ctx.HasOpsetVersion(kOnnxDomain) ? ctx.OpsetVersion(kOnnxDomain) : int{21};
  if (opset < 21) {
    return;
  }

  int64_t num_groups = 0;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "num_groups" && attr.type() == AttributeProto::INT) {
      num_groups = attr.i();
    }
  }
  EXT_ENFORCE_INVALID(num_groups > 0, "ComputeShapeGroupNormalization: attribute 'num_groups' "
                                      "must be greater than zero.");
  EXT_ENFORCE_INVALID(x_shape.Rank() >= 2u,
                      "ComputeShapeGroupNormalization: input 'X' must have rank >= 2.");

  const SymShape &scale_shape = ctx.Get(node.input(1)).Shape();
  const SymShape &bias_shape = ctx.Get(node.input(2)).Shape();
  EXT_ENFORCE_INVALID(scale_shape.Rank() == 1u,
                      "ComputeShapeGroupNormalization: input 'scale' must have rank 1.");
  EXT_ENFORCE_INVALID(bias_shape.Rank() == 1u,
                      "ComputeShapeGroupNormalization: input 'bias' must have rank 1.");

  bool has_channel_count = false;
  int64_t channel_count = 0;
  for (const SymDim *dim : {&x_shape[1], &scale_shape[0], &bias_shape[0]}) {
    if (!dim->IsInt()) {
      continue;
    }
    EXT_ENFORCE_INVALID(!has_channel_count || dim->AsInt() == channel_count,
                        "ComputeShapeGroupNormalization: channel dimensions must match.");
    channel_count = dim->AsInt();
    has_channel_count = true;
  }
  if (has_channel_count) {
    EXT_ENFORCE_INVALID(channel_count % num_groups == 0,
                        "ComputeShapeGroupNormalization: the number of channels must be divisible "
                        "by num_groups.");
  }
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
