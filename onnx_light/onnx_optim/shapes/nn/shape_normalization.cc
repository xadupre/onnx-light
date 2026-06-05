// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeInstanceNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "InstanceNormalization", "ComputeShapeInstanceNormalization");
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeGroupNormalization(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "GroupNormalization", "ComputeShapeGroupNormalization");
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

void ComputeShapeMeanVarianceNormalization(ShapesContext &ctx, const NodeProto &node,
                                           const char *x) {
  CheckNodeOpAndOutput(node, "MeanVarianceNormalization", "ComputeShapeMeanVarianceNormalization");
  const OptimTensor &input = ctx.Get(x);
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), input.Shape()));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
