// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeCastLike(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "CastLike", "ComputeShapeCastLike");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeCastLike: CastLike requires two inputs.");

  const OptimTensor &input = ctx.Get(node.input(0));
  const OptimTensor &target_type = ctx.Get(node.input(1));

  const TensorType out_dtype = target_type.Dtype();
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined,
                      "ComputeShapeCastLike: target_type has an undefined element type.");

  OptimShape out_shape = input.Shape();
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
