// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeCastLike(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "CastLike", "ComputeShapeCastLike");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeCastLike: CastLike requires two inputs.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymTensor &target_type = ctx.Get(node.input(1));

  const TensorType out_dtype = target_type.Dtype();
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined,
                      "ComputeShapeCastLike: target_type has an undefined element type.");

  SymShape out_shape = input.Shape();
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
