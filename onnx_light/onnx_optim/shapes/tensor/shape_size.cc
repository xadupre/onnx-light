// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeSize(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Size", "ComputeShapeSize");
  EXT_ENFORCE_INVALID(node.input_size() >= 1, "ComputeShapeSize: Size requires one input.");

  const OptimTensor &input = ctx.Get(node.input(0).as_string());

  // The output is always a 0-D (scalar) INT64 tensor.
  OptimTensor out_tensor(nullptr, TensorType::kInt64, OptimShape{});

  // When every input dimension is concrete, the scalar value produced by
  // ``Size`` (the product of the dimensions) is itself fully known and can
  // be recorded as a single-element ``ValueAsShape`` so that downstream
  // operators (``Reshape``, ``Expand``, ``ConstantOfShape``, ...) can
  // propagate the data even when only the input shape is available.
  bool all_concrete = true;
  int64_t product = 1;
  for (std::size_t i = 0; i < input.Shape().Rank(); ++i) {
    const OptimDim &d = input.Shape()[i];
    if (!d.IsInt()) {
      all_concrete = false;
      break;
    }
    product *= d.AsInt();
  }
  if (all_concrete) {
    OptimShape value_as_shape;
    value_as_shape.PushBack(OptimDim(product));
    out_tensor.SetValueAsShape(std::move(value_as_shape));
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
