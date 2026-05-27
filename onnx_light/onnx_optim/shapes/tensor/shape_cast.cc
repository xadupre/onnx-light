// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeCast(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Cast", "ComputeShapeCast");

  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeCast: Cast requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  OptimShape out_shape = input.Shape();

  const AttributeProto *to_attr = FindAttribute(node, "to");
  if (to_attr == nullptr) {
    throw std::invalid_argument("ComputeShapeCast: required attribute 'to' is missing.");
  }
  const int64_t to_value = to_attr->i();
  const TensorType out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(to_value));
  if (out_dtype == TensorType::kUndefined) {
    throw std::invalid_argument("ComputeShapeCast: attribute 'to' has unsupported value " +
                                std::to_string(to_value) + ".");
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
