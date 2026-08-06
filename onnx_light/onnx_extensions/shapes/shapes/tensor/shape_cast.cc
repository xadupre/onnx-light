// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeCast(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Cast", "ComputeShapeCast");

  EXT_ENFORCE_INVALID(!(node.input_size() < 1), "ComputeShapeCast: Cast requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  SymShape out_shape = input.Shape();

  const AttributeProto *to_attr = FindAttribute(node, "to");
  EXT_ENFORCE_INVALID(to_attr != nullptr, "ComputeShapeCast: required attribute 'to' is missing.");
  const int64_t to_value = to_attr->i();
  const TensorType out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(to_value));
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined,
                      "ComputeShapeCast: attribute 'to' has unsupported value ", to_value, ".");

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
