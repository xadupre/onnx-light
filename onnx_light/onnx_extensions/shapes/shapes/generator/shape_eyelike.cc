// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/generator/shape_generator.h"

#include <cstdint>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator {

void ComputeShapeEyeLike(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "EyeLike", "ComputeShapeEyeLike");
  EXT_ENFORCE_INVALID(node.input_size() >= 1, "ComputeShapeEyeLike: EyeLike requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  EXT_ENFORCE_INVALID(input.Shape().Rank() == 2,
                      "ComputeShapeEyeLike: input tensor must be 2-dimensional.");

  TensorType out_dtype = input.Dtype();
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  if (dtype_attr != nullptr) {
    const int64_t dtype_value = dtype_attr->i();
    out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
    EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined,
                        "ComputeShapeEyeLike: attribute 'dtype' has unsupported value ",
                        dtype_value, ".");
  }

  SymShape out_shape = input.Shape();
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator
