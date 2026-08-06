// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/quantization/shape_quantization.h"

#include <cstdint>
#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization {

void ComputeShapeQuantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x,
                                const char *y_zero_point) {
  CheckNodeOpAndOutput(node, "QuantizeLinear", "ComputeShapeQuantizeLinear");

  const SymTensor &input = ctx.Get(x);
  SymShape out_shape = input.Shape();

  TensorType out_dtype = TensorType::kUint8;
  const bool has_zero_point = y_zero_point != nullptr && y_zero_point[0] != '\0';
  if (has_zero_point) {
    out_dtype = ctx.Get(y_zero_point).Dtype();
  } else {
    const int64_t output_dtype_attr = GetAttributeOr<int64_t>(node, "output_dtype", 0);
    if (output_dtype_attr != 0) {
      const TensorType from_attr =
          DataTypeToTensorType(static_cast<TensorProto::DataType>(output_dtype_attr));
      EXT_ENFORCE_INVALID(
          from_attr != TensorType::kUndefined,
          "ComputeShapeQuantizeLinear: attribute 'output_dtype' has unsupported value ",
          output_dtype_attr, ".");
      out_dtype = from_attr;
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization
