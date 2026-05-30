// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/quantization/shape_quantization.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace quantization {

void ComputeShapeDequantizeLinear(ShapesContext &ctx, const NodeProto &node, const char *x,
                                  const char *x_scale) {
  CheckNodeOpAndOutput(node, "DequantizeLinear", "ComputeShapeDequantizeLinear");

  const OptimTensor &input = ctx.Get(x);
  OptimShape out_shape = input.Shape();

  TensorType out_dtype = TensorType::kUndefined;
  const int64_t output_dtype_attr = GetAttributeOr<int64_t>(node, "output_dtype", 0);
  if (output_dtype_attr != 0) {
    const TensorType from_attr =
        DataTypeToTensorType(static_cast<TensorProto::DataType>(output_dtype_attr));
    if (from_attr == TensorType::kUndefined) {
      throw std::invalid_argument(
          "ComputeShapeDequantizeLinear: attribute 'output_dtype' has unsupported value " +
          std::to_string(output_dtype_attr) + ".");
    }
    out_dtype = from_attr;
  } else {
    out_dtype = ctx.Get(x_scale).Dtype();
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
