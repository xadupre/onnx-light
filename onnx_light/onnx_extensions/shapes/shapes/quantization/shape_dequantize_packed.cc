// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/dequantize_packed.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/shapes/shapes/quantization/shape_quantization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization {

void ComputeShapeDequantizePacked(ShapesContext &ctx, const NodeProto &node, const char *x) {
  using namespace core::runtime;
  CheckNodeOpAndOutput(node, "DequantizePacked", "ComputeShapeDequantizePacked");
  const auto *shape = FindAttribute(node, "shape");
  EXT_ENFORCE_INVALID(shape != nullptr && shape->type() == AttributeProto::INTS,
                      "DequantizePacked: shape must be an INTS attribute.");
  const Shape dims = GetAttributeShapeOrDefault(node, "shape", {});
  const int64_t dtype = GetAttributeIntOrDefault(node, "output_dtype", TensorProto::FLOAT);
  const auto &profile = PackedQuantizationLayout(GetRequiredAttributeString(node, "format"));
  const size_t bytes = ValidatePackedQuantizationShape(profile, dims, dtype);
  const SymTensor &input = ctx.Get(x);
  EXT_ENFORCE_INVALID(input.Dtype() == TensorType::kUint8 ||
                          input.Dtype() == TensorType::kUndefined,
                      "DequantizePacked: x must have UINT8 dtype.");
  EXT_ENFORCE_INVALID(input.Shape().Rank() == 1, "DequantizePacked: x must have rank one.");
  if (input.Shape()[0].IsInt())
    EXT_ENFORCE_INVALID(input.Shape()[0].AsInt() >= 0 &&
                            static_cast<uint64_t>(input.Shape()[0].AsInt()) == bytes,
                        "DequantizePacked: packed input length must match exactly.");
  SymShape output;
  for (const int64_t dim : dims)
    output.PushBack(SymDim(dim));
  ctx.Set(node.output(0),
          SymTensor(nullptr, DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype)),
                    std::move(output)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization
