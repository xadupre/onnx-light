// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeBitCast(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "BitCast", "ComputeShapeBitCast");

  EXT_ENFORCE_INVALID(!(node.input_size() < 1), "ComputeShapeBitCast: BitCast requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  SymShape out_shape = input.Shape();

  const AttributeProto *to_attr = FindAttribute(node, "to");
  EXT_ENFORCE_INVALID(to_attr != nullptr,
                      "ComputeShapeBitCast: required attribute 'to' is missing.");
  const int64_t to_value = to_attr->i();
  const TensorProto::DataType to_dtype = static_cast<TensorProto::DataType>(to_value);
  const TensorType out_dtype = DataTypeToTensorType(to_dtype);
  EXT_ENFORCE_INVALID(!(out_dtype == TensorType::kUndefined || out_dtype == TensorType::kString),
                      "ComputeShapeBitCast: attribute 'to' has unsupported value ", to_value,
                      " (BitCast does not support STRING or undefined types).");

  // The upstream BitCast schema enforces matching bit-widths between the
  // input and the target type.
  const TensorProto::DataType from_dtype = TensorTypeToDataType(input.Dtype());
  const uint32_t from_bits = FixedBitWidth(from_dtype);
  const uint32_t to_bits = FixedBitWidth(to_dtype);
  EXT_ENFORCE_INVALID(
      !(from_bits != 0 && to_bits != 0 && from_bits != to_bits),
      "ComputeShapeBitCast: BitCast requires input and output types to have the same "
      "bit-width, but input type has ",
      from_bits, " bits and output type has ", to_bits, " bits.");

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
