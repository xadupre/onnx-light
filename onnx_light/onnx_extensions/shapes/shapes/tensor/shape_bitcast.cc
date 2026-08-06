// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

// Returns the per-element bit-width supported by ``BitCast`` (opset 26).
// Returns 0 for STRING and any unsupported value. Mirrors the table in
// ``onnx_lib/defs/tensor/defs.cc`` (BitCast shape inference) and
// ``onnx_extensions/kernels/kernels/tensor/kernel_bitcast.cc``.
constexpr int BitCastBitSize(TensorProto::DataType dtype) {
  switch (dtype) {
  case TensorProto::FLOAT:
  case TensorProto::INT32:
  case TensorProto::UINT32:
    return 32;
  case TensorProto::DOUBLE:
  case TensorProto::INT64:
  case TensorProto::UINT64:
  case TensorProto::COMPLEX64:
    return 64;
  case TensorProto::COMPLEX128:
    return 128;
  case TensorProto::FLOAT16:
  case TensorProto::BFLOAT16:
  case TensorProto::INT16:
  case TensorProto::UINT16:
    return 16;
  case TensorProto::INT8:
  case TensorProto::UINT8:
  case TensorProto::BOOL:
  case TensorProto::FLOAT8E4M3FN:
  case TensorProto::FLOAT8E4M3FNUZ:
  case TensorProto::FLOAT8E5M2:
  case TensorProto::FLOAT8E5M2FNUZ:
  case TensorProto::FLOAT8E8M0:
    return 8;
  case TensorProto::INT4:
  case TensorProto::UINT4:
  case TensorProto::FLOAT4E2M1:
    return 4;
  case TensorProto::INT2:
  case TensorProto::UINT2:
    return 2;
  default:
    return 0;
  }
}

} // namespace

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
  const int from_bits = BitCastBitSize(from_dtype);
  const int to_bits = BitCastBitSize(to_dtype);
  EXT_ENFORCE_INVALID(
      !(from_bits != 0 && to_bits != 0 && from_bits != to_bits),
      "ComputeShapeBitCast: BitCast requires input and output types to have the same "
      "bit-width, but input type has ",
      from_bits, " bits and output type has ", to_bits, " bits.");

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
