// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns the per-element bit-width supported by ``BitCast`` (opset 26).
// Returns 0 for unsupported dtypes (including STRING and UNDEFINED).
// Mirrors the table in ``onnx_lib/defs/tensor/defs.cc``.
int BitCastBitSize(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
  case DataType::INT32:
  case DataType::UINT32:
    return 32;
  case DataType::DOUBLE:
  case DataType::INT64:
  case DataType::UINT64:
  case DataType::COMPLEX64:
    return 64;
  case DataType::COMPLEX128:
    return 128;
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
  case DataType::INT16:
  case DataType::UINT16:
    return 16;
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::BOOL:
  case DataType::FLOAT8E4M3FN:
  case DataType::FLOAT8E4M3FNUZ:
  case DataType::FLOAT8E5M2:
  case DataType::FLOAT8E5M2FNUZ:
  case DataType::FLOAT8E8M0:
    return 8;
  case DataType::INT4:
  case DataType::UINT4:
  case DataType::FLOAT4E2M1:
    return 4;
  case DataType::INT2:
  case DataType::UINT2:
    return 2;
  default:
    return 0;
  }
}

void ValidateBitCast(int32_t from, int32_t to) {
  const int from_bits = BitCastBitSize(from);
  const int to_bits = BitCastBitSize(to);
  EXT_ENFORCE_INVALID(from_bits != 0 && to_bits != 0,
                      "kernel::BitCast: unsupported data type (from=", from, ", to=", to,
                      "); string or undefined types are not allowed.");
  EXT_ENFORCE_INVALID(from_bits == to_bits,
                      "kernel::BitCast: input and output types must have the same bit-width, but "
                      "input has ",
                      from_bits, " bits and output has ", to_bits, " bits.");
}

} // namespace

Tensor BitCast::operator()(const Tensor &x, int32_t to, RuntimeContext *rt) const {
  ValidateBitCast(x.data_type, to);
  // BitCast preserves the exact bit pattern and the shape: same packed
  // byte size for both input and output. Copy the underlying bytes and
  // relabel the dtype.
  const size_t y_n_bytes = x.size_bytes();
  Tensor y = MakeOutputTensor(to, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  std::memcpy(y.mutable_bytes(), x.bytes(), y_n_bytes);
  return y;
}

void BitCast::operator()(const Tensor &x, int32_t to, Tensor &output) const {
  ValidateBitCast(x.data_type, to);
  EXT_ENFORCE_INVALID(output.data_type == to, "kernel::BitCast: preallocated output dtype ",
                      output.data_type, " must match ``to`` (", to, ").");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::BitCast: preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(),
                      "kernel::BitCast: preallocated output buffer has unexpected size in bytes.");
  // Byte-wise copy keeps the bit pattern intact on little-endian hosts
  // (the only ABI exercised by the backend test library).
  std::memcpy(output.mutable_bytes(), x.bytes(), x.size_bytes());
}

void BitCast::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int32_t to = static_cast<int32_t>(GetAttributeIntOrDefault(node, "to", -1));
  EXT_ENFORCE_INVALID(!(to < 0), "RunNode: ", node.op_type(), " requires INT attribute 'to'.");
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, to, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
