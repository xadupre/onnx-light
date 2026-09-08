// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_proto/onnx_helper.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

void ValidateBitCast(int32_t from, int32_t to) {
  const uint32_t from_bits = FixedBitWidth(static_cast<TensorProto::DataType>(from));
  const uint32_t to_bits = FixedBitWidth(static_cast<TensorProto::DataType>(to));
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
  Tensor y = (rt ? rt->MakeOutputTensor(0, to, x.shape, y_n_bytes)
                 : MakeOutputTensor(to, x.shape, y_n_bytes, nullptr));
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
