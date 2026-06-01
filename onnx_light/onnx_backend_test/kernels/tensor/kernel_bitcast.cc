// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

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
  if (from_bits == 0 || to_bits == 0) {
    throw std::invalid_argument("kernel::BitCast: unsupported data_type "
                                "(string or undefined types are not allowed).");
  }
  if (from_bits != to_bits) {
    throw std::invalid_argument(
        "kernel::BitCast: input and output types must have the same bit-width, but "
        "input has " +
        std::to_string(from_bits) + " bits and output has " + std::to_string(to_bits) + " bits.");
  }
}

} // namespace

Tensor BitCast::operator()(const Tensor &x, int32_t to) const {
  ValidateBitCast(x.data_type, to);
  // BitCast preserves the exact bit pattern and the shape: same packed
  // byte size for both input and output. Copy the underlying bytes and
  // relabel the dtype.
  Tensor y("", to, x.shape, x.data);
  return y;
}

void BitCast::operator()(const Tensor &x, int32_t to, Tensor &output) const {
  ValidateBitCast(x.data_type, to);
  if (output.data_type != to) {
    throw std::invalid_argument("kernel::BitCast: preallocated output dtype must match ``to``.");
  }
  if (output.shape != x.shape) {
    throw std::invalid_argument(
        "kernel::BitCast: preallocated output shape must match input shape.");
  }
  if (output.data.size() != x.data.size()) {
    throw std::invalid_argument(
        "kernel::BitCast: preallocated output buffer has unexpected size in bytes.");
  }
  // Byte-wise copy keeps the bit pattern intact on little-endian hosts
  // (the only ABI exercised by the backend test library).
  std::copy(x.data.begin(), x.data.end(), output.data.begin());
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
