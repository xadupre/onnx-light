// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {
namespace detail {

// ---------------------------------------------------------------------------
// Shared validation + iteration helpers used by element-wise backend test
// kernels (Add, And, Or, Xor, ...). They centralize the dtype/shape/buffer
// checks performed both by the allocating ``Tensor operator()(...) const``
// overloads and by the in-place ``void operator()(..., Tensor *) const``
// overloads, plus the broadcasted iteration loop itself.
//
// Error messages match the long-standing per-kernel wording so existing
// tests keep passing. Two short labels are required to build them:
//   * ``op_name``        e.g. "kernel::Add"
//   * ``dtype_name``     e.g. "FLOAT", "BOOL"
// ---------------------------------------------------------------------------

/// Information about a validated binary broadcast: the output shape, total
/// element count, and the individual input element counts (used by the loop
/// to apply scalar broadcasting).
struct BroadcastInfo {
  std::vector<int64_t> shape;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

/// Verifies both inputs have ``expected_dtype`` and that their shapes are
/// equal or broadcastable via scalar broadcasting (one side has a single
/// element). Throws ``std::invalid_argument`` otherwise.
inline BroadcastInfo CheckBinaryBroadcast(const char *op_name, const char *dtype_name,
                                          int32_t expected_dtype, const Tensor &x,
                                          const Tensor &y) {
  if (x.data_type != expected_dtype || y.data_type != expected_dtype) {
    throw std::invalid_argument(std::string(op_name) + " only supports " + dtype_name +
                                " tensors.");
  }
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::invalid_argument(std::string(op_name) +
                                " only supports equal-shape tensors or scalar broadcasting.");
  }
  BroadcastInfo bi;
  bi.nx = nx;
  bi.ny = ny;
  bi.element_count = nx >= ny ? nx : ny;
  bi.shape = nx >= ny ? x.shape : y.shape;
  return bi;
}

/// Verifies the caller-supplied preallocated output tensor is non-null and
/// matches the expected dtype, shape and byte buffer size.
inline void CheckPreallocatedOutput(const char *op_name, const char *dtype_name,
                                    int32_t expected_dtype,
                                    const std::vector<int64_t> &expected_shape,
                                    size_t expected_bytes, const Tensor *output) {
  if (output == nullptr) {
    throw std::invalid_argument(std::string(op_name) +
                                " requires a non-null preallocated output tensor.");
  }
  if (output->data_type != expected_dtype) {
    throw std::invalid_argument(std::string(op_name) + " preallocated output must be a " +
                                dtype_name + " tensor.");
  }
  if (output->shape != expected_shape) {
    throw std::invalid_argument(std::string(op_name) +
                                " preallocated output shape must match the broadcasted "
                                "input shape.");
  }
  if (output->data.size() != expected_bytes) {
    throw std::invalid_argument(std::string(op_name) +
                                " preallocated output buffer has unexpected size in bytes.");
  }
}

/// In-place element-wise binary kernel driver. Validates inputs + output then
/// invokes ``op(a, b) -> TOut`` for each element pair (with scalar
/// broadcasting). ``TIn`` and ``TOut`` must match the byte layout of the
/// ``expected_dtype``.
template <typename TIn, typename TOut, typename Op>
void BinaryElementwise(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                       const Tensor &x, const Tensor &y, Tensor *output, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, expected_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, dtype_name, expected_dtype, bi.shape, expected_bytes, output);

  const TIn *px = reinterpret_cast<const TIn *>(x.data.data());
  const TIn *py = reinterpret_cast<const TIn *>(y.data.data());
  TOut *pz = reinterpret_cast<TOut *>(output->data.data());
  for (int64_t i = 0; i < bi.element_count; ++i) {
    const TIn a = bi.nx == 1 ? px[0] : px[i];
    const TIn b = bi.ny == 1 ? py[0] : py[i];
    pz[static_cast<size_t>(i)] = op(a, b);
  }
}

/// Allocating element-wise binary kernel driver. Builds the output tensor
/// with the broadcasted shape and ``expected_dtype``, then delegates to
/// :cpp:func:`BinaryElementwise` to fill it in.
template <typename TIn, typename TOut, typename Op>
Tensor BinaryElementwiseAlloc(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                              const Tensor &x, const Tensor &y, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, expected_dtype, x, y);
  Tensor z("", expected_dtype, bi.shape,
           std::vector<uint8_t>(static_cast<size_t>(bi.element_count) * sizeof(TOut)));
  BinaryElementwise<TIn, TOut>(op_name, dtype_name, expected_dtype, x, y, &z, op);
  return z;
}

} // namespace detail
} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
