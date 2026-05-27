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
// overloads and by the in-place ``void operator()(..., Tensor &) const``
// overloads, plus the broadcasted iteration loop itself.
//
// Error messages match the long-standing per-kernel wording so existing
// tests keep passing. Two short labels are required to build them:
//   * ``op_name``        e.g. "kernel::Add"
//   * ``dtype_name``     e.g. "FLOAT", "BOOL"
// ---------------------------------------------------------------------------

/// Information about a validated binary broadcast: the output shape, total
/// element count, the individual input element counts, and per-input
/// element-strides aligned to the output rank (a stride of 0 marks a
/// broadcast dimension). The rank-aligned ``shape_x``/``shape_y`` are also
/// reported for diagnostics. ``nx``/``ny`` are kept for fast-path detection
/// (equal-shape or scalar broadcasting).
struct BroadcastInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> shape_x;
  std::vector<int64_t> shape_y;
  std::vector<int64_t> strides_x;
  std::vector<int64_t> strides_y;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

/// Verifies both inputs have ``expected_dtype`` and that their shapes are
/// multidirectional-broadcastable per the standard NumPy/ONNX rules. Throws
/// ``std::invalid_argument`` otherwise.
BroadcastInfo CheckBinaryBroadcast(const char *op_name, const char *dtype_name,
                                   int32_t expected_dtype, const Tensor &x, const Tensor &y);

/// Variant of :cpp:func:`CheckBinaryBroadcast` for kernels whose input and
/// output dtypes differ (e.g. ``Greater``/``Less`` take numeric inputs and
/// return ``BOOL`` outputs). Validates that both inputs have
/// ``expected_in_dtype`` and computes the broadcast info; the caller is
/// responsible for validating the output against its own dtype.
BroadcastInfo CheckBinaryBroadcastInOut(const char *op_name, const char *in_dtype_name,
                                        int32_t expected_in_dtype, const Tensor &x,
                                        const Tensor &y);

/// Verifies the caller-supplied preallocated output tensor matches the
/// expected dtype, shape and byte buffer size.
void CheckPreallocatedOutput(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                             const std::vector<int64_t> &expected_shape, size_t expected_bytes,
                             const Tensor &output);

/// In-place element-wise binary kernel driver. Validates inputs + output then
/// invokes ``op(a, b) -> TOut`` for each element pair, with full
/// multidirectional broadcasting. ``TIn`` and ``TOut`` must match the byte
/// layout of the ``expected_dtype``.
template <typename TIn, typename TOut, typename Op>
void BinaryElementwise(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                       const Tensor &x, const Tensor &y, Tensor &output, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, expected_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, dtype_name, expected_dtype, bi.shape, expected_bytes, output);

  const TIn *px = reinterpret_cast<const TIn *>(x.data.data());
  const TIn *py = reinterpret_cast<const TIn *>(y.data.data());
  TOut *pz = reinterpret_cast<TOut *>(output.data.data());

  // Fast paths: equal-shape and scalar broadcasting.
  if (x.shape == y.shape) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      pz[static_cast<size_t>(i)] = op(px[i], py[i]);
    }
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      const TIn a = bi.nx == 1 ? px[0] : px[i];
      const TIn b = bi.ny == 1 ? py[0] : py[i];
      pz[static_cast<size_t>(i)] = op(a, b);
    }
    return;
  }

  // General multidirectional broadcasting: iterate over output coordinates in
  // row-major order using the pre-computed per-input element strides.
  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    pz[static_cast<size_t>(flat)] = op(px[ox], py[oy]);
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
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
  BinaryElementwise<TIn, TOut>(op_name, dtype_name, expected_dtype, x, y, z, op);
  return z;
}

/// Variant of :cpp:func:`BinaryElementwise` for kernels whose input and
/// output dtypes differ (e.g. ``Greater``/``Less``). Validates that both
/// inputs have ``in_dtype`` and that the preallocated output has
/// ``out_dtype`` and the broadcasted shape, then invokes
/// ``op(a, b) -> TOut`` for each element pair with full multidirectional
/// broadcasting.
template <typename TIn, typename TOut, typename Op>
void BinaryElementwiseInOut(const char *op_name, const char *in_dtype_name, int32_t in_dtype,
                            const char *out_dtype_name, int32_t out_dtype, const Tensor &x,
                            const Tensor &y, Tensor &output, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, out_dtype_name, out_dtype, bi.shape, expected_bytes, output);

  const TIn *px = reinterpret_cast<const TIn *>(x.data.data());
  const TIn *py = reinterpret_cast<const TIn *>(y.data.data());
  TOut *pz = reinterpret_cast<TOut *>(output.data.data());

  if (x.shape == y.shape) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      pz[static_cast<size_t>(i)] = op(px[i], py[i]);
    }
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      const TIn a = bi.nx == 1 ? px[0] : px[i];
      const TIn b = bi.ny == 1 ? py[0] : py[i];
      pz[static_cast<size_t>(i)] = op(a, b);
    }
    return;
  }

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    pz[static_cast<size_t>(flat)] = op(px[ox], py[oy]);
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
}

/// Allocating variant of :cpp:func:`BinaryElementwiseInOut`. Builds the
/// output tensor with the broadcasted shape and ``out_dtype``, then
/// delegates to :cpp:func:`BinaryElementwiseInOut` to fill it in.
template <typename TIn, typename TOut, typename Op>
Tensor BinaryElementwiseAllocInOut(const char *op_name, const char *in_dtype_name, int32_t in_dtype,
                                   const char *out_dtype_name, int32_t out_dtype, const Tensor &x,
                                   const Tensor &y, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  Tensor z("", out_dtype, bi.shape,
           std::vector<uint8_t>(static_cast<size_t>(bi.element_count) * sizeof(TOut)));
  BinaryElementwiseInOut<TIn, TOut>(op_name, in_dtype_name, in_dtype, out_dtype_name, out_dtype, x,
                                    y, z, op);
  return z;
}

} // namespace detail
} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
