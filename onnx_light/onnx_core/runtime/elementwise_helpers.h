// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail {

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
  Shape shape;
  Shape shape_x;
  Shape shape_y;
  std::vector<int64_t> strides_x;
  std::vector<int64_t> strides_y;
  int64_t element_count = 0;
  int64_t nx = 0;
  int64_t ny = 0;
};

/// Returns the multidirectional-broadcast output shape of ``a`` and ``b``.
/// Throws ``std::invalid_argument`` with a message prefixed by ``op_name``
/// when the shapes are not broadcastable.
Shape BroadcastShape(const char *op_name, const Shape &a, const Shape &b);

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
                             const Shape &expected_shape, size_t expected_bytes,
                             const Tensor &output);

/// In-place element-wise binary kernel driver. Validates inputs + output then
/// invokes ``op(a, b) -> TOut`` for each element pair, with full
/// multidirectional broadcasting. ``TIn`` and ``TOut`` must match the byte
/// layout of the ``expected_dtype``.
template <typename TIn, typename TOut, typename Op>
void BinaryElementwise(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                       const Tensor &x, const Tensor &y, Tensor &output, Op op,
                       int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, expected_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, dtype_name, expected_dtype, bi.shape, expected_bytes, output);

  const TIn *px = reinterpret_cast<const TIn *>(x.bytes());
  const TIn *py = reinterpret_cast<const TIn *>(y.bytes());
  TOut *pz = reinterpret_cast<TOut *>(output.mutable_bytes());

  // Fast paths: equal-shape and scalar broadcasting.
  if (x.shape == y.shape) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, op](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    pz[static_cast<size_t>(i)] = op(px[i], py[i]);
                  }
                });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, op, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    const TIn a = nx == 1 ? px[0] : px[i];
                    const TIn b = ny == 1 ? py[0] : py[i];
                    pz[static_cast<size_t>(i)] = op(a, b);
                  }
                });
    return;
  }

  // General multidirectional broadcasting: iterate over output coordinates in
  // row-major order using the pre-computed per-input element strides.
  ParallelFor(bi.element_count, parallel_minimum_elements,
              [px, py, pz, op, &bi](int64_t begin, int64_t end) {
                const size_t rank = bi.shape.size();
                Shape idx;
                idx.assign(rank, 0);
                int64_t remaining = begin;
                for (size_t d = rank; d-- > 0;) {
                  idx[d] = remaining % bi.shape[d];
                  remaining /= bi.shape[d];
                }
                for (int64_t flat = begin; flat < end; ++flat) {
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
              });
}

/// Allocating element-wise binary kernel driver. Builds the output tensor
/// with the broadcasted shape and ``expected_dtype``, then delegates to
/// :cpp:func:`BinaryElementwise` to fill it in.
///
/// When ``allocator`` is non-null the output buffer is acquired from it
/// directly, so no copy is needed later when the tensor is stored in a
/// :cpp:class:`RuntimeContext`. Pass ``nullptr`` (or omit the argument) to
/// fall back to the legacy inline-allocation path.
template <typename TIn, typename TOut, typename Op>
Tensor
BinaryElementwiseAlloc(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                       const Tensor &x, const Tensor &y, Op op,
                       RawBufferAllocator *allocator = nullptr,
                       int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, expected_dtype, x, y);
  const size_t n_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  Tensor z = MakeOutputTensor(expected_dtype, bi.shape, n_bytes, allocator);
  BinaryElementwise<TIn, TOut>(op_name, dtype_name, expected_dtype, x, y, z, op,
                               parallel_minimum_elements);
  return z;
}

/// Variant of :cpp:func:`BinaryElementwise` for kernels whose input and
/// output dtypes differ (e.g. ``Greater``/``Less``). Validates that both
/// inputs have ``in_dtype`` and that the preallocated output has
/// ``out_dtype`` and the broadcasted shape, then invokes
/// ``op(a, b) -> TOut`` for each element pair with full multidirectional
/// broadcasting.
template <typename TIn, typename TOut, typename Op>
void BinaryElementwiseInOut(
    const char *op_name, const char *in_dtype_name, int32_t in_dtype, const char *out_dtype_name,
    int32_t out_dtype, const Tensor &x, const Tensor &y, Tensor &output, Op op,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, out_dtype_name, out_dtype, bi.shape, expected_bytes, output);

  const TIn *px = reinterpret_cast<const TIn *>(x.bytes());
  const TIn *py = reinterpret_cast<const TIn *>(y.bytes());
  TOut *pz = reinterpret_cast<TOut *>(output.mutable_bytes());

  if (x.shape == y.shape) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, op](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    pz[static_cast<size_t>(i)] = op(px[i], py[i]);
                  }
                });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, op, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    const TIn a = nx == 1 ? px[0] : px[i];
                    const TIn b = ny == 1 ? py[0] : py[i];
                    pz[static_cast<size_t>(i)] = op(a, b);
                  }
                });
    return;
  }

  ParallelFor(bi.element_count, parallel_minimum_elements,
              [px, py, pz, op, &bi](int64_t begin, int64_t end) {
                const size_t rank = bi.shape.size();
                Shape idx;
                idx.assign(rank, 0);
                int64_t remaining = begin;
                for (size_t d = rank; d-- > 0;) {
                  idx[d] = remaining % bi.shape[d];
                  remaining /= bi.shape[d];
                }
                for (int64_t flat = begin; flat < end; ++flat) {
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
              });
}

/// Allocating variant of :cpp:func:`BinaryElementwiseInOut`. Builds the
/// output tensor with the broadcasted shape and ``out_dtype``, then
/// delegates to :cpp:func:`BinaryElementwiseInOut` to fill it in.
///
/// When ``allocator`` is non-null the output buffer is acquired from it
/// directly. Pass ``nullptr`` (or omit the argument) to use inline
/// allocation.
template <typename TIn, typename TOut, typename Op>
Tensor BinaryElementwiseAllocInOut(
    const char *op_name, const char *in_dtype_name, int32_t in_dtype, const char *out_dtype_name,
    int32_t out_dtype, const Tensor &x, const Tensor &y, Op op,
    RawBufferAllocator *allocator = nullptr,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  const size_t n_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  Tensor z = MakeOutputTensor(out_dtype, bi.shape, n_bytes, allocator);
  BinaryElementwiseInOut<TIn, TOut>(op_name, in_dtype_name, in_dtype, out_dtype_name, out_dtype, x,
                                    y, z, op, parallel_minimum_elements);
  return z;
}

// ---------------------------------------------------------------------------
// Half-precision (FLOAT16 / BFLOAT16) element-wise helpers
//
// These perform decode → float32 op → encode in a single pass without
// intermediate tensor allocations. They reuse the broadcast machinery
// (BroadcastInfo / CheckBinaryBroadcast) but iterate with uint16_t pointers,
// calling the provided decode/encode functions per element.
// ---------------------------------------------------------------------------

using HalfDecodeFunc = float (*)(uint16_t);
using HalfEncodeFunc = uint16_t (*)(float);

/// In-place half-precision binary element-wise kernel.
template <typename Op>
void BinaryHalfElementwise(
    const char *op_name, const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
    Tensor &output, HalfDecodeFunc decode, HalfEncodeFunc encode, Op op,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(uint16_t);
  CheckPreallocatedOutput(op_name, dtype_name, dtype, bi.shape, expected_bytes, output);

  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  const uint16_t *py = reinterpret_cast<const uint16_t *>(y.bytes());
  uint16_t *pz = reinterpret_cast<uint16_t *>(output.mutable_bytes());

  if (x.shape == y.shape) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, decode, encode, op](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    pz[static_cast<size_t>(i)] = encode(op(decode(px[i]), decode(py[i])));
                  }
                });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(
        bi.element_count, parallel_minimum_elements,
        [px, py, pz, decode, encode, op, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            const float a = nx == 1 ? decode(px[0]) : decode(px[i]);
            const float b = ny == 1 ? decode(py[0]) : decode(py[i]);
            pz[static_cast<size_t>(i)] = encode(op(a, b));
          }
        });
    return;
  }

  ParallelFor(bi.element_count, parallel_minimum_elements,
              [px, py, pz, decode, encode, op, &bi](int64_t begin, int64_t end) {
                const size_t rank = bi.shape.size();
                Shape idx;
                idx.assign(rank, 0);
                int64_t remaining = begin;
                for (size_t d = rank; d-- > 0;) {
                  idx[d] = remaining % bi.shape[d];
                  remaining /= bi.shape[d];
                }
                for (int64_t flat = begin; flat < end; ++flat) {
                  int64_t ox = 0, oy = 0;
                  for (size_t d = 0; d < rank; ++d) {
                    ox += idx[d] * bi.strides_x[d];
                    oy += idx[d] * bi.strides_y[d];
                  }
                  pz[static_cast<size_t>(flat)] = encode(op(decode(px[ox]), decode(py[oy])));
                  for (size_t d = rank; d-- > 0;) {
                    if (++idx[d] < bi.shape[d]) {
                      break;
                    }
                    idx[d] = 0;
                  }
                }
              });
}

/// Allocating half-precision binary element-wise kernel.
///
/// When ``allocator`` is non-null the output buffer is acquired from it
/// directly. Pass ``nullptr`` (or omit the argument) to use inline
/// allocation.
template <typename Op>
Tensor BinaryHalfElementwiseAlloc(
    const char *op_name, const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
    HalfDecodeFunc decode, HalfEncodeFunc encode, Op op, RawBufferAllocator *allocator = nullptr,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcast(op_name, dtype_name, dtype, x, y);
  const size_t n_bytes = static_cast<size_t>(bi.element_count) * sizeof(uint16_t);
  Tensor z = MakeOutputTensor(dtype, bi.shape, n_bytes, allocator);
  BinaryHalfElementwise(op_name, dtype_name, dtype, x, y, z, decode, encode, op,
                        parallel_minimum_elements);
  return z;
}

/// In-place half-precision binary kernel for FLOAT16/BFLOAT16 inputs and a
/// distinct POD output dtype (for example BOOL in comparison operators).
template <typename TOut, typename Op>
void BinaryHalfElementwiseInOut(const char *op_name, const char *in_dtype_name, int32_t in_dtype,
                                const char *out_dtype_name, int32_t out_dtype, const Tensor &x,
                                const Tensor &y, Tensor &output, HalfDecodeFunc decode, Op op) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  CheckPreallocatedOutput(op_name, out_dtype_name, out_dtype, bi.shape, expected_bytes, output);

  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  const uint16_t *py = reinterpret_cast<const uint16_t *>(y.bytes());
  TOut *pz = reinterpret_cast<TOut *>(output.mutable_bytes());

  if (x.shape == y.shape) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      pz[static_cast<size_t>(i)] = op(decode(px[i]), decode(py[i]));
    }
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      const float a = bi.nx == 1 ? decode(px[0]) : decode(px[i]);
      const float b = bi.ny == 1 ? decode(py[0]) : decode(py[i]);
      pz[static_cast<size_t>(i)] = op(a, b);
    }
    return;
  }

  const size_t rank = bi.shape.size();
  Shape idx;
  idx.assign(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    pz[static_cast<size_t>(flat)] = op(decode(px[ox]), decode(py[oy]));
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
}

/// Allocating variant of :cpp:func:`BinaryHalfElementwiseInOut`.
///
/// When ``allocator`` is non-null the output buffer is acquired from it
/// directly. Pass ``nullptr`` (or omit the argument) to use inline
/// allocation.
template <typename TOut, typename Op>
Tensor BinaryHalfElementwiseAllocInOut(const char *op_name, const char *in_dtype_name,
                                       int32_t in_dtype, const char *out_dtype_name,
                                       int32_t out_dtype, const Tensor &x, const Tensor &y,
                                       HalfDecodeFunc decode, Op op,
                                       RawBufferAllocator *allocator = nullptr) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, in_dtype_name, in_dtype, x, y);
  const size_t n_bytes = static_cast<size_t>(bi.element_count) * sizeof(TOut);
  Tensor z = MakeOutputTensor(out_dtype, bi.shape, n_bytes, allocator);
  BinaryHalfElementwiseInOut<TOut>(op_name, in_dtype_name, in_dtype, out_dtype_name, out_dtype, x,
                                   y, z, decode, op);
  return z;
}

/// Unary half-precision element-wise kernel (single-pass, no allocation).
template <typename Op>
void UnaryHalfElementwise(const Tensor &x, Tensor &output, HalfDecodeFunc decode,
                          HalfEncodeFunc encode, int64_t grain_size, Op op) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py, decode, encode, op](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[i] = encode(op(decode(px[i])));
    }
  });
}

/// Unary half-precision element-wise kernel using the default parallel grain.
template <typename Op>
void UnaryHalfElementwise(const Tensor &x, Tensor &output, HalfDecodeFunc decode,
                          HalfEncodeFunc encode, Op op) {
  UnaryHalfElementwise(x, output, decode, encode, kParallelForGrainSize, op);
}

/// In-place half-precision binary comparison kernel (decode→compare→BOOL).
template <typename Op>
void BinaryHalfCompareElementwise(
    const char *op_name, const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
    Tensor &output, HalfDecodeFunc decode, Op op,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, dtype_name, dtype, x, y);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * sizeof(uint8_t);
  CheckPreallocatedOutput(op_name, "BOOL", DataType::BOOL, bi.shape, expected_bytes, output);

  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  const uint16_t *py = reinterpret_cast<const uint16_t *>(y.bytes());
  uint8_t *pz = output.mutable_bytes();

  if (x.shape == y.shape) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, decode, op](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    pz[static_cast<size_t>(i)] = op(decode(px[i]), decode(py[i]));
                  }
                });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(bi.element_count, parallel_minimum_elements,
                [px, py, pz, decode, op, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    const float a = nx == 1 ? decode(px[0]) : decode(px[i]);
                    const float b = ny == 1 ? decode(py[0]) : decode(py[i]);
                    pz[static_cast<size_t>(i)] = op(a, b);
                  }
                });
    return;
  }

  ParallelFor(bi.element_count, parallel_minimum_elements,
              [px, py, pz, decode, op, &bi](int64_t begin, int64_t end) {
                const size_t rank = bi.shape.size();
                Shape idx;
                idx.assign(rank, 0);
                int64_t remaining = begin;
                for (size_t d = rank; d-- > 0;) {
                  idx[d] = remaining % bi.shape[d];
                  remaining /= bi.shape[d];
                }
                for (int64_t flat = begin; flat < end; ++flat) {
                  int64_t ox = 0, oy = 0;
                  for (size_t d = 0; d < rank; ++d) {
                    ox += idx[d] * bi.strides_x[d];
                    oy += idx[d] * bi.strides_y[d];
                  }
                  pz[static_cast<size_t>(flat)] = op(decode(px[ox]), decode(py[oy]));
                  for (size_t d = rank; d-- > 0;) {
                    if (++idx[d] < bi.shape[d]) {
                      break;
                    }
                    idx[d] = 0;
                  }
                }
              });
}

/// Allocating half-precision binary comparison kernel (decode→compare→BOOL).
///
/// When ``allocator`` is non-null the output buffer is acquired from it
/// directly. Pass ``nullptr`` (or omit the argument) to use inline
/// allocation.
template <typename Op>
Tensor BinaryHalfCompareElementwiseAlloc(
    const char *op_name, const char *dtype_name, int32_t dtype, const Tensor &x, const Tensor &y,
    HalfDecodeFunc decode, Op op, RawBufferAllocator *allocator = nullptr,
    int64_t parallel_minimum_elements = std::numeric_limits<int64_t>::max()) {
  const BroadcastInfo bi = CheckBinaryBroadcastInOut(op_name, dtype_name, dtype, x, y);
  const size_t n_bytes = static_cast<size_t>(bi.element_count) * sizeof(uint8_t);
  Tensor z = MakeOutputTensor(DataType::BOOL, bi.shape, n_bytes, allocator);
  BinaryHalfCompareElementwise(op_name, dtype_name, dtype, x, y, z, decode, op,
                               parallel_minimum_elements);
  return z;
}

/** Dispatches one numeric binary comparison and allocates its BOOL output. */
template <typename Op>
Tensor BinaryComparisonAlloc(const char *op_name, const Tensor &x, const Tensor &y, Op op,
                             RawBufferAllocator *allocator, int64_t parallel_minimum_elements) {
#define ONNX_LIGHT_COMPARE_CASE(ENUM, NAME, CTYPE)                                                 \
  case DataType::ENUM:                                                                             \
    return BinaryElementwiseAllocInOut<CTYPE, uint8_t>(op_name, NAME, DataType::ENUM, "BOOL",      \
                                                       DataType::BOOL, x, y, op, allocator,        \
                                                       parallel_minimum_elements)
  switch (x.data_type) {
    ONNX_LIGHT_COMPARE_CASE(FLOAT, "FLOAT", float);
    ONNX_LIGHT_COMPARE_CASE(INT8, "INT8", int8_t);
    ONNX_LIGHT_COMPARE_CASE(INT16, "INT16", int16_t);
    ONNX_LIGHT_COMPARE_CASE(INT32, "INT32", int32_t);
    ONNX_LIGHT_COMPARE_CASE(INT64, "INT64", int64_t);
    ONNX_LIGHT_COMPARE_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_COMPARE_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_COMPARE_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_COMPARE_CASE(UINT64, "UINT64", uint64_t);
  case DataType::FLOAT16:
    return BinaryHalfCompareElementwiseAlloc(op_name, "FLOAT16", DataType::FLOAT16, x, y,
                                             Float16BitsToFloat, op, allocator,
                                             parallel_minimum_elements);
  case DataType::BFLOAT16:
    return BinaryHalfCompareElementwiseAlloc(op_name, "BFLOAT16", DataType::BFLOAT16, x, y,
                                             Bfloat16BitsToFloat, op, allocator,
                                             parallel_minimum_elements);
  default:
    EXT_THROW_INVALID(op_name, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
#undef ONNX_LIGHT_COMPARE_CASE
}

/** Dispatches one numeric binary comparison into a preallocated BOOL output. */
template <typename Op>
void BinaryComparison(const char *op_name, const Tensor &x, const Tensor &y, Tensor &output, Op op,
                      int64_t parallel_minimum_elements) {
#define ONNX_LIGHT_COMPARE_CASE(ENUM, NAME, CTYPE)                                                 \
  case DataType::ENUM:                                                                             \
    BinaryElementwiseInOut<CTYPE, uint8_t>(op_name, NAME, DataType::ENUM, "BOOL", DataType::BOOL,  \
                                           x, y, output, op, parallel_minimum_elements);           \
    return
  switch (x.data_type) {
    ONNX_LIGHT_COMPARE_CASE(FLOAT, "FLOAT", float);
    ONNX_LIGHT_COMPARE_CASE(INT8, "INT8", int8_t);
    ONNX_LIGHT_COMPARE_CASE(INT16, "INT16", int16_t);
    ONNX_LIGHT_COMPARE_CASE(INT32, "INT32", int32_t);
    ONNX_LIGHT_COMPARE_CASE(INT64, "INT64", int64_t);
    ONNX_LIGHT_COMPARE_CASE(UINT8, "UINT8", uint8_t);
    ONNX_LIGHT_COMPARE_CASE(UINT16, "UINT16", uint16_t);
    ONNX_LIGHT_COMPARE_CASE(UINT32, "UINT32", uint32_t);
    ONNX_LIGHT_COMPARE_CASE(UINT64, "UINT64", uint64_t);
  case DataType::FLOAT16:
    BinaryHalfCompareElementwise(op_name, "FLOAT16", DataType::FLOAT16, x, y, output,
                                 Float16BitsToFloat, op, parallel_minimum_elements);
    return;
  case DataType::BFLOAT16:
    BinaryHalfCompareElementwise(op_name, "BFLOAT16", DataType::BFLOAT16, x, y, output,
                                 Bfloat16BitsToFloat, op, parallel_minimum_elements);
    return;
  default:
    EXT_THROW_INVALID(op_name, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32 and UINT64 inputs.");
  }
#undef ONNX_LIGHT_COMPARE_CASE
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail
