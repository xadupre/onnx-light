// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/math/matmul_shape_utils.h"

#include "onnx_kernels/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::MatMulInteger";

inline bool IsInt8OrUint8(int32_t dt) {
  return dt == static_cast<int32_t>(DataType::INT8) || dt == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadIntElem(const Tensor &t, int64_t idx) {
  if (t.data_type == static_cast<int32_t>(DataType::INT8)) {
    return static_cast<int32_t>(t.AsInt8()[idx]);
  }
  return static_cast<int32_t>(t.AsUint8()[idx]);
}

// Holds zero-point values either in a fallback std::vector<int32_t> or in
// allocator-backed RawBuffer storage. Takes ownership of allocator-backed
// storage and releases it upon destruction.
struct ZeroPointValues {
  std::vector<int32_t> fallback;
  // `buffer` is the active-storage discriminator: non-null means allocator-backed
  // storage, null means fallback vector storage.
  RawBufferAllocator *allocator = nullptr;
  RawBuffer *buffer = nullptr;
  size_t size = 0;

  ZeroPointValues() = default;
  ZeroPointValues(const ZeroPointValues &) = delete;
  ZeroPointValues &operator=(const ZeroPointValues &) = delete;

  ZeroPointValues(ZeroPointValues &&other) noexcept
      : fallback(std::move(other.fallback)), allocator(other.allocator), buffer(other.buffer),
        size(other.size) {
    other.allocator = nullptr;
    other.buffer = nullptr;
    other.size = 0;
  }

  ZeroPointValues &operator=(ZeroPointValues &&other) noexcept {
    if (this != &other) {
      if (buffer != nullptr && allocator != nullptr) {
        allocator->Free(buffer);
      }
      fallback = std::move(other.fallback);
      allocator = other.allocator;
      buffer = other.buffer;
      size = other.size;
      other.allocator = nullptr;
      other.buffer = nullptr;
      other.size = 0;
    }
    return *this;
  }

  ~ZeroPointValues() {
    if (buffer != nullptr && allocator != nullptr) {
      allocator->Free(buffer);
    }
  }

  bool is_allocator_backed() const noexcept { return buffer != nullptr; }

  // Returns writable zero-point values from the active storage backend.
  int32_t *mutable_data() {
    if (!is_allocator_backed()) {
      return fallback.data();
    }
    return reinterpret_cast<int32_t *>(buffer->data());
  }
  // Returns read-only zero-point values from the active storage backend.
  const int32_t *data() const {
    if (!is_allocator_backed()) {
      return fallback.data();
    }
    return reinterpret_cast<const int32_t *>(buffer->data());
  }
};

// Returns zero-point values for the given optional zero-point tensor.
// - Empty tensor (absent input): returns {0} — scalar zero broadcast to all positions.
// - Scalar (0-D) or 1-D of size 1: returns a one-element vector (per-tensor).
// - 1-D of size `expected_size`: returns all values (per-row or per-column).
// Any other shape triggers an assertion failure.
// Uses allocator-backed storage when `allocator` is provided.
ZeroPointValues ReadZeroPoints(const Tensor &t, int32_t expected_dtype, int64_t expected_size,
                               const char *name, RawBufferAllocator *allocator) {
  ZeroPointValues zps;
  if (t.shape.empty() && t.size_bytes() == 0) {
    zps.fallback = {0};
    zps.size = 1;
    return zps;
  }
  EXT_ENFORCE_INVALID(t.data_type == expected_dtype, kName, ": '", name,
                      "' dtype must match its data input.");
  EXT_ENFORCE_INVALID(t.shape.size() <= 1, kName, ": '", name,
                      "' must be a scalar or a 1-D tensor.");
  int64_t numel = 1;
  for (int64_t d : t.shape) {
    numel *= d;
  }
  EXT_ENFORCE_INVALID(numel == 1 || numel == expected_size, kName, ": '", name,
                      "' must be a scalar, a one-element 1-D tensor, or a 1-D tensor whose "
                      "size matches the corresponding matrix dimension.");
  const size_t numel_u = static_cast<size_t>(numel);

  zps.size = numel_u;
  if (allocator != nullptr) {
    zps.allocator = allocator;
    zps.buffer = allocator->Allocate(numel_u * sizeof(int32_t));
    EXT_ENFORCE_INVALID(zps.buffer != nullptr, kName, ": zero-point allocator returned null.");
    // RawBufferAllocator::Allocate returns at least n_bytes, so >= is expected.
    EXT_ENFORCE_INVALID(zps.buffer->size() >= numel_u * sizeof(int32_t), kName,
                        ": zero-point allocator returned too small a buffer.");
    EXT_ENFORCE_INVALID(reinterpret_cast<std::uintptr_t>(zps.buffer->data()) % alignof(int32_t) ==
                            0,
                        kName, ": allocator returned misaligned zero-point buffer.");
  } else {
    zps.fallback.resize(numel_u);
  }

  int32_t *out = zps.mutable_data();
  for (int64_t i = 0; i < numel; ++i) {
    out[static_cast<size_t>(i)] = ReadIntElem(t, i);
  }
  return zps;
}

std::vector<int64_t> ComputeStrides(const Shape &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

Shape ComputeOutputShape(const Shape &a_shape, const Shape &b_shape) {
  return detail::ComputeMatMulOutputShape(
      a_shape, b_shape, kName, ": rank-0 inputs are not accepted.",
      ": incompatible inner dimensions.",
      ": inputs are not broadcast-compatible on batch dimensions.");
}

void RunMatMulInteger(const Tensor &a, const ZeroPointValues &a_zps, const Tensor &b,
                      const ZeroPointValues &b_zps, Tensor &output) {
  const Shape a2 = detail::PromoteMatMulShape(a.shape, true);
  const Shape b2 = detail::PromoteMatMulShape(b.shape, false);
  const int64_t M = a2[a2.size() - 2];
  const int64_t K = a2[a2.size() - 1];
  const int64_t N = b2[b2.size() - 1];

  Shape a_prefix, b_prefix;
  a_prefix.insert(a_prefix.begin(), a2.begin(), a2.end() - 2);
  b_prefix.insert(b_prefix.begin(), b2.begin(), b2.end() - 2);
  const Shape out_prefix = detail::BroadcastMatMulPrefix(
      a_prefix, b_prefix, kName, ": inputs are not broadcast-compatible on batch dimensions.");
  const size_t batch_rank = out_prefix.size();

  const std::vector<int64_t> a_strides = ComputeStrides(a2);
  const std::vector<int64_t> b_strides = ComputeStrides(b2);
  const std::vector<int64_t> out_strides = ComputeStrides(output.shape);

  const size_t a_prefix_rank = a_prefix.size();
  const size_t b_prefix_rank = b_prefix.size();

  int64_t batch_count = 1;
  for (int64_t d : out_prefix) {
    batch_count *= d;
  }

  int32_t *py = output.AsInt32();

  const bool a_per_row = a_zps.size > 1;
  const bool b_per_col = b_zps.size > 1;
  const int32_t *a_values = a_zps.data();
  const int32_t *b_values = b_zps.data();

  std::vector<int64_t> batch_idx(batch_rank, 0);
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t y_base = 0;
    for (size_t d = 0; d < batch_rank; ++d) {
      const int64_t coord = batch_idx[d];
      if (d + a_prefix_rank >= batch_rank) {
        const size_t a_dim = d - (batch_rank - a_prefix_rank);
        const int64_t a_coord = (a_prefix[a_dim] == 1) ? 0 : coord;
        a_base += a_coord * a_strides[a_dim];
      }
      if (d + b_prefix_rank >= batch_rank) {
        const size_t b_dim = d - (batch_rank - b_prefix_rank);
        const int64_t b_coord = (b_prefix[b_dim] == 1) ? 0 : coord;
        b_base += b_coord * b_strides[b_dim];
      }
      y_base += coord * out_strides[d];
    }

    const int64_t a_row_stride = a_strides[a2.size() - 2];
    const int64_t a_k_stride = a_strides[a2.size() - 1];
    const int64_t b_k_stride = b_strides[b2.size() - 2];
    const int64_t b_col_stride = b_strides[b2.size() - 1];

    for (int64_t i = 0; i < M; ++i) {
      const int32_t a_row_zp = a_per_row ? a_values[static_cast<size_t>(i)] : a_values[0];
      for (int64_t j = 0; j < N; ++j) {
        const int32_t b_col_zp = b_per_col ? b_values[static_cast<size_t>(j)] : b_values[0];
        int32_t acc = 0;
        for (int64_t kk = 0; kk < K; ++kk) {
          const int32_t av = ReadIntElem(a, a_base + i * a_row_stride + kk * a_k_stride);
          const int32_t bv = ReadIntElem(b, b_base + kk * b_k_stride + j * b_col_stride);
          acc += (av - a_row_zp) * (bv - b_col_zp);
        }
        int64_t y_index = y_base;
        if (a.shape.size() != 1 && b.shape.size() != 1) {
          y_index += i * out_strides[batch_rank] + j * out_strides[batch_rank + 1];
        } else if (a.shape.size() == 1 && b.shape.size() != 1) {
          y_index += j * out_strides[batch_rank];
        } else if (a.shape.size() != 1 && b.shape.size() == 1) {
          y_index += i * out_strides[batch_rank];
        }
        py[y_index] = acc;
      }
    }

    for (size_t d = batch_rank; d-- > 0;) {
      if (++batch_idx[d] < out_prefix[d]) {
        break;
      }
      batch_idx[d] = 0;
    }
  }
}

void ComputeMatMulInteger(const Tensor &a, const Tensor &b, const Tensor &a_zero_point,
                          const Tensor &b_zero_point, Tensor &output,
                          RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type), kName, ": A must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type), kName, ": B must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT32), kName,
                      ": output dtype must be INT32.");

  const Shape out_shape = ComputeOutputShape(a.shape, b.shape);
  EXT_ENFORCE_INVALID(output.shape == out_shape, kName, ": preallocated output has invalid shape.");
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(total) * sizeof(int32_t), kName,
                      ": preallocated output buffer size does not match its shape.");

  const Shape a2 = detail::PromoteMatMulShape(a.shape, true);
  const Shape b2 = detail::PromoteMatMulShape(b.shape, false);
  const int64_t M = a2[a2.size() - 2];
  const int64_t N = b2[b2.size() - 1];

  ZeroPointValues a_zps = ReadZeroPoints(a_zero_point, a.data_type, M, "a_zero_point", allocator);
  ZeroPointValues b_zps = ReadZeroPoints(b_zero_point, b.data_type, N, "b_zero_point", allocator);

  RunMatMulInteger(a, a_zps, b, b_zps, output);
}

} // namespace

Tensor MatMulInteger::operator()(const Tensor &a, const Tensor &b, const Tensor &a_zero_point,
                                 const Tensor &b_zero_point, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type), kName, ": A must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type), kName, ": B must be INT8 or UINT8.");

  const Shape out_shape = ComputeOutputShape(a.shape, b.shape);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const size_t out_n_bytes = static_cast<size_t>(total) * sizeof(int32_t);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::INT32), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  ComputeMatMulInteger(a, b, a_zero_point, b_zero_point, out, rt ? rt->allocator() : nullptr);
  return out;
}

void MatMulInteger::operator()(const Tensor &a, const Tensor &b, const Tensor &a_zero_point,
                               const Tensor &b_zero_point, Tensor &output) const {
  ComputeMatMulInteger(a, b, a_zero_point, b_zero_point, output, nullptr);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
