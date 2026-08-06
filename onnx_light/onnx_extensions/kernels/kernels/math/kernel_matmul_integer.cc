// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/math/matmul_shape_utils.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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

// Validates the optional zero-point tensor `t` and returns the number of
// zero-point values it contributes.
// - Empty tensor (absent input): returns 1 — a single zero broadcast to all positions.
// - Scalar (0-D) or 1-D of size 1: returns 1 (per-tensor).
// - 1-D of size `expected_size`: returns `expected_size` (per-row or per-column).
// Any other shape triggers an assertion failure.
size_t ZeroPointCount(const Tensor &t, int32_t expected_dtype, int64_t expected_size,
                      const char *name) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    return 1;
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
  return static_cast<size_t>(numel);
}

// Fills `out` with `count` zero-point values read from `t`. For an absent input
// (empty tensor) a single zero is written; allocator-backed storage is not
// zero-initialized, so the zero is written explicitly.
void FillZeroPoints(const Tensor &t, int32_t *out, size_t count) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    out[0] = 0;
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    out[i] = ReadIntElem(t, static_cast<int64_t>(i));
  }
}

Shape ComputeStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
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

void RunMatMulInteger(const Tensor &a, const int32_t *a_values, size_t a_size, const Tensor &b,
                      const int32_t *b_values, size_t b_size, Tensor &output) {
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

  const Shape a_strides = ComputeStrides(a2);
  const Shape b_strides = ComputeStrides(b2);
  const Shape out_strides = ComputeStrides(output.shape);

  const size_t a_prefix_rank = a_prefix.size();
  const size_t b_prefix_rank = b_prefix.size();

  int64_t batch_count = 1;
  for (int64_t d : out_prefix) {
    batch_count *= d;
  }

  int32_t *py = output.AsInt32();

  const bool a_per_row = a_size > 1;
  const bool b_per_col = b_size > 1;

  Shape batch_idx;
  batch_idx.assign(batch_rank, 0);
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

  const size_t a_count = ZeroPointCount(a_zero_point, a.data_type, M, "a_zero_point");
  const size_t b_count = ZeroPointCount(b_zero_point, b.data_type, N, "b_zero_point");
  core::runtime::detail::TemporaryTypedBuffer<int32_t> a_zps(a_count, allocator, "a_zero_point");
  core::runtime::detail::TemporaryTypedBuffer<int32_t> b_zps(b_count, allocator, "b_zero_point");
  FillZeroPoints(a_zero_point, a_zps.data(), a_count);
  FillZeroPoints(b_zero_point, b_zps.data(), b_count);

  RunMatMulInteger(a, a_zps.data(), a_count, b, b_zps.data(), b_count, output);
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

void MatMulInteger::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 2);
  EXT_ENFORCE_INVALID(!(node.input_size() > 4),
                      "RunNode: op 'MatMulInteger' expects at most 4 inputs, got ",
                      node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Tensor &a = GetInput(node, 0, rt.tensors());
  const Tensor &b = GetInput(node, 1, rt.tensors());
  const Tensor *a_zp = GetOptionalInput(node, 2, rt.tensors());
  const Tensor *b_zp = GetOptionalInput(node, 3, rt.tensors());
  onnx_kernels::kernel::MatMulInteger k(rt.kernel_ctx());
  SetOutput(node, 0,
            k(a, b, a_zp != nullptr ? *a_zp : Tensor{}, b_zp != nullptr ? *b_zp : Tensor{}),
            rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
