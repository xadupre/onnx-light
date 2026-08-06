// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/simple_tensor.h"

#include <algorithm>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::detail {

/// Promotes a 1-D MatMul operand to its logical 2-D shape.
///
/// For the left operand, ``[K]`` becomes ``[1, K]``. For the right operand,
/// ``[K]`` becomes ``[K, 1]``. Higher-rank shapes are returned unchanged.
///
/// Parameters:
///   shape: Input operand shape.
///   is_left: Indicates whether the operand is the left-hand MatMul input.
///
/// Returns:
///   The promoted shape, suitable for MatMul rank handling.
inline Shape PromoteMatMulShape(const Shape &shape, bool is_left) {
  if (shape.size() == 1) {
    if (is_left) {
      return {1, shape[0]};
    }
    return {shape[0], 1};
  }
  return shape;
}

/// Broadcasts MatMul batch-prefix dimensions with NumPy/ONNX rules.
///
/// ``a_prefix`` and ``b_prefix`` are the prefixes before trailing matrix
/// dimensions. ``op_name`` and ``broadcast_error_suffix`` build kernel-specific
/// error messages when prefixes are incompatible.
///
/// Parameters:
///   a_prefix: Left input batch-prefix shape.
///   b_prefix: Right input batch-prefix shape.
///   op_name: Kernel/operator name used in error messages.
///   broadcast_error_suffix: Message suffix for incompatible broadcast prefixes.
///
/// Returns:
///   The broadcasted batch-prefix shape.
inline Shape BroadcastMatMulPrefix(const Shape &a_prefix, const Shape &b_prefix,
                                   const char *op_name, const char *broadcast_error_suffix) {
  const size_t rank = std::max(a_prefix.size(), b_prefix.size());
  Shape out;
  out.assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const bool has_a = i + a_prefix.size() >= rank;
    const bool has_b = i + b_prefix.size() >= rank;
    const int64_t da = has_a ? a_prefix[i - (rank - a_prefix.size())] : 1;
    const int64_t db = has_b ? b_prefix[i - (rank - b_prefix.size())] : 1;
    if (da == db || da == 1) {
      out[i] = db;
    } else if (db == 1) {
      out[i] = da;
    } else {
      EXT_THROW_INVALID(op_name, broadcast_error_suffix);
    }
  }
  return out;
}

/// Computes the MatMul output shape for two input shapes.
///
/// The helper applies 1-D promotions, validates inner dimensions, broadcasts
/// batch prefixes, and appends matrix dimensions according to MatMul rules.
/// Error-message suffixes are provided by callers to preserve per-kernel text.
///
/// Parameters:
///   a_shape: Left MatMul input shape.
///   b_shape: Right MatMul input shape.
///   op_name: Kernel/operator name used in error messages.
///   rank0_error_suffix: Message suffix for rank-0 input validation failures.
///   inner_dim_error_suffix: Message suffix for inner-dimension mismatches.
///   broadcast_error_suffix: Message suffix for batch-prefix broadcast failures.
///
/// Returns:
///   The full MatMul output shape.
inline Shape ComputeMatMulOutputShape(const Shape &a_shape, const Shape &b_shape,
                                      const char *op_name, const char *rank0_error_suffix,
                                      const char *inner_dim_error_suffix,
                                      const char *broadcast_error_suffix) {
  EXT_ENFORCE_INVALID(!a_shape.empty() && !b_shape.empty(), op_name, rank0_error_suffix);
  const Shape a2 = PromoteMatMulShape(a_shape, true);
  const Shape b2 = PromoteMatMulShape(b_shape, false);
  EXT_ENFORCE_INVALID(a2[a2.size() - 1] == b2[b2.size() - 2], op_name, inner_dim_error_suffix);

  Shape a_prefix, b_prefix;
  a_prefix.insert(a_prefix.begin(), a2.begin(), a2.end() - 2);
  b_prefix.insert(b_prefix.begin(), b2.begin(), b2.end() - 2);
  Shape out_shape = BroadcastMatMulPrefix(a_prefix, b_prefix, op_name, broadcast_error_suffix);

  if (a_shape.size() != 1) {
    out_shape.push_back(a2[a2.size() - 2]);
  }
  if (b_shape.size() != 1) {
    out_shape.push_back(b2[b2.size() - 1]);
  }
  return out_shape;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel::detail
