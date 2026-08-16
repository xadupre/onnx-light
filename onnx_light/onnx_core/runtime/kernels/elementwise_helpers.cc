// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail {

Shape BroadcastShape(const char *op_name, const Shape &a, const Shape &b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  Shape sa, sb, out;
  sa.assign(rank, 1);
  sb.assign(rank, 1);
  out.assign(rank, 1);
  for (size_t i = 0; i < a.size(); ++i) {
    sa[rank - a.size() + i] = a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    sb[rank - b.size() + i] = b[i];
  }
  for (size_t d = 0; d < rank; ++d) {
    // Explicit branching is used rather than max() so that a size-1 dimension
    // broadcasting against a size-0 dimension yields 0, not 1 -- otherwise
    // the output element count is over-estimated.
    if (sa[d] == sb[d]) {
      out[d] = sa[d];
    } else if (sa[d] == 1) {
      out[d] = sb[d];
    } else if (sb[d] == 1) {
      out[d] = sa[d];
    } else {
      EXT_THROW_INVALID(op_name, " input shapes are not multidirectional-broadcastable.");
    }
  }
  return out;
}

BroadcastInfo CheckBinaryBroadcast(const char *op_name, const char *dtype_name,
                                   int32_t expected_dtype, const Tensor &x, const Tensor &y) {
  EXT_ENFORCE_INVALID(x.data_type == expected_dtype && y.data_type == expected_dtype, op_name,
                      " only supports ", dtype_name, " tensors.");
  return CheckBinaryBroadcastInOut(op_name, dtype_name, expected_dtype, x, y);
}

BroadcastInfo CheckBinaryBroadcastInOut(const char *op_name, const char *in_dtype_name,
                                        int32_t expected_in_dtype, const Tensor &x,
                                        const Tensor &y) {
  EXT_ENFORCE_INVALID(x.data_type == expected_in_dtype && y.data_type == expected_in_dtype, op_name,
                      " only supports ", in_dtype_name, " inputs.");
  // Right-align the input shapes and validate multidirectional-broadcast
  // compatibility per the standard NumPy/ONNX rules: for each pair of aligned
  // dimensions (dx, dy), one of them must be 1 or they must be equal.
  const size_t rank = x.shape.size() > y.shape.size() ? x.shape.size() : y.shape.size();
  Shape sx;
  Shape sy;
  Shape out;
  sx.assign(rank, 1);
  sy.assign(rank, 1);
  out.assign(rank, 1);
  for (size_t i = 0; i < x.shape.size(); ++i) {
    sx[rank - x.shape.size() + i] = x.shape[i];
  }
  for (size_t i = 0; i < y.shape.size(); ++i) {
    sy[rank - y.shape.size() + i] = y.shape[i];
  }
  for (size_t d = 0; d < rank; ++d) {
    // Multidirectional broadcasting: a size-1 dimension takes the size of the
    // other operand. This must follow the broadcast rule literally rather than
    // ``max(dx, dy)`` so that a size-1 dimension broadcasting against a size-0
    // (empty) dimension yields 0, not 1 -- otherwise the output element count is
    // over-estimated and the iteration reads past an empty input buffer.
    if (sx[d] == sy[d]) {
      out[d] = sx[d];
    } else if (sx[d] == 1) {
      out[d] = sy[d];
    } else if (sy[d] == 1) {
      out[d] = sx[d];
    } else {
      EXT_THROW_INVALID(op_name, " input shapes are not multidirectional-broadcastable.");
    }
  }

  BroadcastInfo bi;
  bi.shape = std::move(out);
  bi.shape_x = sx;
  bi.shape_y = sy;
  bi.nx = x.element_count();
  bi.ny = y.element_count();
  bi.element_count = 1;
  for (int64_t d : bi.shape) {
    bi.element_count *= d;
  }
  // Pre-compute per-input element strides aligned to the output rank. A stride
  // of 0 marks a broadcast (size-1) dimension; the underlying buffer is
  // row-major.
  bi.strides_x.assign(rank, 0);
  bi.strides_y.assign(rank, 0);
  int64_t acc_x = 1, acc_y = 1;
  for (size_t i = rank; i-- > 0;) {
    bi.strides_x[i] = sx[i] == 1 ? 0 : acc_x;
    bi.strides_y[i] = sy[i] == 1 ? 0 : acc_y;
    acc_x *= sx[i];
    acc_y *= sy[i];
  }
  return bi;
}

void CheckPreallocatedOutput(const char *op_name, const char *dtype_name, int32_t expected_dtype,
                             const Shape &expected_shape, size_t expected_bytes,
                             const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == expected_dtype, op_name,
                      " preallocated output must be a ", dtype_name, " tensor.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape, op_name,
                      " preallocated output shape must match the broadcasted "
                      "input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes, op_name,
                      " preallocated output buffer has unexpected size in bytes.");
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime::detail
