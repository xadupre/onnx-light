// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceLogSumOp: axis is out of range.");
  return resolved;
}

std::vector<int64_t> ComputeOutputShape(const std::vector<int64_t> &input_shape,
                                        const std::vector<bool> &is_reduced, bool keepdims) {
  std::vector<int64_t> out_shape;
  out_shape.reserve(input_shape.size());
  for (size_t d = 0; d < input_shape.size(); ++d) {
    if (is_reduced[d]) {
      if (keepdims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input_shape[d]);
    }
  }
  return out_shape;
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::ReduceLogSumOp: ") + name + " must be a FLOAT tensor.");
}

// Computes ``y = log(sum(x, axes))`` (kLogSum) or the numerically-stable
// ``y = m + log(sum(exp(x - m), axes))`` where ``m = max(x, axes)``
// (kLogSumExp). The output layout used by the inner loop is the
// "no-keepdims" shape; callers that pass a ``keepdims`` output share the
// same byte buffer because element count and row-major layout match. When
// no element contributes to an output cell (the reduced extent is 0), the
// result is ``-inf`` (the ONNX empty-set identity for both ops).
void LogSumReduce(const Tensor &data, const std::vector<bool> &is_reduced,
                  const std::vector<int64_t> &output_shape_noreduce, ReduceLogSumOp::Mode mode,
                  Tensor &output) {
  const std::vector<int64_t> out_strides = RowMajorStrides(output_shape_noreduce);
  float *py = output.AsFloat();
  const int64_t out_count = output.element_count();

  // Number of input elements that aggregate into each output element: the
  // product of the sizes of the reduced dimensions. When this is 0 the
  // output is the empty-set identity (-inf) for every cell.
  int64_t reduced_count = 1;
  for (size_t d = 0; d < data.shape.size(); ++d) {
    if (is_reduced[d]) {
      reduced_count *= data.shape[d];
    }
  }
  if (reduced_count == 0) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = neg_inf;
    }
    return;
  }

  const float *px = data.AsFloat();
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const int64_t total = data.element_count();

  if (mode == ReduceLogSumOp::Mode::kLogSum) {
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = 0.0f;
    }
    std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
    for (int64_t i = 0; i < total; ++i) {
      int64_t out_offset = 0;
      size_t out_dim = 0;
      for (int64_t d = 0; d < rank; ++d) {
        if (!is_reduced[static_cast<size_t>(d)]) {
          out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
          ++out_dim;
        }
      }
      py[out_offset] += px[i];
      for (int64_t d = rank - 1; d >= 0; --d) {
        ++idx[static_cast<size_t>(d)];
        if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
          break;
        }
        idx[static_cast<size_t>(d)] = 0;
      }
    }
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = std::log(py[i]);
    }
    return;
  }

  // kLogSumExp: use the max-shift trick.
  // First pass: compute the per-output maximum (initialize to -inf).
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> max_vals(static_cast<size_t>(out_count), neg_inf);
  std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
  for (int64_t i = 0; i < total; ++i) {
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    const float v = px[i];
    if (v > max_vals[static_cast<size_t>(out_offset)]) {
      max_vals[static_cast<size_t>(out_offset)] = v;
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  // Second pass: accumulate sum(exp(x - m)).
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = 0.0f;
  }
  std::fill(idx.begin(), idx.end(), 0);
  for (int64_t i = 0; i < total; ++i) {
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    const float m = max_vals[static_cast<size_t>(out_offset)];
    // When m == -inf every input is -inf as well; treat exp(-inf - -inf) as 0
    // and let the post-processing produce m == -inf in the output.
    if (std::isfinite(m)) {
      py[out_offset] += std::exp(px[i] - m);
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
  for (int64_t i = 0; i < out_count; ++i) {
    const float m = max_vals[static_cast<size_t>(i)];
    if (!std::isfinite(m)) {
      py[i] = m;
    } else {
      py[i] = m + std::log(py[i]);
    }
  }
}

// Applies the per-element transform implied by ``mode`` without performing
// any reduction. ONNX's ``noop_with_empty_axes`` semantics still apply the
// element-wise function: ``log(x)`` for LogSum, ``log(exp(x)) == x`` for
// LogSumExp (i.e. identity). Only the summation across axes is skipped.
void LogSumNoopElementwise(const Tensor &data, ReduceLogSumOp::Mode mode, Tensor &output) {
  const float *px = data.AsFloat();
  float *py = output.AsFloat();
  const int64_t total = data.element_count();
  switch (mode) {
  case ReduceLogSumOp::Mode::kLogSum:
    for (int64_t i = 0; i < total; ++i) {
      py[i] = std::log(px[i]);
    }
    break;
  case ReduceLogSumOp::Mode::kLogSumExp:
    std::memcpy(output.data.data(), data.bytes(), data.size_bytes());
    break;
  }
}

} // namespace

Tensor ReduceLogSumOp::operator()(const Tensor &data, bool keepdims,
                                  bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const std::vector<int64_t> out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float), 0u));
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const std::vector<int64_t> expected_out_shape =
      ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

Tensor ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                  bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceLogSumOp: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), true);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = true;
    }
  }
  const std::vector<int64_t> out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float), 0u));
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceLogSumOp: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());

  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  const int64_t naxes = axes.element_count();
  if (naxes == 0) {
    if (!noop_with_empty_axes) {
      std::fill(is_reduced.begin(), is_reduced.end(), true);
    }
  } else {
    const int64_t *pa = axes.AsInt64();
    for (int64_t i = 0; i < naxes; ++i) {
      const int64_t a = ResolveAxis(pa[i], rank);
      is_reduced[static_cast<size_t>(a)] = true;
    }
  }
  const std::vector<int64_t> expected_out_shape =
      ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
