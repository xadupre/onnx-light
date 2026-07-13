// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_kernels/runtime_context.h"
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

Shape ComputeOutputShape(const Shape &input_shape, const std::vector<bool> &is_reduced,
                         bool keepdims) {
  Shape out_shape;
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

Shape RowMajorStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

void ValidateFloatOrDouble(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          t.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::ReduceLogSumOp: ", name, " must be a FLOAT or DOUBLE tensor.");
}

// Templated reduction core for both FLOAT and DOUBLE.
template <typename T>
void LogSumReduceT(const T *px, T *py, int64_t out_count, int64_t total, int64_t rank,
                   const Shape &data_shape, const std::vector<bool> &is_reduced,
                   const Shape &output_shape_noreduce, ReduceLogSumOp::Mode mode) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);

  int64_t reduced_count = 1;
  for (size_t d = 0; d < data_shape.size(); ++d) {
    if (is_reduced[d]) {
      reduced_count *= data_shape[d];
    }
  }
  if (reduced_count == 0) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = neg_inf;
    }
    return;
  }

  if (mode == ReduceLogSumOp::Mode::kLogSum) {
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = static_cast<T>(0);
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
        if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
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
  const T neg_inf = -std::numeric_limits<T>::infinity();
  std::vector<T> max_vals(static_cast<size_t>(out_count), neg_inf);
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
    const T v = px[i];
    if (v > max_vals[static_cast<size_t>(out_offset)]) {
      max_vals[static_cast<size_t>(out_offset)] = v;
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = static_cast<T>(0);
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
    const T m = max_vals[static_cast<size_t>(out_offset)];
    if (std::isfinite(m)) {
      py[out_offset] += std::exp(px[i] - m);
    }
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data_shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
  for (int64_t i = 0; i < out_count; ++i) {
    const T m = max_vals[static_cast<size_t>(i)];
    if (!std::isfinite(m)) {
      py[i] = m;
    } else {
      py[i] = m + std::log(py[i]);
    }
  }
}

void LogSumReduce(const Tensor &data, const std::vector<bool> &is_reduced,
                  const Shape &output_shape_noreduce, ReduceLogSumOp::Mode mode, Tensor &output) {
  const int64_t out_count = output.element_count();
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const int64_t total = data.element_count();
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    LogSumReduceT<double>(data.AsDouble(), output.AsDouble(), out_count, total, rank, data.shape,
                          is_reduced, output_shape_noreduce, mode);
  } else {
    LogSumReduceT<float>(data.AsFloat(), output.AsFloat(), out_count, total, rank, data.shape,
                         is_reduced, output_shape_noreduce, mode);
  }
}

template <typename T>
void LogSumNoopElementwiseT(const T *px, T *py, int64_t total, ReduceLogSumOp::Mode mode) {
  switch (mode) {
  case ReduceLogSumOp::Mode::kLogSum:
    for (int64_t i = 0; i < total; ++i) {
      py[i] = std::log(px[i]);
    }
    break;
  case ReduceLogSumOp::Mode::kLogSumExp:
    // identity: log(exp(x)) == x
    break;
  }
}

void LogSumNoopElementwise(const Tensor &data, ReduceLogSumOp::Mode mode, Tensor &output) {
  const int64_t total = data.element_count();
  if (mode == ReduceLogSumOp::Mode::kLogSumExp) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    LogSumNoopElementwiseT<double>(data.AsDouble(), output.AsDouble(), total, mode);
  } else {
    LogSumNoopElementwiseT<float>(data.AsFloat(), output.AsFloat(), total, mode);
  }
}

} // namespace

Tensor ReduceLogSumOp::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                  RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out =
      MakeOutputTensor(data.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                                Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ReduceLogSumOp: output dtype must match data dtype.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

Tensor ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                  bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
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
  const Shape out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(out_count) * elem_size;
  Tensor out =
      MakeOutputTensor(data.data_type, out_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceLogSumOp::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                                bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ReduceLogSumOp: output dtype must match data dtype.");
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
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceLogSumOp preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  const size_t elem_size =
      (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) ? sizeof(double) : sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
      "kernel::ReduceLogSumOp preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    LogSumNoopElementwise(data, mode_, output);
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  LogSumReduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
