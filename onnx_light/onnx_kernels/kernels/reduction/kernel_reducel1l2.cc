// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
                      "kernel::ReduceL1L2: axis is out of range.");
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
                      std::string("kernel::ReduceL1L2: ") + name + " must be a FLOAT tensor.");
}

// Accumulates either |x| (L1) or x*x (L2) into the output buffer, then for L2
// applies sqrt(.) to every element. The output layout used by the inner loop
// is the "no-keepdims" shape; callers that pass a ``keepdims`` output share
// the same byte buffer because element count and row-major layout match.
void L1L2Reduce(const Tensor &data, const std::vector<bool> &is_reduced,
                const std::vector<int64_t> &output_shape_noreduce, ReduceL1L2::Mode mode,
                Tensor &output) {
  const std::vector<int64_t> out_strides = RowMajorStrides(output_shape_noreduce);
  std::memset(output.data.data(), 0, output.data.size());

  const float *px = data.AsFloat();
  float *py = output.AsFloat();
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
  const int64_t total = data.element_count();
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
    py[out_offset] += mode == ReduceL1L2::Mode::kL1 ? std::fabs(v) : v * v;

    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  if (mode == ReduceL1L2::Mode::kL2) {
    const int64_t out_count = output.element_count();
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] = std::sqrt(py[i]);
    }
  }
}

// Applies the per-element transform implied by ``mode`` without performing any
// reduction. ONNX's ``noop_with_empty_axes`` semantics for these reductions
// still apply the element-wise function (``|x|`` for L1, ``x*x`` for
// SumSquare, ``sqrt(x*x) == |x|`` for L2); only the summation across axes is
// skipped. This matches the behaviour of the ONNX reference implementation
// and onnxruntime.
void L1L2NoopElementwise(const Tensor &data, ReduceL1L2::Mode mode, Tensor &output) {
  const float *px = data.AsFloat();
  float *py = output.AsFloat();
  const int64_t total = data.element_count();
  for (int64_t i = 0; i < total; ++i) {
    const float v = px[i];
    switch (mode) {
    case ReduceL1L2::Mode::kL1:
      py[i] = std::fabs(v);
      break;
    case ReduceL1L2::Mode::kSumSquare:
      py[i] = v * v;
      break;
    case ReduceL1L2::Mode::kL2:
      py[i] = std::fabs(v);
      break;
    }
  }
}

} // namespace

Tensor ReduceL1L2::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes) const {
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

void ReduceL1L2::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
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
      "kernel::ReduceL1L2 preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceL1L2 preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    L1L2NoopElementwise(data, mode_, output);
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  L1L2Reduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

Tensor ReduceL1L2::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceL1L2: axes must be an INT64 tensor.");
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

void ReduceL1L2::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                            bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceL1L2: axes must be an INT64 tensor.");
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
      "kernel::ReduceL1L2 preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceL1L2 preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    L1L2NoopElementwise(data, mode_, output);
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  L1L2Reduce(data, is_reduced, out_shape_noreduce, mode_, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
