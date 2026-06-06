// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Resolves a possibly-negative axis (ONNX semantics: ``axis`` in
// ``[-rank, rank - 1]``) to a non-negative axis. Throws on out-of-range.
int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, "kernel::ReduceSum: axis is out of range.");
  return resolved;
}

// Computes the output shape of a ReduceSum: dimensions in ``reduce_axes`` are
// either dropped (when ``keepdims`` is false) or replaced by 1.
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

// Row-major strides for ``shape``. Each stride is the number of elements one
// must skip to advance by one along that dimension.
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Sums ``data`` into ``output`` by reducing along the dimensions for which
// ``is_reduced[d]`` is true. ``output`` is laid out with the non-reduced
// dimensions only (i.e. ``keepdims == false``); callers that want the
// keepdims layout reshape the same byte buffer afterwards.
void SumReduce(const Tensor &data, const std::vector<bool> &is_reduced,
               const std::vector<int64_t> &output_shape_noreduce, Tensor &output) {
  const std::vector<int64_t> out_strides = RowMajorStrides(output_shape_noreduce);

  // Zero-initialize the output bytes so we can accumulate into it.
  std::memset(output.data.data(), 0, output.data.size());

  const float *px = data.AsFloat();
  float *py = output.AsFloat();

  // Iterate over every element of the input using a multi-dimensional index.
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<int64_t> idx(static_cast<size_t>(rank), 0);
  const int64_t total = data.element_count();
  for (int64_t i = 0; i < total; ++i) {
    // Compute the output offset by walking through the non-reduced dims.
    int64_t out_offset = 0;
    size_t out_dim = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!is_reduced[static_cast<size_t>(d)]) {
        out_offset += idx[static_cast<size_t>(d)] * out_strides[out_dim];
        ++out_dim;
      }
    }
    py[out_offset] += px[i];

    // Increment the multi-dimensional index (row-major / C order).
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
}

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::ReduceSum: ") + name + " must be a FLOAT tensor.");
}

} // namespace

Tensor ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());

  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    // Reduce over all dimensions.
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }

  const std::vector<int64_t> out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  const int64_t out_count = out_shape.empty() ? 1 : [&out_shape]() {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return n;
  }();
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(float), 0u));
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
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
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::ReduceSum preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    // Identity: copy input bytes verbatim.
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
    return;
  }

  // Layout used by the inner loop is the "no-keepdims" shape (i.e. the list
  // of non-reduced dimensions). When ``keepdims`` is true the output's shape
  // vector contains 1's for the reduced dims but the byte buffer is identical
  // because the element count and row-major layout are preserved.
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                             bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
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

void ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                           bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
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
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::ReduceSum preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
    return;
  }

  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
