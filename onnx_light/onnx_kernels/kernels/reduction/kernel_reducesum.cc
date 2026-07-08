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
#include "onnx_kernels/runtime_context.h"

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
template <typename T>
void SumReduceT(const Tensor &data, const std::vector<bool> &is_reduced,
                const std::vector<int64_t> &output_shape_noreduce, Tensor &output) {
  const std::vector<int64_t> out_strides = RowMajorStrides(output_shape_noreduce);

  // Zero-initialize the output bytes so we can accumulate into it.
  std::memset(output.mutable_bytes(), 0, output.size_bytes());

  const T *px = reinterpret_cast<const T *>(data.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());

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

void SumReduce(const Tensor &data, const std::vector<bool> &is_reduced,
               const std::vector<int64_t> &output_shape_noreduce, Tensor &output) {
  if (data.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    SumReduceT<double>(data, is_reduced, output_shape_noreduce, output);
  } else {
    SumReduceT<float>(data, is_reduced, output_shape_noreduce, output);
  }
}

void ValidateFloatOrDouble(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          t.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::ReduceSum: ", name, " must be a FLOAT or DOUBLE tensor.");
}

} // namespace

Tensor ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

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
  Tensor out("", data.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * elem_size, 0u));
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceSum::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                           Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  ValidateFloatOrDouble(output, "output");
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);
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
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }

  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                             bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloatOrDouble(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

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
  Tensor out("", data.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * elem_size, 0u));
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceSum::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                           bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloatOrDouble(data, "data");
  ValidateFloatOrDouble(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceSum: axes must be an INT64 tensor.");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  const size_t elem_size =
      data.data_type == static_cast<int32_t>(DataType::DOUBLE) ? sizeof(double) : sizeof(float);

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
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(out_count) * elem_size,
                      "kernel::ReduceSum preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }

  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  SumReduce(data, is_reduced, out_shape_noreduce, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
