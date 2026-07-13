// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include "onnx_kernels/runtime_context.h"
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

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceProd: axis is out of range.");
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

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ReduceProd: ", name, " must be a FLOAT tensor.");
}

// Multiplies elements of ``data`` into the output buffer. The empty-set
// identity for ``prod`` is ``1`` so the output buffer is initialized to
// ``1.0f`` before the reduction loop.
void ProdReduce(const Tensor &data, const std::vector<bool> &is_reduced,
                const Shape &output_shape_noreduce, Tensor &output) {
  const Shape out_strides = RowMajorStrides(output_shape_noreduce);
  float *py = output.AsFloat();
  const int64_t out_count = output.element_count();
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = 1.0f;
  }

  const float *px = data.AsFloat();
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
    py[out_offset] *= px[i];
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }
}

} // namespace

Tensor ReduceProd::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                              RuntimeContext *rt) const {
  ValidateFloat(data, "data");
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
  const size_t out_n_bytes = static_cast<size_t>(out_count) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  (*this)(data, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceProd::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
                            Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<bool> is_reduced(static_cast<size_t>(rank), false);
  if (!noop_with_empty_axes) {
    std::fill(is_reduced.begin(), is_reduced.end(), true);
  }
  const Shape expected_out_shape = ComputeOutputShape(data.shape, is_reduced, keepdims);
  EXT_ENFORCE_INVALID(
      output.shape == expected_out_shape,
      "kernel::ReduceProd preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceProd preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  ProdReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceProd::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes, RuntimeContext *rt) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceProd: axes must be an INT64 tensor.");
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
  const size_t out_n_bytes = static_cast<size_t>(out_count) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  (*this)(data, axes, keepdims, noop_with_empty_axes, out);
  return out;
}

void ReduceProd::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                            bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceProd: axes must be an INT64 tensor.");
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
      "kernel::ReduceProd preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceProd preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
    return;
  }
  const Shape out_shape_noreduce = ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  ProdReduce(data, is_reduced, out_shape_noreduce, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
