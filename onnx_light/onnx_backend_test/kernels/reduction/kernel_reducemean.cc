// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::ReduceMean: axis is out of range.");
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
                      std::string("kernel::ReduceMean: ") + name + " must be a FLOAT tensor.");
}

// Computes the arithmetic mean of elements of ``data`` along the reduced
// dimensions and writes the result into the output buffer. The empty-set
// identity for ``mean`` is undefined (division by zero); ONNX leaves the
// behaviour unspecified in that case, but no upstream reference test
// exercises it so we simply produce ``0`` values like the other reductions
// do when no elements are aggregated.
void MeanReduce(const Tensor &data, const std::vector<bool> &is_reduced,
                const std::vector<int64_t> &output_shape_noreduce, Tensor &output) {
  const std::vector<int64_t> out_strides = RowMajorStrides(output_shape_noreduce);
  float *py = output.AsFloat();
  const int64_t out_count = output.element_count();
  for (int64_t i = 0; i < out_count; ++i) {
    py[i] = 0.0f;
  }

  // Number of input elements aggregated into each output element: product of
  // the sizes of the reduced dimensions.
  int64_t reduced_count = 1;
  for (size_t d = 0; d < data.shape.size(); ++d) {
    if (is_reduced[d]) {
      reduced_count *= data.shape[d];
    }
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
    py[out_offset] += px[i];
    for (int64_t d = rank - 1; d >= 0; --d) {
      ++idx[static_cast<size_t>(d)];
      if (idx[static_cast<size_t>(d)] < data.shape[static_cast<size_t>(d)]) {
        break;
      }
      idx[static_cast<size_t>(d)] = 0;
    }
  }

  if (reduced_count > 0) {
    const float denom = static_cast<float>(reduced_count);
    for (int64_t i = 0; i < out_count; ++i) {
      py[i] /= denom;
    }
  }
}

} // namespace

Tensor ReduceMean::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes) const {
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

void ReduceMean::operator()(const Tensor &data, bool keepdims, bool noop_with_empty_axes,
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
      "kernel::ReduceMean preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceMean preallocated output buffer has unexpected size in bytes.");

  if (noop_with_empty_axes) {
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MeanReduce(data, is_reduced, out_shape_noreduce, output);
}

Tensor ReduceMean::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                              bool noop_with_empty_axes) const {
  ValidateFloat(data, "data");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMean: axes must be an INT64 tensor.");
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

void ReduceMean::operator()(const Tensor &data, const Tensor &axes, bool keepdims,
                            bool noop_with_empty_axes, Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateFloat(output, "output");
  EXT_ENFORCE_INVALID(axes.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ReduceMean: axes must be an INT64 tensor.");
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
      "kernel::ReduceMean preallocated output shape does not match expected shape.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(out_count) * sizeof(float),
      "kernel::ReduceMean preallocated output buffer has unexpected size in bytes.");

  if (naxes == 0 && noop_with_empty_axes) {
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
    return;
  }
  const std::vector<int64_t> out_shape_noreduce =
      ComputeOutputShape(data.shape, is_reduced, /*keepdims=*/false);
  MeanReduce(data, is_reduced, out_shape_noreduce, output);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
