// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

std::vector<int64_t> ReadGatherElementsIndices(const Tensor &indices) {
  const int32_t int64_dt = static_cast<int32_t>(DataType::INT64);
  const int32_t int32_dt = static_cast<int32_t>(DataType::INT32);
  EXT_ENFORCE_INVALID(indices.data_type == int64_dt || indices.data_type == int32_dt,
                      "kernel::GatherElements: 'indices' input must be INT32 or INT64.");
  int64_t n = indices.element_count();
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n == 0) {
    return out;
  }
  if (indices.data_type == int64_dt) {
    std::memcpy(out.data(), indices.data.data(), static_cast<std::size_t>(n) * sizeof(int64_t));
  } else {
    const int32_t *p = reinterpret_cast<const int32_t *>(indices.data.data());
    for (int64_t i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] = static_cast<int64_t>(p[i]);
    }
  }
  return out;
}

} // namespace

Tensor GatherElements::operator()(const Tensor &data, const Tensor &indices, int64_t axis) const {
  const std::size_t elem_size = ElementSize(data.data_type);
  int64_t total = indices.element_count();
  Tensor out("", data.data_type, indices.shape,
             std::vector<uint8_t>(static_cast<std::size_t>(total) * elem_size));
  (*this)(data, indices, axis, out);
  return out;
}

void GatherElements::operator()(const Tensor &data, const Tensor &indices, int64_t axis,
                                Tensor &output) const {
  const int64_t r = static_cast<int64_t>(data.shape.size());
  EXT_ENFORCE_INVALID(r >= 1, "kernel::GatherElements: 'data' must have rank >= 1.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(indices.shape.size()) == r,
                      "kernel::GatherElements: 'data' and 'indices' must have the same rank.");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::GatherElements: output dtype must match data dtype.");
  EXT_ENFORCE_INVALID(output.shape == indices.shape,
                      "kernel::GatherElements: output shape must match indices shape.");
  if (axis < 0) {
    axis += r;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < r, "kernel::GatherElements: axis out of range.");

  const std::vector<int64_t> idx_values = ReadGatherElementsIndices(indices);
  const int64_t axis_dim = data.shape[static_cast<std::size_t>(axis)];
  const std::size_t elem_size = ElementSize(data.data_type);

  // Row-major strides for data and indices (in elements).
  std::vector<int64_t> data_strides(static_cast<std::size_t>(r), 1);
  for (int64_t k = r - 2; k >= 0; --k) {
    data_strides[static_cast<std::size_t>(k)] =
        data_strides[static_cast<std::size_t>(k + 1)] * data.shape[static_cast<std::size_t>(k + 1)];
  }
  std::vector<int64_t> idx_strides(static_cast<std::size_t>(r), 1);
  for (int64_t k = r - 2; k >= 0; --k) {
    idx_strides[static_cast<std::size_t>(k)] = idx_strides[static_cast<std::size_t>(k + 1)] *
                                               indices.shape[static_cast<std::size_t>(k + 1)];
  }

  const int64_t total = static_cast<int64_t>(idx_values.size());
  std::vector<int64_t> coord(static_cast<std::size_t>(r), 0);
  for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
    // Decode coord from out_idx using indices' (== output's) strides.
    int64_t remaining = out_idx;
    for (int64_t k = 0; k < r; ++k) {
      coord[static_cast<std::size_t>(k)] = remaining / idx_strides[static_cast<std::size_t>(k)];
      remaining %= idx_strides[static_cast<std::size_t>(k)];
    }
    int64_t idx_value = idx_values[static_cast<std::size_t>(out_idx)];
    if (idx_value < 0) {
      idx_value += axis_dim;
    }
    EXT_ENFORCE_INVALID(idx_value >= 0 && idx_value < axis_dim,
                        "kernel::GatherElements: index out of range.");
    int64_t data_idx = 0;
    for (int64_t k = 0; k < r; ++k) {
      const int64_t c = (k == axis) ? idx_value : coord[static_cast<std::size_t>(k)];
      data_idx += c * data_strides[static_cast<std::size_t>(k)];
    }
    std::memcpy(output.data.data() + static_cast<std::size_t>(out_idx) * elem_size,
                data.data.data() + static_cast<std::size_t>(data_idx) * elem_size, elem_size);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
