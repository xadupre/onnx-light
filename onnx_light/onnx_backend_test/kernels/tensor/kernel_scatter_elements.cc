// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

std::vector<int64_t> ReadScatterElementsIndices(const Tensor &indices) {
  const int32_t int64_dt = static_cast<int32_t>(DataType::INT64);
  const int32_t int32_dt = static_cast<int32_t>(DataType::INT32);
  EXT_ENFORCE_INVALID(indices.data_type == int64_dt || indices.data_type == int32_dt,
                      "kernel::ScatterElements: 'indices' input must be INT32 or INT64.");
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

template <typename T>
void ApplyScatterElementsTyped(const Tensor &updates, std::vector<uint8_t> &out_bytes,
                               const std::vector<int64_t> &idx_values, int64_t axis,
                               int64_t axis_dim, const std::vector<int64_t> &data_strides,
                               const std::vector<int64_t> &idx_strides,
                               const std::string &reduction, int64_t r) {
  T *out = reinterpret_cast<T *>(out_bytes.data());
  const T *upd = reinterpret_cast<const T *>(updates.data.data());
  const int64_t total = static_cast<int64_t>(idx_values.size());
  std::vector<int64_t> coord(static_cast<std::size_t>(r), 0);
  for (int64_t u_idx = 0; u_idx < total; ++u_idx) {
    int64_t remaining = u_idx;
    for (int64_t k = 0; k < r; ++k) {
      coord[static_cast<std::size_t>(k)] = remaining / idx_strides[static_cast<std::size_t>(k)];
      remaining %= idx_strides[static_cast<std::size_t>(k)];
    }
    int64_t idx_value = idx_values[static_cast<std::size_t>(u_idx)];
    if (idx_value < 0) {
      idx_value += axis_dim;
    }
    EXT_ENFORCE_INVALID(idx_value >= 0 && idx_value < axis_dim,
                        "kernel::ScatterElements: index out of range.");
    int64_t data_idx = 0;
    for (int64_t k = 0; k < r; ++k) {
      const int64_t c = (k == axis) ? idx_value : coord[static_cast<std::size_t>(k)];
      data_idx += c * data_strides[static_cast<std::size_t>(k)];
    }
    const T v = upd[u_idx];
    T &dst = out[data_idx];
    if (reduction == "none") {
      dst = v;
    } else if (reduction == "add") {
      dst = static_cast<T>(dst + v);
    } else if (reduction == "mul") {
      dst = static_cast<T>(dst * v);
    } else if (reduction == "max") {
      dst = (dst > v) ? dst : v;
    } else if (reduction == "min") {
      dst = (dst < v) ? dst : v;
    } else {
      throw std::invalid_argument("kernel::ScatterElements: unsupported reduction '" + reduction +
                                  "'.");
    }
  }
}

} // namespace

Tensor ScatterElements::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                                   const Attributes &attrs) const {
  Tensor out("", data.data_type, data.shape, data.data);
  (*this)(data, indices, updates, attrs, out);
  return out;
}

void ScatterElements::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                                 const Attributes &attrs, Tensor &output) const {
  const int64_t r = static_cast<int64_t>(data.shape.size());
  EXT_ENFORCE_INVALID(r >= 1, "kernel::ScatterElements: 'data' must have rank >= 1.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(indices.shape.size()) == r,
                      "kernel::ScatterElements: 'data' and 'indices' must have the same rank.");
  EXT_ENFORCE_INVALID(updates.shape == indices.shape,
                      "kernel::ScatterElements: 'updates' must have the same shape as 'indices'.");
  EXT_ENFORCE_INVALID(updates.data_type == data.data_type,
                      "kernel::ScatterElements: 'updates' dtype must match 'data' dtype.");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ScatterElements: output dtype must match data dtype.");
  EXT_ENFORCE_INVALID(output.shape == data.shape,
                      "kernel::ScatterElements: output shape must match data shape.");
  int64_t axis = attrs.axis;
  if (axis < 0) {
    axis += r;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < r, "kernel::ScatterElements: axis out of range.");

  // Initialize output as a copy of data (when caller passes a fresh tensor, it
  // may already share storage with data; only copy when the buffers differ).
  if (output.data.data() != data.data.data()) {
    EXT_ENFORCE_INVALID(output.data.size() == data.data.size(),
                        "kernel::ScatterElements: output buffer size must match data.");
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
  }

  const std::vector<int64_t> idx_values = ReadScatterElementsIndices(indices);
  const int64_t axis_dim = data.shape[static_cast<std::size_t>(axis)];

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

  const int32_t dt = data.data_type;
  if (dt == static_cast<int32_t>(DataType::FLOAT)) {
    ApplyScatterElementsTyped<float>(updates, output.data, idx_values, axis, axis_dim, data_strides,
                                     idx_strides, attrs.reduction, r);
  } else if (dt == static_cast<int32_t>(DataType::DOUBLE)) {
    ApplyScatterElementsTyped<double>(updates, output.data, idx_values, axis, axis_dim,
                                      data_strides, idx_strides, attrs.reduction, r);
  } else if (dt == static_cast<int32_t>(DataType::INT32)) {
    ApplyScatterElementsTyped<int32_t>(updates, output.data, idx_values, axis, axis_dim,
                                       data_strides, idx_strides, attrs.reduction, r);
  } else if (dt == static_cast<int32_t>(DataType::INT64)) {
    ApplyScatterElementsTyped<int64_t>(updates, output.data, idx_values, axis, axis_dim,
                                       data_strides, idx_strides, attrs.reduction, r);
  } else if (dt == static_cast<int32_t>(DataType::UINT8)) {
    ApplyScatterElementsTyped<uint8_t>(updates, output.data, idx_values, axis, axis_dim,
                                       data_strides, idx_strides, attrs.reduction, r);
  } else if (dt == static_cast<int32_t>(DataType::INT8)) {
    ApplyScatterElementsTyped<int8_t>(updates, output.data, idx_values, axis, axis_dim,
                                      data_strides, idx_strides, attrs.reduction, r);
  } else {
    // Fall back to byte-wise copy when reduction == "none".
    EXT_ENFORCE_INVALID(attrs.reduction == "none",
                        "kernel::ScatterElements: reductions other than 'none' are not "
                        "supported for this dtype.");
    const std::size_t elem_size = ElementSize(data.data_type);
    const int64_t total = static_cast<int64_t>(idx_values.size());
    std::vector<int64_t> coord(static_cast<std::size_t>(r), 0);
    for (int64_t u_idx = 0; u_idx < total; ++u_idx) {
      int64_t remaining = u_idx;
      for (int64_t k = 0; k < r; ++k) {
        coord[static_cast<std::size_t>(k)] = remaining / idx_strides[static_cast<std::size_t>(k)];
        remaining %= idx_strides[static_cast<std::size_t>(k)];
      }
      int64_t idx_value = idx_values[static_cast<std::size_t>(u_idx)];
      if (idx_value < 0) {
        idx_value += axis_dim;
      }
      EXT_ENFORCE_INVALID(idx_value >= 0 && idx_value < axis_dim,
                          "kernel::ScatterElements: index out of range.");
      int64_t data_idx = 0;
      for (int64_t k = 0; k < r; ++k) {
        const int64_t c = (k == axis) ? idx_value : coord[static_cast<std::size_t>(k)];
        data_idx += c * data_strides[static_cast<std::size_t>(k)];
      }
      std::memcpy(output.data.data() + static_cast<std::size_t>(data_idx) * elem_size,
                  updates.data.data() + static_cast<std::size_t>(u_idx) * elem_size, elem_size);
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
