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

template <typename T>
void ReduceSliceTyped(uint8_t *out_bytes, const uint8_t *upd_bytes, std::size_t n_elements,
                      const std::string &reduction) {
  T *out = reinterpret_cast<T *>(out_bytes);
  const T *upd = reinterpret_cast<const T *>(upd_bytes);
  if (reduction == "none") {
    std::memcpy(out_bytes, upd_bytes, n_elements * sizeof(T));
    return;
  }
  for (std::size_t i = 0; i < n_elements; ++i) {
    const T v = upd[i];
    T &dst = out[i];
    if (reduction == "add") {
      dst = static_cast<T>(dst + v);
    } else if (reduction == "mul") {
      dst = static_cast<T>(dst * v);
    } else if (reduction == "max") {
      dst = (dst > v) ? dst : v;
    } else if (reduction == "min") {
      dst = (dst < v) ? dst : v;
    } else {
      throw std::invalid_argument("kernel::ScatterND: unsupported reduction '" + reduction + "'.");
    }
  }
}

void ReduceSlice(int32_t data_type, uint8_t *out_bytes, const uint8_t *upd_bytes,
                 std::size_t n_elements, const std::string &reduction) {
  if (reduction == "none") {
    std::memcpy(out_bytes, upd_bytes, n_elements * ElementSize(data_type));
    return;
  }
  if (data_type == static_cast<int32_t>(DataType::FLOAT)) {
    ReduceSliceTyped<float>(out_bytes, upd_bytes, n_elements, reduction);
  } else if (data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    ReduceSliceTyped<double>(out_bytes, upd_bytes, n_elements, reduction);
  } else if (data_type == static_cast<int32_t>(DataType::INT32)) {
    ReduceSliceTyped<int32_t>(out_bytes, upd_bytes, n_elements, reduction);
  } else if (data_type == static_cast<int32_t>(DataType::INT64)) {
    ReduceSliceTyped<int64_t>(out_bytes, upd_bytes, n_elements, reduction);
  } else if (data_type == static_cast<int32_t>(DataType::UINT8)) {
    ReduceSliceTyped<uint8_t>(out_bytes, upd_bytes, n_elements, reduction);
  } else if (data_type == static_cast<int32_t>(DataType::INT8)) {
    ReduceSliceTyped<int8_t>(out_bytes, upd_bytes, n_elements, reduction);
  } else {
    throw std::invalid_argument(
        "kernel::ScatterND: unsupported dtype for reduction other than 'none'.");
  }
}

} // namespace

Tensor ScatterND::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                             const Attributes &attrs) const {
  Tensor out("", data.data_type, data.shape, data.data);
  (*this)(data, indices, updates, attrs, out);
  return out;
}

void ScatterND::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                           const Attributes &attrs, Tensor &output) const {
  EXT_ENFORCE_INVALID(indices.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ScatterND: 'indices' input must be INT64.");
  EXT_ENFORCE_INVALID(updates.data_type == data.data_type,
                      "kernel::ScatterND: 'updates' dtype must match 'data' dtype.");
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::ScatterND: output dtype must match data dtype.");
  EXT_ENFORCE_INVALID(output.shape == data.shape,
                      "kernel::ScatterND: output shape must match data shape.");

  const int64_t r = static_cast<int64_t>(data.shape.size());
  EXT_ENFORCE_INVALID(r >= 1, "kernel::ScatterND: 'data' must have rank >= 1.");
  const int64_t q = static_cast<int64_t>(indices.shape.size());
  EXT_ENFORCE_INVALID(q >= 1, "kernel::ScatterND: 'indices' must have rank >= 1.");
  const int64_t k = indices.shape[static_cast<std::size_t>(q - 1)];
  EXT_ENFORCE_INVALID(k >= 1 && k <= r,
                      "kernel::ScatterND: last dim of 'indices' must be in [1, rank(data)].");

  // Expected updates shape = indices.shape[:-1] + data.shape[k:]
  std::vector<int64_t> expected_updates_shape(
      indices.shape.begin(), indices.shape.begin() + static_cast<std::ptrdiff_t>(q - 1));
  for (std::size_t i = static_cast<std::size_t>(k); i < data.shape.size(); ++i) {
    expected_updates_shape.push_back(data.shape[i]);
  }
  EXT_ENFORCE_INVALID(updates.shape == expected_updates_shape,
                      "kernel::ScatterND: 'updates' shape must be indices.shape[:-1] + "
                      "data.shape[indices.shape[-1]:].");

  // Initialize output with a copy of data.
  if (output.data.data() != data.data.data()) {
    EXT_ENFORCE_INVALID(output.data.size() == data.data.size(),
                        "kernel::ScatterND: output buffer size must match data.");
    std::memcpy(output.data.data(), data.data.data(), data.data.size());
  }

  // Number of index tuples = prod(indices.shape[:-1]).
  int64_t n_tuples = 1;
  for (int64_t i = 0; i < q - 1; ++i) {
    n_tuples *= indices.shape[static_cast<std::size_t>(i)];
  }

  // Number of elements per slice = prod(data.shape[k:]).
  int64_t slice_elems = 1;
  for (int64_t i = k; i < r; ++i) {
    slice_elems *= data.shape[static_cast<std::size_t>(i)];
  }

  // Row-major strides for data (in elements).
  std::vector<int64_t> data_strides(static_cast<std::size_t>(r), 1);
  for (int64_t i = r - 2; i >= 0; --i) {
    data_strides[static_cast<std::size_t>(i)] =
        data_strides[static_cast<std::size_t>(i + 1)] * data.shape[static_cast<std::size_t>(i + 1)];
  }

  const std::size_t elem_size = ElementSize(data.data_type);
  const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(indices.data.data());

  for (int64_t t = 0; t < n_tuples; ++t) {
    int64_t data_offset = 0;
    for (int64_t j = 0; j < k; ++j) {
      int64_t v = idx_ptr[t * k + j];
      const int64_t dim = data.shape[static_cast<std::size_t>(j)];
      if (v < 0) {
        v += dim;
      }
      EXT_ENFORCE_INVALID(v >= 0 && v < dim, "kernel::ScatterND: index out of range.");
      data_offset += v * data_strides[static_cast<std::size_t>(j)];
    }
    uint8_t *dst = output.data.data() + static_cast<std::size_t>(data_offset) * elem_size;
    const uint8_t *src =
        updates.data.data() + static_cast<std::size_t>(t * slice_elems) * elem_size;
    ReduceSlice(data.data_type, dst, src, static_cast<std::size_t>(slice_elems), attrs.reduction);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
