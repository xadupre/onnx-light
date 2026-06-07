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

std::vector<int64_t> ReadIndicesAsInt64(const Tensor &indices, const std::string &op_name) {
  const int32_t int64_dt = static_cast<int32_t>(DataType::INT64);
  const int32_t int32_dt = static_cast<int32_t>(DataType::INT32);
  EXT_ENFORCE_INVALID(indices.data_type == int64_dt || indices.data_type == int32_dt,
                      "kernel::" + op_name + ": 'indices' input must be INT32 or INT64.");
  int64_t n = indices.element_count();
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n == 0) {
    return out;
  }
  if (indices.data_type == int64_dt) {
    std::memcpy(out.data(), indices.bytes(), static_cast<std::size_t>(n) * sizeof(int64_t));
  } else {
    const int32_t *p = reinterpret_cast<const int32_t *>(indices.bytes());
    for (int64_t i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] = static_cast<int64_t>(p[i]);
    }
  }
  return out;
}

int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::Gather: 'data' must have rank >= 1.");
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank,
                      "kernel::Gather: 'axis' out of range for 'data' rank.");
  return axis;
}

} // namespace

Tensor Gather::operator()(const Tensor &data, const Tensor &indices, int64_t axis) const {
  const int64_t r = static_cast<int64_t>(data.shape.size());
  const int64_t a = NormalizeAxis(axis, r);
  std::vector<int64_t> out_shape;
  out_shape.reserve(data.shape.size() + indices.shape.size() - 1);
  for (int64_t k = 0; k < a; ++k) {
    out_shape.push_back(data.shape[static_cast<std::size_t>(k)]);
  }
  for (int64_t d : indices.shape) {
    out_shape.push_back(d);
  }
  for (int64_t k = a + 1; k < r; ++k) {
    out_shape.push_back(data.shape[static_cast<std::size_t>(k)]);
  }
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  const std::size_t elem_size = ElementSize(data.data_type);
  Tensor out("", data.data_type, out_shape,
             std::vector<uint8_t>(static_cast<std::size_t>(total) * elem_size));
  (*this)(data, indices, axis, out);
  return out;
}

void Gather::operator()(const Tensor &data, const Tensor &indices, int64_t axis,
                        Tensor &output) const {
  const int64_t r = static_cast<int64_t>(data.shape.size());
  const int64_t a = NormalizeAxis(axis, r);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Gather: preallocated output dtype must match data dtype.");

  const std::vector<int64_t> idx_values = ReadIndicesAsInt64(indices, "Gather");
  const int64_t axis_dim = data.shape[static_cast<std::size_t>(a)];

  // Build per-element strides (in bytes) for data and output.
  const std::size_t elem_size = ElementSize(data.data_type);

  // Number of elements per "outer" slice (dims before axis) and "inner" slice
  // (dims after axis) of data, plus the indexed-axis dimension.
  int64_t outer = 1;
  for (int64_t k = 0; k < a; ++k) {
    outer *= data.shape[static_cast<std::size_t>(k)];
  }
  int64_t inner = 1;
  for (int64_t k = a + 1; k < r; ++k) {
    inner *= data.shape[static_cast<std::size_t>(k)];
  }
  const int64_t q_count = static_cast<int64_t>(idx_values.size());
  const int64_t inner_bytes = inner * static_cast<int64_t>(elem_size);
  const int64_t data_axis_stride_bytes = inner_bytes * axis_dim;
  const int64_t out_outer_bytes = inner_bytes * q_count;

  for (int64_t o = 0; o < outer; ++o) {
    const uint8_t *data_outer = data.bytes() + static_cast<std::size_t>(o * data_axis_stride_bytes);
    uint8_t *out_outer = output.data.data() + static_cast<std::size_t>(o * out_outer_bytes);
    for (int64_t qi = 0; qi < q_count; ++qi) {
      int64_t idx = idx_values[static_cast<std::size_t>(qi)];
      if (idx < 0) {
        idx += axis_dim;
      }
      EXT_ENFORCE_INVALID(idx >= 0 && idx < axis_dim,
                          "kernel::Gather: index out of range for axis dim.");
      std::memcpy(out_outer + static_cast<std::size_t>(qi * inner_bytes),
                  data_outer + static_cast<std::size_t>(idx * inner_bytes),
                  static_cast<std::size_t>(inner_bytes));
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
