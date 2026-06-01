// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kMatMulName = "kernel::MatMul";
constexpr const char *kSupportedMatMulTypesMsg =
    " only supports FLOAT, DOUBLE, INT32, INT64, UINT32 and UINT64 inputs.";

int64_t NumElements(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> PromoteMatMulShape(const std::vector<int64_t> &shape, bool is_left) {
  if (shape.size() == 1) {
    if (is_left) {
      return {1, shape[0]};
    }
    return {shape[0], 1};
  }
  return shape;
}

std::vector<int64_t> BroadcastPrefix(const std::vector<int64_t> &a_prefix,
                                     const std::vector<int64_t> &b_prefix) {
  const size_t rank = std::max(a_prefix.size(), b_prefix.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const bool has_a = i + a_prefix.size() >= rank;
    const bool has_b = i + b_prefix.size() >= rank;
    const int64_t da = has_a ? a_prefix[i - (rank - a_prefix.size())] : 1;
    const int64_t db = has_b ? b_prefix[i - (rank - b_prefix.size())] : 1;
    if (da == db || da == 1) {
      out[i] = db;
    } else if (db == 1) {
      out[i] = da;
    } else {
      throw std::invalid_argument(std::string(kMatMulName) +
                                  " inputs are not broadcast-compatible on batch dimensions.");
    }
  }
  return out;
}

std::vector<int64_t> ComputeMatMulOutputShape(const std::vector<int64_t> &a_shape,
                                              const std::vector<int64_t> &b_shape) {
  EXT_ENFORCE_INVALID(!a_shape.empty() && !b_shape.empty(),
                      std::string(kMatMulName) + " does not accept rank-0 inputs.");
  const std::vector<int64_t> a2 = PromoteMatMulShape(a_shape, true);
  const std::vector<int64_t> b2 = PromoteMatMulShape(b_shape, false);
  EXT_ENFORCE_INVALID(a2[a2.size() - 1] == b2[b2.size() - 2],
                      std::string(kMatMulName) + " got incompatible inner dimensions.");

  const std::vector<int64_t> a_prefix(a2.begin(), a2.end() - 2);
  const std::vector<int64_t> b_prefix(b2.begin(), b2.end() - 2);
  std::vector<int64_t> out_shape = BroadcastPrefix(a_prefix, b_prefix);

  if (a_shape.size() != 1) {
    out_shape.push_back(a2[a2.size() - 2]);
  }
  if (b_shape.size() != 1) {
    out_shape.push_back(b2[b2.size() - 1]);
  }
  return out_shape;
}

template <typename T> void MatMulCompute(const Tensor &a, const Tensor &b, Tensor &output) {
  const std::vector<int64_t> a2 = PromoteMatMulShape(a.shape, true);
  const std::vector<int64_t> b2 = PromoteMatMulShape(b.shape, false);
  const int64_t m = a2[a2.size() - 2];
  const int64_t k = a2[a2.size() - 1];
  const int64_t n = b2[b2.size() - 1];
  EXT_ENFORCE_INVALID(k == b2[b2.size() - 2],
                      std::string(kMatMulName) + " got incompatible inner dimensions.");

  const std::vector<int64_t> a_prefix(a2.begin(), a2.end() - 2);
  const std::vector<int64_t> b_prefix(b2.begin(), b2.end() - 2);
  const std::vector<int64_t> out_prefix = BroadcastPrefix(a_prefix, b_prefix);
  const size_t batch_rank = out_prefix.size();

  const std::vector<int64_t> a_strides = ComputeStrides(a2);
  const std::vector<int64_t> b_strides = ComputeStrides(b2);
  const std::vector<int64_t> out_strides = ComputeStrides(output.shape);

  const T *pa = a.As<T>();
  const T *pb = b.As<T>();
  T *py = output.As<T>();

  const int64_t batch_count = NumElements(out_prefix);
  std::vector<int64_t> batch_idx(batch_rank, 0);
  const size_t a_prefix_rank = a_prefix.size();
  const size_t b_prefix_rank = b_prefix.size();

  for (int64_t batch = 0; batch < batch_count; ++batch) {
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t y_base = 0;
    for (size_t d = 0; d < batch_rank; ++d) {
      const int64_t coord = batch_idx[d];
      if (d + a_prefix_rank >= batch_rank) {
        const size_t a_dim = d - (batch_rank - a_prefix_rank);
        const int64_t a_coord = (a_prefix[a_dim] == 1) ? 0 : coord;
        a_base += a_coord * a_strides[a_dim];
      }
      if (d + b_prefix_rank >= batch_rank) {
        const size_t b_dim = d - (batch_rank - b_prefix_rank);
        const int64_t b_coord = (b_prefix[b_dim] == 1) ? 0 : coord;
        b_base += b_coord * b_strides[b_dim];
      }
      y_base += coord * out_strides[d];
    }

    const int64_t a_row_stride = a_strides[a2.size() - 2];
    const int64_t a_k_stride = a_strides[a2.size() - 1];
    const int64_t b_k_stride = b_strides[b2.size() - 2];
    const int64_t b_col_stride = b_strides[b2.size() - 1];

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        T sum = T{0};
        for (int64_t kk = 0; kk < k; ++kk) {
          const T av = pa[a_base + i * a_row_stride + kk * a_k_stride];
          const T bv = pb[b_base + kk * b_k_stride + j * b_col_stride];
          sum += av * bv;
        }
        int64_t y_index = y_base;
        if (a.shape.size() != 1 && b.shape.size() != 1) {
          y_index += i * out_strides[batch_rank] + j * out_strides[batch_rank + 1];
        } else if (a.shape.size() == 1 && b.shape.size() != 1) {
          y_index += j * out_strides[batch_rank];
        } else if (a.shape.size() != 1 && b.shape.size() == 1) {
          y_index += i * out_strides[batch_rank];
        }
        py[y_index] = sum;
      }
    }

    for (size_t d = batch_rank; d-- > 0;) {
      if (++batch_idx[d] < out_prefix[d]) {
        break;
      }
      batch_idx[d] = 0;
    }
  }
}

template <typename T> Tensor MatMulAlloc(const Tensor &a, const Tensor &b) {
  const std::vector<int64_t> out_shape = ComputeMatMulOutputShape(a.shape, b.shape);
  Tensor y("", a.data_type, out_shape,
           std::vector<uint8_t>(static_cast<size_t>(NumElements(out_shape)) * sizeof(T)));
  MatMulCompute<T>(a, b, y);
  return y;
}

template <typename T> void MatMulInPlace(const Tensor &a, const Tensor &b, Tensor &output) {
  const std::vector<int64_t> out_shape = ComputeMatMulOutputShape(a.shape, b.shape);
  EXT_ENFORCE_INVALID(output.data_type == a.data_type,
                      std::string(kMatMulName) +
                          " preallocated output must have the same dtype as input A.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      std::string(kMatMulName) + " preallocated output has an invalid shape.");
  const size_t expected_bytes = static_cast<size_t>(NumElements(out_shape)) * sizeof(T);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      std::string(kMatMulName) +
                          " preallocated output buffer size does not match its shape.");
  MatMulCompute<T>(a, b, output);
}

} // namespace

Tensor MatMul::operator()(const Tensor &a, const Tensor &b) const {
  EXT_ENFORCE_INVALID(a.data_type == b.data_type,
                      std::string(kMatMulName) + " inputs must share the same dtype.");
  switch (a.data_type) {
  case DataType::FLOAT:
    return MatMulAlloc<float>(a, b);
  case DataType::DOUBLE:
    return MatMulAlloc<double>(a, b);
  case DataType::INT32:
    return MatMulAlloc<int32_t>(a, b);
  case DataType::INT64:
    return MatMulAlloc<int64_t>(a, b);
  case DataType::UINT32:
    return MatMulAlloc<uint32_t>(a, b);
  case DataType::UINT64:
    return MatMulAlloc<uint64_t>(a, b);
  default:
    throw std::invalid_argument(std::string(kMatMulName) + kSupportedMatMulTypesMsg);
  }
}

void MatMul::operator()(const Tensor &a, const Tensor &b, Tensor &output) const {
  EXT_ENFORCE_INVALID(a.data_type == b.data_type,
                      std::string(kMatMulName) + " inputs must share the same dtype.");
  switch (a.data_type) {
  case DataType::FLOAT:
    return MatMulInPlace<float>(a, b, output);
  case DataType::DOUBLE:
    return MatMulInPlace<double>(a, b, output);
  case DataType::INT32:
    return MatMulInPlace<int32_t>(a, b, output);
  case DataType::INT64:
    return MatMulInPlace<int64_t>(a, b, output);
  case DataType::UINT32:
    return MatMulInPlace<uint32_t>(a, b, output);
  case DataType::UINT64:
    return MatMulInPlace<uint64_t>(a, b, output);
  default:
    throw std::invalid_argument(std::string(kMatMulName) + kSupportedMatMulTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
