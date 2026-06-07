// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

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

constexpr const char *kName = "kernel::MatMulInteger";

inline bool IsInt8OrUint8(int32_t dt) {
  return dt == static_cast<int32_t>(DataType::INT8) || dt == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadIntElem(const Tensor &t, int64_t idx) {
  if (t.data_type == static_cast<int32_t>(DataType::INT8)) {
    return static_cast<int32_t>(t.AsInt8()[idx]);
  }
  return static_cast<int32_t>(t.AsUint8()[idx]);
}

// Returns 0 for default-constructed (empty) optional tensor. Otherwise the
// tensor must hold exactly one element (scalar shape, or 1-D shape of size 1)
// matching the dtype of the corresponding data tensor.
int32_t ReadOptionalScalarZP(const Tensor &t, int32_t expected_dtype, const char *name) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    return 0;
  }
  EXT_ENFORCE_INVALID(t.data_type == expected_dtype,
                      std::string(kName) + ": '" + name + "' dtype must match its data input.");
  int64_t numel = 1;
  for (int64_t d : t.shape) {
    numel *= d;
  }
  EXT_ENFORCE_INVALID(numel == 1,
                      std::string(kName) + ": '" + name +
                          "' must be a scalar or a one-element 1-D tensor in this reference "
                          "implementation.");
  return ReadIntElem(t, 0);
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

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
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
      throw std::invalid_argument(std::string(kName) +
                                  ": inputs are not broadcast-compatible on batch dimensions.");
    }
  }
  return out;
}

std::vector<int64_t> ComputeOutputShape(const std::vector<int64_t> &a_shape,
                                        const std::vector<int64_t> &b_shape) {
  EXT_ENFORCE_INVALID(!a_shape.empty() && !b_shape.empty(),
                      std::string(kName) + ": rank-0 inputs are not accepted.");
  const std::vector<int64_t> a2 = PromoteMatMulShape(a_shape, true);
  const std::vector<int64_t> b2 = PromoteMatMulShape(b_shape, false);
  EXT_ENFORCE_INVALID(a2[a2.size() - 1] == b2[b2.size() - 2],
                      std::string(kName) + ": incompatible inner dimensions.");
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

void RunMatMulInteger(const Tensor &a, int32_t a_zp, const Tensor &b, int32_t b_zp,
                      Tensor &output) {
  const std::vector<int64_t> a2 = PromoteMatMulShape(a.shape, true);
  const std::vector<int64_t> b2 = PromoteMatMulShape(b.shape, false);
  const int64_t M = a2[a2.size() - 2];
  const int64_t K = a2[a2.size() - 1];
  const int64_t N = b2[b2.size() - 1];

  const std::vector<int64_t> a_prefix(a2.begin(), a2.end() - 2);
  const std::vector<int64_t> b_prefix(b2.begin(), b2.end() - 2);
  const std::vector<int64_t> out_prefix = BroadcastPrefix(a_prefix, b_prefix);
  const size_t batch_rank = out_prefix.size();

  const std::vector<int64_t> a_strides = ComputeStrides(a2);
  const std::vector<int64_t> b_strides = ComputeStrides(b2);
  const std::vector<int64_t> out_strides = ComputeStrides(output.shape);

  const size_t a_prefix_rank = a_prefix.size();
  const size_t b_prefix_rank = b_prefix.size();

  int64_t batch_count = 1;
  for (int64_t d : out_prefix) {
    batch_count *= d;
  }

  int32_t *py = output.AsInt32();

  std::vector<int64_t> batch_idx(batch_rank, 0);
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

    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        int32_t acc = 0;
        for (int64_t kk = 0; kk < K; ++kk) {
          const int32_t av = ReadIntElem(a, a_base + i * a_row_stride + kk * a_k_stride);
          const int32_t bv = ReadIntElem(b, b_base + kk * b_k_stride + j * b_col_stride);
          acc += (av - a_zp) * (bv - b_zp);
        }
        int64_t y_index = y_base;
        if (a.shape.size() != 1 && b.shape.size() != 1) {
          y_index += i * out_strides[batch_rank] + j * out_strides[batch_rank + 1];
        } else if (a.shape.size() == 1 && b.shape.size() != 1) {
          y_index += j * out_strides[batch_rank];
        } else if (a.shape.size() != 1 && b.shape.size() == 1) {
          y_index += i * out_strides[batch_rank];
        }
        py[y_index] = acc;
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

} // namespace

Tensor MatMulInteger::operator()(const Tensor &a, const Tensor &b, const Tensor &a_zero_point,
                                 const Tensor &b_zero_point) const {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type),
                      std::string(kName) + ": A must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type),
                      std::string(kName) + ": B must be INT8 or UINT8.");

  const std::vector<int64_t> out_shape = ComputeOutputShape(a.shape, b.shape);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor out("", static_cast<int32_t>(DataType::INT32), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(total) * sizeof(int32_t)));
  (*this)(a, b, a_zero_point, b_zero_point, out);
  return out;
}

void MatMulInteger::operator()(const Tensor &a, const Tensor &b, const Tensor &a_zero_point,
                               const Tensor &b_zero_point, Tensor &output) const {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type),
                      std::string(kName) + ": A must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type),
                      std::string(kName) + ": B must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT32),
                      std::string(kName) + ": output dtype must be INT32.");

  const std::vector<int64_t> out_shape = ComputeOutputShape(a.shape, b.shape);
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      std::string(kName) + ": preallocated output has invalid shape.");
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * sizeof(int32_t),
                      std::string(kName) +
                          ": preallocated output buffer size does not match its shape.");

  const int32_t a_zp = ReadOptionalScalarZP(a_zero_point, a.data_type, "a_zero_point");
  const int32_t b_zp = ReadOptionalScalarZP(b_zero_point, b.data_type, "b_zero_point");

  RunMatMulInteger(a, a_zp, b, b_zp, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
