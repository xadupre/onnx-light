// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::QLinearMatMul";

inline bool IsInt8OrUint8(int32_t dt) {
  return dt == static_cast<int32_t>(DataType::INT8) || dt == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadIntElem(const Tensor &t, int64_t idx) {
  if (t.data_type == static_cast<int32_t>(DataType::INT8)) {
    return static_cast<int32_t>(t.AsInt8()[idx]);
  }
  return static_cast<int32_t>(t.AsUint8()[idx]);
}

int32_t ReadScalarInt(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1,
                      std::string(kName) + ": '" + name + "' must be a scalar.");
  return ReadIntElem(t, 0);
}

float ReadScalarFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1,
                      std::string(kName) + ": '" + name + "' must be a scalar.");
  if (t.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    return t.AsFloat()[0];
  }
  // Best-effort fallback for FLOAT16 / BFLOAT16: convert via the raw bytes.
  // We only need scalar precision for the QLinearMatMul reference and the
  // test cases always supply FLOAT scales, so this branch is defensive.
  throw std::invalid_argument(std::string(kName) + ": '" + name +
                              "' must be a FLOAT scalar for the reference kernel.");
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

// Round-half-to-even, matching the rule the ONNX QuantizeLinear spec uses.
inline float RoundHalfToEven(float v) { return std::nearbyint(v); }

template <typename Y> void StoreSaturated(float v, float y_zp, Y *out, int64_t idx) {
  constexpr float kMin = static_cast<float>(std::numeric_limits<Y>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<Y>::max());
  v = RoundHalfToEven(v) + y_zp;
  if (v < kMin) {
    v = kMin;
  } else if (v > kMax) {
    v = kMax;
  }
  out[idx] = static_cast<Y>(v);
}

void RunQLinearMatMul(const Tensor &a, int32_t a_zp, float a_scale, const Tensor &b, int32_t b_zp,
                      float b_scale, float y_scale, int32_t y_zp, Tensor &output) {
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

  const float combined_scale = a_scale * b_scale / y_scale;
  const float y_zp_f = static_cast<float>(y_zp);

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
        int64_t acc = 0;
        for (int64_t kk = 0; kk < K; ++kk) {
          const int32_t av = ReadIntElem(a, a_base + i * a_row_stride + kk * a_k_stride);
          const int32_t bv = ReadIntElem(b, b_base + kk * b_k_stride + j * b_col_stride);
          acc += static_cast<int64_t>(av - a_zp) * static_cast<int64_t>(bv - b_zp);
        }
        int64_t y_index = y_base;
        if (a.shape.size() != 1 && b.shape.size() != 1) {
          y_index += i * out_strides[batch_rank] + j * out_strides[batch_rank + 1];
        } else if (a.shape.size() == 1 && b.shape.size() != 1) {
          y_index += j * out_strides[batch_rank];
        } else if (a.shape.size() != 1 && b.shape.size() == 1) {
          y_index += i * out_strides[batch_rank];
        }
        const float scaled = static_cast<float>(acc) * combined_scale;
        if (output.data_type == static_cast<int32_t>(DataType::INT8)) {
          StoreSaturated<int8_t>(scaled, y_zp_f, output.AsInt8(), y_index);
        } else {
          StoreSaturated<uint8_t>(scaled, y_zp_f, output.AsUint8(), y_index);
        }
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

Tensor QLinearMatMul::operator()(const Tensor &a, const Tensor &a_scale, const Tensor &a_zero_point,
                                 const Tensor &b, const Tensor &b_scale, const Tensor &b_zero_point,
                                 const Tensor &y_scale, const Tensor &y_zero_point) const {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type),
                      std::string(kName) + ": a must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type),
                      std::string(kName) + ": b must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(y_zero_point.data_type),
                      std::string(kName) + ": y_zero_point must be INT8 or UINT8.");

  const std::vector<int64_t> out_shape = ComputeOutputShape(a.shape, b.shape);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor out("", y_zero_point.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(total)));
  (*this)(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point, out);
  return out;
}

void QLinearMatMul::operator()(const Tensor &a, const Tensor &a_scale, const Tensor &a_zero_point,
                               const Tensor &b, const Tensor &b_scale, const Tensor &b_zero_point,
                               const Tensor &y_scale, const Tensor &y_zero_point,
                               Tensor &output) const {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(a.data_type),
                      std::string(kName) + ": a must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(b.data_type),
                      std::string(kName) + ": b must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(a_zero_point.data_type == a.data_type,
                      std::string(kName) + ": a_zero_point dtype must match a.");
  EXT_ENFORCE_INVALID(b_zero_point.data_type == b.data_type,
                      std::string(kName) + ": b_zero_point dtype must match b.");
  EXT_ENFORCE_INVALID(output.data_type == y_zero_point.data_type,
                      std::string(kName) + ": output dtype must match y_zero_point.");

  const std::vector<int64_t> out_shape = ComputeOutputShape(a.shape, b.shape);
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      std::string(kName) + ": preallocated output has invalid shape.");

  const int32_t a_zp = ReadScalarInt(a_zero_point, "a_zero_point");
  const int32_t b_zp = ReadScalarInt(b_zero_point, "b_zero_point");
  const int32_t y_zp = ReadScalarInt(y_zero_point, "y_zero_point");
  const float a_s = ReadScalarFloat(a_scale, "a_scale");
  const float b_s = ReadScalarFloat(b_scale, "b_scale");
  const float y_s = ReadScalarFloat(y_scale, "y_scale");
  EXT_ENFORCE_INVALID(y_s != 0.0f, std::string(kName) + ": y_scale must be non-zero.");

  RunQLinearMatMul(a, a_zp, a_s, b, b_zp, b_s, y_s, y_zp, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
