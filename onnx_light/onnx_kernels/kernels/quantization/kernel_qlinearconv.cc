// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::QLinearConv";

inline bool IsInt8OrUint8(int32_t dt) {
  return dt == static_cast<int32_t>(DataType::INT8) || dt == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadElem(const Tensor &t, int64_t idx) {
  if (t.data_type == static_cast<int32_t>(DataType::INT8)) {
    return static_cast<int32_t>(t.AsInt8()[idx]);
  }
  return static_cast<int32_t>(t.AsUint8()[idx]);
}

int32_t ReadScalarInt(const Tensor &t, const char *name) {
  int64_t numel = 1;
  for (int64_t d : t.shape) {
    numel *= d;
  }
  EXT_ENFORCE_INVALID(numel == 1, std::string(kName) + ": '" + name + "' must be a scalar.");
  return ReadElem(t, 0);
}

float ReadFloatElem(const Tensor &t, int64_t idx) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string(kName) + ": scale must be FLOAT for the reference kernel.");
  return t.AsFloat()[idx];
}

void ResolveAttributes(const Tensor &x, const Tensor &w, QLinearConv::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  if (attrs.kernel_shape.empty()) {
    attrs.kernel_shape.assign(w.shape.begin() + 2, w.shape.end());
  }
  EXT_ENFORCE_INVALID(attrs.kernel_shape.size() == spatial_rank,
                      std::string(kName) + ": 'kernel_shape' size must match input spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == spatial_rank,
                      std::string(kName) + ": 'strides' size must match input spatial rank.");
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == spatial_rank,
                      std::string(kName) + ": 'dilations' size must match input spatial rank.");
  if (attrs.auto_pad.empty() || attrs.auto_pad == "NOTSET") {
    if (attrs.pads.empty()) {
      attrs.pads.assign(spatial_rank * 2, 0);
    }
    EXT_ENFORCE_INVALID(attrs.pads.size() == spatial_rank * 2,
                        std::string(kName) + ": 'pads' size must be 2 * spatial rank.");
  } else {
    EXT_ENFORCE_INVALID(attrs.auto_pad == "SAME_UPPER" || attrs.auto_pad == "SAME_LOWER" ||
                            attrs.auto_pad == "VALID",
                        std::string(kName) +
                            ": 'auto_pad' must be NOTSET/SAME_UPPER/SAME_LOWER/VALID.");
    attrs.pads.assign(spatial_rank * 2, 0);
  }
}

std::vector<int64_t> ComputeOutputSpatial(const Tensor &x, QLinearConv::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  std::vector<int64_t> out_spatial(spatial_rank);
  const bool use_auto_pad = !attrs.auto_pad.empty() && attrs.auto_pad != "NOTSET";
  for (size_t i = 0; i < spatial_rank; ++i) {
    const int64_t iD = x.shape[i + 2];
    const int64_t k = attrs.kernel_shape[i];
    const int64_t s = attrs.strides[i];
    const int64_t d = attrs.dilations[i];
    const int64_t eff_k = (k - 1) * d + 1;
    if (use_auto_pad) {
      if (attrs.auto_pad == "VALID") {
        attrs.pads[i] = 0;
        attrs.pads[i + spatial_rank] = 0;
        out_spatial[i] = (iD - eff_k) / s + 1;
      } else {
        const int64_t out = (iD + s - 1) / s;
        const int64_t pad_needed = std::max<int64_t>(0, (out - 1) * s + eff_k - iD);
        if (attrs.auto_pad == "SAME_UPPER") {
          attrs.pads[i] = pad_needed / 2;
          attrs.pads[i + spatial_rank] = pad_needed - pad_needed / 2;
        } else {
          attrs.pads[i + spatial_rank] = pad_needed / 2;
          attrs.pads[i] = pad_needed - pad_needed / 2;
        }
        out_spatial[i] = out;
      }
    } else {
      out_spatial[i] = (iD + attrs.pads[i] + attrs.pads[i + spatial_rank] - eff_k) / s + 1;
    }
  }
  return out_spatial;
}

inline float RoundHalfToEven(float v) { return std::nearbyint(v); }

template <typename Y> inline Y SaturateRound(float scaled, float y_zp_f) {
  constexpr float kMin = static_cast<float>(std::numeric_limits<Y>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<Y>::max());
  float v = RoundHalfToEven(scaled) + y_zp_f;
  if (v < kMin) {
    v = kMin;
  } else if (v > kMax) {
    v = kMax;
  }
  return static_cast<Y>(v);
}

} // namespace

Tensor QLinearConv::operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point,
                               const Tensor &w, const Tensor &w_scale, const Tensor &w_zero_point,
                               const Tensor &y_scale, const Tensor &y_zero_point, const Tensor &B,
                               const Attributes &attrs) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  std::vector<int64_t> out_spatial = ComputeOutputSpatial(x, resolved);
  std::vector<int64_t> out_shape;
  out_shape.reserve(x.shape.size());
  out_shape.push_back(x.shape[0]);
  out_shape.push_back(w.shape[0]);
  for (int64_t d : out_spatial) {
    out_shape.push_back(d);
  }
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor out(
      "", y_zero_point.data_type, out_shape,
      std::vector<uint8_t>(static_cast<size_t>(total) * ElementSize(y_zero_point.data_type)));
  (*this)(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, resolved,
          out);
  return out;
}

void QLinearConv::operator()(const Tensor &x, const Tensor &x_scale, const Tensor &x_zero_point,
                             const Tensor &w, const Tensor &w_scale, const Tensor &w_zero_point,
                             const Tensor &y_scale, const Tensor &y_zero_point, const Tensor &B,
                             const Attributes &attrs, Tensor &output) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  EXT_ENFORCE_INVALID(IsInt8OrUint8(x.data_type),
                      std::string(kName) + ": x must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(w.data_type),
                      std::string(kName) + ": w must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(y_zero_point.data_type),
                      std::string(kName) + ": y_zero_point must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 3, std::string(kName) + ": x must have rank >= 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == x.shape.size(),
                      std::string(kName) + ": w rank must match x rank.");
  EXT_ENFORCE_INVALID(resolved.group >= 1, std::string(kName) + ": 'group' must be >= 1.");
  const int64_t C = x.shape[1];
  const int64_t M = w.shape[0];
  EXT_ENFORCE_INVALID(C == w.shape[1] * resolved.group,
                      std::string(kName) + ": x.shape[1] must equal w.shape[1] * group.");
  EXT_ENFORCE_INVALID(M % resolved.group == 0,
                      std::string(kName) + ": w.shape[0] must be divisible by group.");

  std::vector<int64_t> out_spatial = ComputeOutputSpatial(x, resolved);
  std::vector<int64_t> expected_shape;
  expected_shape.reserve(x.shape.size());
  expected_shape.push_back(x.shape[0]);
  expected_shape.push_back(M);
  for (int64_t d : out_spatial) {
    expected_shape.push_back(d);
  }
  EXT_ENFORCE_INVALID(output.data_type == y_zero_point.data_type,
                      std::string(kName) + ": output dtype must match y_zero_point.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      std::string(kName) + ": preallocated output shape mismatch.");

  const int32_t x_zp = ReadScalarInt(x_zero_point, "x_zero_point");
  const int32_t y_zp = ReadScalarInt(y_zero_point, "y_zero_point");
  EXT_ENFORCE_INVALID(x_zero_point.data_type == x.data_type,
                      std::string(kName) + ": x_zero_point dtype must match x.");
  EXT_ENFORCE_INVALID(w_zero_point.data_type == w.data_type,
                      std::string(kName) + ": w_zero_point dtype must match w.");

  const float x_s = ReadFloatElem(x_scale, 0);
  const float y_s = ReadFloatElem(y_scale, 0);
  EXT_ENFORCE_INVALID(y_s != 0.0f, std::string(kName) + ": y_scale must be non-zero.");
  EXT_ENFORCE_INVALID(x_scale.element_count() == 1 && y_scale.element_count() == 1,
                      std::string(kName) + ": 'x_scale' and 'y_scale' must be scalar.");

  // ``w_scale`` and ``w_zero_point`` are either scalar (per-tensor) or 1-D
  // length-M (per-output-channel).
  int64_t w_scale_numel = 1;
  for (int64_t d : w_scale.shape) {
    w_scale_numel *= d;
  }
  int64_t w_zp_numel = 1;
  for (int64_t d : w_zero_point.shape) {
    w_zp_numel *= d;
  }
  EXT_ENFORCE_INVALID(w_scale_numel == 1 || w_scale_numel == M,
                      std::string(kName) + ": 'w_scale' must be scalar or length-M.");
  EXT_ENFORCE_INVALID(w_zp_numel == 1 || w_zp_numel == M,
                      std::string(kName) + ": 'w_zero_point' must be scalar or length-M.");
  const bool w_scale_per_channel = w_scale_numel == M;
  const bool w_zp_per_channel = w_zp_numel == M;

  const bool has_bias = !B.shape.empty() || B.size_bytes() > 0;
  if (has_bias) {
    EXT_ENFORCE_INVALID(B.data_type == static_cast<int32_t>(DataType::INT32),
                        std::string(kName) + ": B must be INT32.");
    int64_t b_numel = 1;
    for (int64_t d : B.shape) {
      b_numel *= d;
    }
    EXT_ENFORCE_INVALID(b_numel == M, std::string(kName) + ": B must be 1-D with length M.");
  }

  const size_t spatial_rank = x.shape.size() - 2;
  const int64_t N = x.shape[0];
  const int64_t C_per_group = w.shape[1];
  const int64_t M_per_group = M / resolved.group;

  std::vector<int64_t> iD(spatial_rank), oD = out_spatial;
  for (size_t i = 0; i < spatial_rank; ++i) {
    iD[i] = x.shape[i + 2];
  }

  auto compute_strides = [](const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
  };
  const std::vector<int64_t> in_strides = compute_strides(iD);
  const std::vector<int64_t> &k_shape = resolved.kernel_shape;
  const std::vector<int64_t> ker_strides = compute_strides(k_shape);

  int64_t in_spatial_size = 1;
  for (int64_t d : iD) {
    in_spatial_size *= d;
  }
  int64_t out_spatial_size = 1;
  for (int64_t d : oD) {
    out_spatial_size *= d;
  }
  int64_t kernel_size = 1;
  for (int64_t d : k_shape) {
    kernel_size *= d;
  }

  const float y_zp_f = static_cast<float>(y_zp);
  const int32_t *pB = has_bias ? B.AsInt32() : nullptr;
  const bool out_is_int8 = output.data_type == static_cast<int32_t>(DataType::INT8);

  std::vector<int64_t> oidx(spatial_rank, 0);
  std::vector<int64_t> kidx(spatial_rank, 0);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const int64_t g = m / M_per_group;
      const int32_t w_zp = w_zp_per_channel ? ReadElem(w_zero_point, m) : ReadElem(w_zero_point, 0);
      const float w_s = w_scale_per_channel ? ReadFloatElem(w_scale, m) : ReadFloatElem(w_scale, 0);
      const float combined_scale = x_s * w_s / y_s;
      const int64_t out_plane_off = (n * M + m) * out_spatial_size;

      std::fill(oidx.begin(), oidx.end(), 0);
      for (int64_t op = 0; op < out_spatial_size; ++op) {
        int64_t acc = 0;
        for (int64_t ic_in_group = 0; ic_in_group < C_per_group; ++ic_in_group) {
          const int64_t ic = g * C_per_group + ic_in_group;
          const int64_t x_off_base = (n * C + ic) * in_spatial_size;
          const int64_t w_off_base = (m * C_per_group + ic_in_group) * kernel_size;
          std::fill(kidx.begin(), kidx.end(), 0);
          for (int64_t kp = 0; kp < kernel_size; ++kp) {
            bool in_bounds = true;
            int64_t in_off = 0;
            for (size_t a = 0; a < spatial_rank; ++a) {
              const int64_t ia = oidx[a] * resolved.strides[a] + kidx[a] * resolved.dilations[a] -
                                 resolved.pads[a];
              if (ia < 0 || ia >= iD[a]) {
                in_bounds = false;
                break;
              }
              in_off += ia * in_strides[a];
            }
            if (in_bounds) {
              const int32_t xv = ReadElem(x, x_off_base + in_off);
              const int32_t wv = ReadElem(w, w_off_base + kp);
              acc += static_cast<int64_t>(xv - x_zp) * static_cast<int64_t>(wv - w_zp);
            }
            for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
              if (++kidx[a] < k_shape[a]) {
                break;
              }
              kidx[a] = 0;
            }
          }
        }
        // Bias is pre-quantized: bias is added in the integer accumulator
        // domain because it shares scale ``x_scale * w_scale`` (per ONNX).
        if (pB != nullptr) {
          acc += static_cast<int64_t>(pB[m]);
        }
        const float scaled = static_cast<float>(acc) * combined_scale;
        if (out_is_int8) {
          output.AsInt8()[out_plane_off + op] = SaturateRound<int8_t>(scaled, y_zp_f);
        } else {
          output.AsUint8()[out_plane_off + op] = SaturateRound<uint8_t>(scaled, y_zp_f);
        }
        for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
          if (++oidx[a] < oD[a]) {
            break;
          }
          oidx[a] = 0;
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
