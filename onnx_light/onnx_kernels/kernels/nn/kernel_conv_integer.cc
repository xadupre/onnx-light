// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

void ResolveAttributes(const Tensor &x, const Tensor &w, ConvInteger::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  if (attrs.kernel_shape.empty()) {
    attrs.kernel_shape.assign(w.shape.begin() + 2, w.shape.end());
  }
  EXT_ENFORCE_INVALID(attrs.kernel_shape.size() == spatial_rank,
                      "kernel::ConvInteger: 'kernel_shape' size must match input spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == spatial_rank,
                      "kernel::ConvInteger: 'strides' size must match input spatial rank.");
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == spatial_rank,
                      "kernel::ConvInteger: 'dilations' size must match input spatial rank.");
  if (attrs.auto_pad.empty() || attrs.auto_pad == "NOTSET") {
    if (attrs.pads.empty()) {
      attrs.pads.assign(spatial_rank * 2, 0);
    }
    EXT_ENFORCE_INVALID(attrs.pads.size() == spatial_rank * 2,
                        "kernel::ConvInteger: 'pads' size must be 2 * spatial rank.");
  } else {
    EXT_ENFORCE_INVALID(
        attrs.auto_pad == "SAME_UPPER" || attrs.auto_pad == "SAME_LOWER" ||
            attrs.auto_pad == "VALID",
        "kernel::ConvInteger: 'auto_pad' must be NOTSET/SAME_UPPER/SAME_LOWER/VALID.");
    attrs.pads.assign(spatial_rank * 2, 0);
  }
}

bool IsInt8OrUint8(int32_t dt) {
  return dt == static_cast<int32_t>(DataType::INT8) || dt == static_cast<int32_t>(DataType::UINT8);
}

int32_t ReadElem(const Tensor &t, int64_t idx) {
  if (t.data_type == static_cast<int32_t>(DataType::INT8)) {
    return static_cast<int32_t>(t.AsInt8()[idx]);
  }
  return static_cast<int32_t>(t.AsUint8()[idx]);
}

int32_t ReadScalarOrZero(const Tensor &t) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    return 0;
  }
  return ReadElem(t, 0);
}

void ValidateInputs(const Tensor &x, const Tensor &w, const Tensor &x_zp, const Tensor &w_zp,
                    const ConvInteger::Attributes &attrs) {
  EXT_ENFORCE_INVALID(IsInt8OrUint8(x.data_type), "kernel::ConvInteger: x must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(IsInt8OrUint8(w.data_type), "kernel::ConvInteger: w must be INT8 or UINT8.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 3, "kernel::ConvInteger: x must have rank >= 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == x.shape.size(),
                      "kernel::ConvInteger: w rank must match x rank.");
  EXT_ENFORCE_INVALID(attrs.group >= 1, "kernel::ConvInteger: 'group' must be >= 1.");
  const int64_t C = x.shape[1];
  const int64_t M = w.shape[0];
  EXT_ENFORCE_INVALID(C == w.shape[1] * attrs.group,
                      "kernel::ConvInteger: x.shape[1] must equal w.shape[1] * group.");
  EXT_ENFORCE_INVALID(M % attrs.group == 0,
                      "kernel::ConvInteger: w.shape[0] must be divisible by group.");
  if (!x_zp.shape.empty() || x_zp.size_bytes() > 0) {
    EXT_ENFORCE_INVALID(x_zp.data_type == x.data_type,
                        "kernel::ConvInteger: x_zero_point dtype must match x.");
    int64_t numel = 1;
    for (int64_t d : x_zp.shape) {
      numel *= d;
    }
    EXT_ENFORCE_INVALID(numel == 1, "kernel::ConvInteger: x_zero_point must be a scalar.");
  }
  if (!w_zp.shape.empty() || w_zp.size_bytes() > 0) {
    EXT_ENFORCE_INVALID(w_zp.data_type == w.data_type,
                        "kernel::ConvInteger: w_zero_point dtype must match w.");
    int64_t numel = 1;
    for (int64_t d : w_zp.shape) {
      numel *= d;
    }
    EXT_ENFORCE_INVALID(numel == 1 || numel == M,
                        "kernel::ConvInteger: w_zero_point must be a scalar or 1-D of length M.");
  }
}

std::vector<int64_t> ComputeOutputSpatial(const Tensor &x, ConvInteger::Attributes &attrs) {
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

} // namespace

Tensor ConvInteger::operator()(const Tensor &x, const Tensor &w, const Tensor &x_zero_point,
                               const Tensor &w_zero_point, const Attributes &attrs) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, x_zero_point, w_zero_point, resolved);
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
  Tensor out("", static_cast<int32_t>(DataType::INT32), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(total) * sizeof(int32_t)));
  (*this)(x, w, x_zero_point, w_zero_point, resolved, out);
  return out;
}

void ConvInteger::operator()(const Tensor &x, const Tensor &w, const Tensor &x_zero_point,
                             const Tensor &w_zero_point, const Attributes &attrs,
                             Tensor &output) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, x_zero_point, w_zero_point, resolved);
  std::vector<int64_t> out_spatial = ComputeOutputSpatial(x, resolved);
  std::vector<int64_t> expected_shape;
  expected_shape.reserve(x.shape.size());
  expected_shape.push_back(x.shape[0]);
  expected_shape.push_back(w.shape[0]);
  for (int64_t d : out_spatial) {
    expected_shape.push_back(d);
  }
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT32),
                      "kernel::ConvInteger preallocated output must be INT32.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::ConvInteger preallocated output shape mismatch.");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * sizeof(int32_t),
                      "kernel::ConvInteger preallocated output buffer has unexpected size.");

  const size_t spatial_rank = x.shape.size() - 2;
  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t M = w.shape[0];
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
  const std::vector<int64_t> out_strides = compute_strides(oD);
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

  const int32_t x_zp = ReadScalarOrZero(x_zero_point);
  // Per-channel or scalar w_zp. ``has_w_zp_per_channel`` true means there is
  // a length-M 1-D tensor; otherwise either scalar or absent.
  const bool w_zp_present = !w_zero_point.shape.empty() || !w_zero_point.size_bytes() == 0;
  const bool w_zp_per_channel = w_zp_present && !w_zero_point.shape.empty() &&
                                w_zero_point.shape.size() == 1 && w_zero_point.shape[0] == M;
  const int32_t w_zp_scalar = w_zp_present && !w_zp_per_channel ? ReadElem(w_zero_point, 0) : 0;

  int32_t *py = output.AsInt32();
  std::vector<int64_t> oidx(spatial_rank, 0);
  std::vector<int64_t> kidx(spatial_rank, 0);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const int64_t g = m / M_per_group;
      const int32_t w_zp = w_zp_per_channel ? ReadElem(w_zero_point, m) : w_zp_scalar;
      int32_t *out_plane = py + ((n * M + m) * out_spatial_size);
      std::fill(oidx.begin(), oidx.end(), 0);
      for (int64_t op = 0; op < out_spatial_size; ++op) {
        int32_t acc = 0;
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
              acc += (xv - x_zp) * (wv - w_zp);
            }
            for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
              if (++kidx[a] < k_shape[a]) {
                break;
              }
              kidx[a] = 0;
            }
          }
        }
        out_plane[op] = acc;
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
