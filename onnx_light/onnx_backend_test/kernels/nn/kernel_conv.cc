// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Resolves the four spatial attribute vectors (kernel_shape, strides,
// dilations, pads) to their canonical sizes given the input/weight tensors.
// When ``auto_pad`` is not ``NOTSET`` the ``pads`` argument is recomputed
// once the output spatial size is known (see ``ResolveAutoPad`` below).
void ResolveAttributes(const Tensor &x, const Tensor &w, Conv::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  if (attrs.kernel_shape.empty()) {
    attrs.kernel_shape.assign(w.shape.begin() + 2, w.shape.end());
  }
  EXT_ENFORCE_INVALID(attrs.kernel_shape.size() == spatial_rank,
                      "kernel::Conv: 'kernel_shape' size must match input spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == spatial_rank,
                      "kernel::Conv: 'strides' size must match input spatial rank.");
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == spatial_rank,
                      "kernel::Conv: 'dilations' size must match input spatial rank.");
  if (attrs.auto_pad.empty() || attrs.auto_pad == "NOTSET") {
    if (attrs.pads.empty()) {
      attrs.pads.assign(spatial_rank * 2, 0);
    }
    EXT_ENFORCE_INVALID(attrs.pads.size() == spatial_rank * 2,
                        "kernel::Conv: 'pads' size must be 2 * spatial rank.");
  } else {
    EXT_ENFORCE_INVALID(
        attrs.auto_pad == "SAME_UPPER" || attrs.auto_pad == "SAME_LOWER" ||
            attrs.auto_pad == "VALID",
        "kernel::Conv: 'auto_pad' must be NOTSET, SAME_UPPER, SAME_LOWER or VALID.");
    attrs.pads.assign(spatial_rank * 2, 0);
  }
}

void ValidateInputs(const Tensor &x, const Tensor &w, const Tensor &b,
                    const Conv::Attributes &attrs) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::Conv: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::Conv: W must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 3, "kernel::Conv: X must have rank >= 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == x.shape.size(), "kernel::Conv: W rank must match X rank.");
  EXT_ENFORCE_INVALID(attrs.group >= 1, "kernel::Conv: 'group' must be >= 1.");
  const int64_t C = x.shape[1];
  const int64_t M = w.shape[0];
  EXT_ENFORCE_INVALID(C == w.shape[1] * attrs.group,
                      "kernel::Conv: X.shape[1] must equal W.shape[1] * group.");
  EXT_ENFORCE_INVALID(M % attrs.group == 0, "kernel::Conv: W.shape[0] must be divisible by group.");
  if (!b.shape.empty()) {
    EXT_ENFORCE_INVALID(b.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::Conv: B must be FLOAT when present.");
    EXT_ENFORCE_INVALID(b.shape.size() == 1 && b.shape[0] == M,
                        "kernel::Conv: B must be 1-D of length M.");
  }
}

// Computes the per-axis output spatial dimension and (for SAME_*/VALID auto-pad)
// fills the begin/end pad vectors at indices ``i`` and ``i + spatial_rank``.
std::vector<int64_t> ComputeOutputSpatial(const Tensor &x, Conv::Attributes &attrs) {
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
        const int64_t out = (iD + s - 1) / s; // ceil(iD / s)
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

Tensor Conv::operator()(const Tensor &x, const Tensor &w, const Tensor &b,
                        const Attributes &attrs) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, b, resolved);
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
  Tensor out("", x.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(total) * sizeof(float)));
  (*this)(x, w, b, resolved, out);
  return out;
}

void Conv::operator()(const Tensor &x, const Tensor &w, const Tensor &b, const Attributes &attrs,
                      Tensor &output) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, b, resolved);
  std::vector<int64_t> out_spatial = ComputeOutputSpatial(x, resolved);
  std::vector<int64_t> expected_shape;
  expected_shape.reserve(x.shape.size());
  expected_shape.push_back(x.shape[0]);
  expected_shape.push_back(w.shape[0]);
  for (int64_t d : out_spatial) {
    expected_shape.push_back(d);
  }
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::Conv preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::Conv preallocated output shape must equal (N, M, oD1, ..., oDk).");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::Conv preallocated output buffer has unexpected size.");

  const size_t spatial_rank = x.shape.size() - 2;
  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t M = w.shape[0];
  const int64_t C_per_group = w.shape[1];
  const int64_t M_per_group = M / resolved.group;

  // Per-axis input spatial sizes and per-axis output spatial sizes.
  std::vector<int64_t> iD(spatial_rank), oD = out_spatial;
  for (size_t i = 0; i < spatial_rank; ++i) {
    iD[i] = x.shape[i + 2];
  }

  // Helper: strides through a contiguous (D1..Dk) buffer.
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

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pb = b.shape.empty() ? nullptr : b.AsFloat();
  float *py = output.AsFloat();

  // Index into spatial outputs by decomposition.
  std::vector<int64_t> oidx(spatial_rank, 0);
  std::vector<int64_t> kidx(spatial_rank, 0);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t m = 0; m < M; ++m) {
      const int64_t g = m / M_per_group;
      const float bias = pb ? pb[m] : 0.0f;
      float *out_plane = py + ((n * M + m) * out_spatial_size);
      // Loop over output spatial positions.
      std::fill(oidx.begin(), oidx.end(), 0);
      for (int64_t op = 0; op < out_spatial_size; ++op) {
        float acc = bias;
        // Loop over input channels in this group.
        for (int64_t ic_in_group = 0; ic_in_group < C_per_group; ++ic_in_group) {
          const int64_t ic = g * C_per_group + ic_in_group;
          const float *x_plane = px + ((n * C + ic) * in_spatial_size);
          const float *w_plane = pw + ((m * C_per_group + ic_in_group) * kernel_size);
          // Loop over kernel positions.
          std::fill(kidx.begin(), kidx.end(), 0);
          for (int64_t kp = 0; kp < kernel_size; ++kp) {
            // Compute corresponding input spatial coordinate.
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
              acc += x_plane[in_off] * w_plane[kp];
            }
            // Increment kidx (rightmost is fastest).
            for (int a = static_cast<int>(spatial_rank) - 1; a >= 0; --a) {
              if (++kidx[a] < k_shape[a]) {
                break;
              }
              kidx[a] = 0;
            }
          }
        }
        out_plane[op] = acc;
        // Increment oidx (rightmost is fastest).
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
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
