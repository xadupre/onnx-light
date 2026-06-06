// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Standard bilinear interpolation of the input plane ``X[batch, channel]``
// at floating-point coordinate (y, x). Samples that fall outside the input
// feature map return ``0``, matching the ONNX ``DeformConv`` spec
// ("Sampling locations outside of the padded input tensor gives zero.").
float BilinearSample(const float *plane, int64_t H, int64_t W, float y, float x) {
  if (y <= -1.0f || y >= static_cast<float>(H) || x <= -1.0f || x >= static_cast<float>(W)) {
    return 0.0f;
  }
  const int64_t y_low = static_cast<int64_t>(std::floor(y));
  const int64_t x_low = static_cast<int64_t>(std::floor(x));
  const int64_t y_high = y_low + 1;
  const int64_t x_high = x_low + 1;
  const float ly = y - static_cast<float>(y_low);
  const float lx = x - static_cast<float>(x_low);
  const float hy = 1.0f - ly;
  const float hx = 1.0f - lx;
  const auto sample = [&](int64_t yi, int64_t xi) -> float {
    if (yi < 0 || yi >= H || xi < 0 || xi >= W) {
      return 0.0f;
    }
    return plane[yi * W + xi];
  };
  const float v1 = sample(y_low, x_low);
  const float v2 = sample(y_low, x_high);
  const float v3 = sample(y_high, x_low);
  const float v4 = sample(y_high, x_high);
  return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

void ResolveAttributes(const Tensor &x, const Tensor &w, DeformConv::Attributes &attrs) {
  const size_t spatial_rank = x.shape.size() - 2;
  if (attrs.kernel_shape.empty()) {
    attrs.kernel_shape.assign(w.shape.begin() + 2, w.shape.end());
  }
  EXT_ENFORCE_INVALID(attrs.kernel_shape.size() == spatial_rank,
                      "kernel::DeformConv: 'kernel_shape' size must match input spatial rank.");
  if (attrs.strides.empty()) {
    attrs.strides.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.strides.size() == spatial_rank,
                      "kernel::DeformConv: 'strides' size must match input spatial rank.");
  if (attrs.dilations.empty()) {
    attrs.dilations.assign(spatial_rank, 1);
  }
  EXT_ENFORCE_INVALID(attrs.dilations.size() == spatial_rank,
                      "kernel::DeformConv: 'dilations' size must match input spatial rank.");
  if (attrs.pads.empty()) {
    attrs.pads.assign(spatial_rank * 2, 0);
  }
  EXT_ENFORCE_INVALID(attrs.pads.size() == spatial_rank * 2,
                      "kernel::DeformConv: 'pads' size must be 2 * spatial rank.");
}

void ValidateInputs(const Tensor &x, const Tensor &w, const Tensor &offset, const Tensor &b,
                    const Tensor &mask, const DeformConv::Attributes &attrs) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DeformConv: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DeformConv: W must be FLOAT.");
  EXT_ENFORCE_INVALID(offset.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DeformConv: offset must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() == 4,
                      "kernel::DeformConv: only the 2-D (rank-4) case is supported.");
  EXT_ENFORCE_INVALID(w.shape.size() == 4, "kernel::DeformConv: W must be rank 4.");
  EXT_ENFORCE_INVALID(offset.shape.size() == 4, "kernel::DeformConv: offset must be rank 4.");
  EXT_ENFORCE_INVALID(attrs.group >= 1, "kernel::DeformConv: 'group' must be >= 1.");
  EXT_ENFORCE_INVALID(attrs.offset_group >= 1, "kernel::DeformConv: 'offset_group' must be >= 1.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t oC = w.shape[0];
  EXT_ENFORCE_INVALID(C == w.shape[1] * attrs.group,
                      "kernel::DeformConv: X.shape[1] must equal W.shape[1] * group.");
  EXT_ENFORCE_INVALID(oC % attrs.group == 0,
                      "kernel::DeformConv: W.shape[0] must be divisible by group.");
  EXT_ENFORCE_INVALID(C % attrs.offset_group == 0,
                      "kernel::DeformConv: X.shape[1] must be divisible by offset_group.");

  const int64_t kH = attrs.kernel_shape[0];
  const int64_t kW = attrs.kernel_shape[1];
  EXT_ENFORCE_INVALID(offset.shape[0] == N,
                      "kernel::DeformConv: offset.shape[0] must match X.shape[0].");
  EXT_ENFORCE_INVALID(offset.shape[1] == attrs.offset_group * kH * kW * 2,
                      "kernel::DeformConv: offset.shape[1] must equal "
                      "offset_group * kH * kW * 2.");

  if (!b.shape.empty()) {
    EXT_ENFORCE_INVALID(b.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::DeformConv: B must be FLOAT when present.");
    EXT_ENFORCE_INVALID(b.shape.size() == 1 && b.shape[0] == oC,
                        "kernel::DeformConv: B must be 1-D of length oC.");
  }
  if (!mask.shape.empty()) {
    EXT_ENFORCE_INVALID(mask.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::DeformConv: mask must be FLOAT when present.");
    EXT_ENFORCE_INVALID(mask.shape.size() == 4 && mask.shape[0] == N &&
                            mask.shape[1] == attrs.offset_group * kH * kW &&
                            mask.shape[2] == offset.shape[2] && mask.shape[3] == offset.shape[3],
                        "kernel::DeformConv: mask shape must be "
                        "(N, offset_group * kH * kW, oH, oW).");
  }
}

std::vector<int64_t> InferOutputShape(const Tensor &x, const Tensor &w, const Tensor &offset,
                                      const DeformConv::Attributes &attrs) {
  const int64_t N = x.shape[0];
  const int64_t oC = w.shape[0];
  const int64_t oH = offset.shape[2];
  const int64_t oW = offset.shape[3];

  // Cross-check against the convolution arithmetic.
  const int64_t iH = x.shape[2];
  const int64_t iW = x.shape[3];
  const int64_t kH = attrs.kernel_shape[0];
  const int64_t kW = attrs.kernel_shape[1];
  const int64_t dH = attrs.dilations[0];
  const int64_t dW = attrs.dilations[1];
  const int64_t sH = attrs.strides[0];
  const int64_t sW = attrs.strides[1];
  const int64_t pad_top = attrs.pads[0];
  const int64_t pad_left = attrs.pads[1];
  const int64_t pad_bot = attrs.pads[2];
  const int64_t pad_right = attrs.pads[3];
  const int64_t eff_kH = (kH - 1) * dH + 1;
  const int64_t eff_kW = (kW - 1) * dW + 1;
  const int64_t expected_oH = (iH + pad_top + pad_bot - eff_kH) / sH + 1;
  const int64_t expected_oW = (iW + pad_left + pad_right - eff_kW) / sW + 1;
  EXT_ENFORCE_INVALID(expected_oH == oH && expected_oW == oW,
                      "kernel::DeformConv: pads/strides/dilations/kernel_shape are "
                      "inconsistent with offset spatial dimensions (oH, oW).");

  return {N, oC, oH, oW};
}

} // namespace

Tensor DeformConv::operator()(const Tensor &x, const Tensor &w, const Tensor &offset,
                              const Tensor &b, const Tensor &mask, const Attributes &attrs) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, offset, b, mask, resolved);
  std::vector<int64_t> out_shape = InferOutputShape(x, w, offset, resolved);
  int64_t total = 1;
  for (int64_t d : out_shape) {
    total *= d;
  }
  Tensor out("", x.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(total) * sizeof(float)));
  (*this)(x, w, offset, b, mask, resolved, out);
  return out;
}

void DeformConv::operator()(const Tensor &x, const Tensor &w, const Tensor &offset, const Tensor &b,
                            const Tensor &mask, const Attributes &attrs, Tensor &output) const {
  Attributes resolved = attrs;
  ResolveAttributes(x, w, resolved);
  ValidateInputs(x, w, offset, b, mask, resolved);
  const std::vector<int64_t> expected_shape = InferOutputShape(x, w, offset, resolved);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DeformConv preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::DeformConv preallocated output shape must be (N, oC, oH, oW).");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::DeformConv preallocated output buffer has unexpected size.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t iH = x.shape[2];
  const int64_t iW = x.shape[3];
  const int64_t oC = w.shape[0];
  const int64_t C_per_group = w.shape[1];
  const int64_t kH = resolved.kernel_shape[0];
  const int64_t kW = resolved.kernel_shape[1];
  const int64_t oH = expected_shape[2];
  const int64_t oW = expected_shape[3];
  const int64_t dH = resolved.dilations[0];
  const int64_t dW = resolved.dilations[1];
  const int64_t sH = resolved.strides[0];
  const int64_t sW = resolved.strides[1];
  const int64_t pad_top = resolved.pads[0];
  const int64_t pad_left = resolved.pads[1];
  const int64_t oC_per_group = oC / resolved.group;
  const int64_t C_per_offset_group = C / resolved.offset_group;

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *poff = offset.AsFloat();
  const float *pb = b.shape.empty() ? nullptr : b.AsFloat();
  const float *pm = mask.shape.empty() ? nullptr : mask.AsFloat();
  float *py = output.AsFloat();

  // offset memory layout: (N, og*kH*kW*2, oH, oW). For batch n, offset_group
  // og, kernel (kh, kw), spatial dim d in {0=y,1=x}, output position (i, j):
  //   poff[((n * (og_total*kH*kW*2) + (((og * kH + kh) * kW + kw) * 2 + d))
  //         * oH + i) * oW + j]
  const int64_t off_C = resolved.offset_group * kH * kW * 2;
  const int64_t mask_C = resolved.offset_group * kH * kW;

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t oc = 0; oc < oC; ++oc) {
      const int64_t g = oc / oC_per_group;
      const float bias = pb ? pb[oc] : 0.0f;
      float *out_plane = py + ((n * oC + oc) * oH) * oW;
      for (int64_t i = 0; i < oH; ++i) {
        for (int64_t j = 0; j < oW; ++j) {
          out_plane[i * oW + j] = bias;
        }
      }
      // Sum contributions from input channels in this group.
      for (int64_t ic_in_group = 0; ic_in_group < C_per_group; ++ic_in_group) {
        const int64_t ic = g * C_per_group + ic_in_group;
        const int64_t og_idx = ic / C_per_offset_group;
        const float *x_plane = px + ((n * C + ic) * iH) * iW;
        const float *w_plane = pw + ((oc * C_per_group + ic_in_group) * kH) * kW;
        for (int64_t i = 0; i < oH; ++i) {
          for (int64_t j = 0; j < oW; ++j) {
            float acc = 0.0f;
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t off_chan_y = ((og_idx * kH + kh) * kW + kw) * 2;
                const int64_t off_chan_x = off_chan_y + 1;
                const float off_y = poff[((n * off_C + off_chan_y) * oH + i) * oW + j];
                const float off_x = poff[((n * off_C + off_chan_x) * oH + i) * oW + j];
                const float sample_y = static_cast<float>(-pad_top + sH * i + kh * dH) + off_y;
                const float sample_x = static_cast<float>(-pad_left + sW * j + kw * dW) + off_x;
                float sample = BilinearSample(x_plane, iH, iW, sample_y, sample_x);
                if (pm != nullptr) {
                  const int64_t m_chan = (og_idx * kH + kh) * kW + kw;
                  sample *= pm[((n * mask_C + m_chan) * oH + i) * oW + j];
                }
                acc += sample * w_plane[kh * kW + kw];
              }
            }
            out_plane[i * oW + j] += acc;
          }
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
