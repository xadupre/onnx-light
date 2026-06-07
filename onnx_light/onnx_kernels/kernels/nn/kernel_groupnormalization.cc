// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Validates that ``t`` is a 1-D FLOAT tensor of length ``c`` and returns its
// data pointer. ``role`` identifies the parameter in error messages.
const float *AsFloat1D(const Tensor &t, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::GroupNormalization: ") + role + " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u,
                      std::string("kernel::GroupNormalization: ") + role + " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, std::string("kernel::GroupNormalization: ") + role +
                                           " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

Tensor GroupNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                      int64_t num_groups, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GroupNormalization: X must be FLOAT.");
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(x.size_bytes()));
  (*this)(x, scale, bias, num_groups, out, epsilon);
  return out;
}

void GroupNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                    int64_t num_groups, Tensor &output, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GroupNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GroupNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2u, "kernel::GroupNormalization: X must have rank >= 2.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::GroupNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.data.size() == x.size_bytes(),
      "kernel::GroupNormalization: output buffer must have the same byte size as X.");
  EXT_ENFORCE_INVALID(num_groups > 0, "kernel::GroupNormalization: num_groups must be > 0.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  EXT_ENFORCE_INVALID(C % num_groups == 0,
                      "kernel::GroupNormalization: num_groups must divide the channel dimension.");

  const int64_t group_size = C / num_groups;

  const float *p_scale = AsFloat1D(scale, C, "scale");
  const float *p_bias = AsFloat1D(bias, C, "bias");

  // Number of spatial elements per channel.
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }
  const int64_t group_block = group_size * spatial; // elements per (n, group)

  const float *px = x.AsFloat();
  float *py = output.AsFloat();

  // For each (n, group), compute mean / var over the ``group_size *
  // spatial`` block, then apply the per-channel affine.
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t g = 0; g < num_groups; ++g) {
      const int64_t base = (n * num_groups + g) * group_block;
      double sum = 0.0;
      for (int64_t i = 0; i < group_block; ++i) {
        sum += static_cast<double>(px[base + i]);
      }
      const double mean = group_block > 0 ? sum / static_cast<double>(group_block) : 0.0;
      double sqsum = 0.0;
      for (int64_t i = 0; i < group_block; ++i) {
        const double d = static_cast<double>(px[base + i]) - mean;
        sqsum += d * d;
      }
      const double var = group_block > 0 ? sqsum / static_cast<double>(group_block) : 0.0;
      const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + epsilon);
      const float fmean = static_cast<float>(mean);
      for (int64_t k = 0; k < group_size; ++k) {
        const int64_t c = g * group_size + k;
        const float s = p_scale[c] * inv_std;
        const float o = p_bias[c] - fmean * s;
        const int64_t ch_base = base + k * spatial;
        for (int64_t i = 0; i < spatial; ++i) {
          py[ch_base + i] = px[ch_base + i] * s + o;
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
