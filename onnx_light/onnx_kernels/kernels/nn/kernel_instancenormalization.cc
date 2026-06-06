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
                      std::string("kernel::InstanceNormalization: ") + role + " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u,
                      std::string("kernel::InstanceNormalization: ") + role + " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, std::string("kernel::InstanceNormalization: ") + role +
                                           " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

Tensor InstanceNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                         float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::InstanceNormalization: X must be FLOAT.");
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(x.data.size()));
  (*this)(x, scale, bias, out, epsilon);
  return out;
}

void InstanceNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                       Tensor &output, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::InstanceNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::InstanceNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2u,
                      "kernel::InstanceNormalization: X must have rank >= 2.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::InstanceNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.data.size() == x.data.size(),
      "kernel::InstanceNormalization: output buffer must have the same byte size as X.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  const float *p_scale = AsFloat1D(scale, C, "scale");
  const float *p_bias = AsFloat1D(bias, C, "B");

  // Number of elements per (n, c) slice.
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }

  const float *px = x.AsFloat();
  float *py = output.AsFloat();

  // For each (n, c) slice compute mean / var across the spatial dims and
  // then ``y = scale[c] * (x - mean) / sqrt(var + epsilon) + bias[c]``.
  // ``spatial`` may be zero in pathological cases; guard the divisions.
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      double sum = 0.0;
      for (int64_t i = 0; i < spatial; ++i) {
        sum += static_cast<double>(px[base + i]);
      }
      const double mean = spatial > 0 ? sum / static_cast<double>(spatial) : 0.0;
      double sqsum = 0.0;
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = static_cast<double>(px[base + i]) - mean;
        sqsum += d * d;
      }
      const double var = spatial > 0 ? sqsum / static_cast<double>(spatial) : 0.0;
      const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + epsilon);
      const float s = p_scale[c] * inv_std;
      const float o = p_bias[c] - static_cast<float>(mean) * s;
      for (int64_t i = 0; i < spatial; ++i) {
        py[base + i] = px[base + i] * s + o;
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
