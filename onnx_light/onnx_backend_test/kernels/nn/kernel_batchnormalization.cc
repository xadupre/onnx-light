// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Validates that ``t`` is a 1-D FLOAT tensor of length ``c`` and returns its
// data pointer. ``role`` identifies the parameter in error messages.
const float *AsFloat1D(const Tensor &t, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      std::string("kernel::BatchNormalization: ") + role + " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u,
                      std::string("kernel::BatchNormalization: ") + role + " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, std::string("kernel::BatchNormalization: ") + role +
                                           " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

Tensor BatchNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                      const Tensor &input_mean, const Tensor &input_var,
                                      float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::BatchNormalization: X must be FLOAT.");
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), x.shape,
             std::vector<uint8_t>(x.data.size()));
  (*this)(x, scale, bias, input_mean, input_var, out, epsilon);
  return out;
}

void BatchNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                    const Tensor &input_mean, const Tensor &input_var,
                                    Tensor &output, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::BatchNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::BatchNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(!x.shape.empty(), "kernel::BatchNormalization: X must have rank >= 1.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::BatchNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.data.size() == x.data.size(),
      "kernel::BatchNormalization: output buffer must have the same byte size as X.");

  // Per the opset 9+ spec, when X is rank 1 it is interpreted as N values
  // with C == 1. Otherwise C is the dim at index 1.
  const int64_t N = x.shape[0];
  const int64_t C = x.shape.size() >= 2u ? x.shape[1] : static_cast<int64_t>(1);

  const float *p_scale = AsFloat1D(scale, C, "scale");
  const float *p_bias = AsFloat1D(bias, C, "B");
  const float *p_mean = AsFloat1D(input_mean, C, "input_mean");
  const float *p_var = AsFloat1D(input_var, C, "input_var");

  // Total elements per channel block (D1 * D2 * ... * Dk) when rank >= 2,
  // and 1 when rank == 1 (each value belongs to the single channel).
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }

  const float *px = x.AsFloat();
  float *py = output.AsFloat();

  // Pre-compute the per-channel normalization scale and offset:
  //   y = (x - mean) * inv_std * scale + B
  //     = x * (scale * inv_std) + (B - mean * scale * inv_std)
  std::vector<float> scale_inv_std(static_cast<size_t>(C));
  std::vector<float> offset(static_cast<size_t>(C));
  for (int64_t c = 0; c < C; ++c) {
    const float inv_std = 1.0f / std::sqrt(p_var[c] + epsilon);
    scale_inv_std[c] = p_scale[c] * inv_std;
    offset[c] = p_bias[c] - p_mean[c] * scale_inv_std[c];
  }

  if (x.shape.size() == 1u) {
    // Rank-1 input: every element is in channel 0.
    const float s = scale_inv_std[0];
    const float o = offset[0];
    for (int64_t i = 0; i < N; ++i) {
      py[i] = px[i] * s + o;
    }
    return;
  }

  // Rank >= 2: iterate over (n, c, spatial_idx).
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const float s = scale_inv_std[c];
      const float o = offset[c];
      const int64_t base = (n * C + c) * spatial;
      for (int64_t i = 0; i < spatial; ++i) {
        py[base + i] = px[base + i] * s + o;
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
