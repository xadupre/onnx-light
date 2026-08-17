// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates that ``t`` is a 1-D FLOAT tensor of length ``c`` and returns its
// data pointer. ``role`` identifies the parameter in error messages.
const float *AsFloat1D(const Tensor &t, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::InstanceNormalization: ", role, " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u, "kernel::InstanceNormalization: ", role,
                      " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, "kernel::InstanceNormalization: ", role,
                      " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

Tensor InstanceNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                         float epsilon, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::InstanceNormalization: X must be FLOAT.");
  const size_t out_n_bytes = x.size_bytes();
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes, nullptr);
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
      output.size_bytes() == x.size_bytes(),
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

void InstanceNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  onnx_kernels::kernel::InstanceNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, bias, GetEpsilon(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
