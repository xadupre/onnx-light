// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates that ``t`` is a 1-D FLOAT tensor of length ``c`` and returns its
// data pointer. ``role`` identifies the parameter in error messages.
const float *AsFloat1D(const Tensor &t, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::BatchNormalization: ", role, " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u, "kernel::BatchNormalization: ", role,
                      " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, "kernel::BatchNormalization: ", role,
                      " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

Tensor BatchNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                      const Tensor &input_mean, const Tensor &input_var,
                                      float epsilon, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::BatchNormalization: X must be FLOAT.");
  const size_t out_n_bytes = x.size_bytes();
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  (*this)(x, scale, bias, input_mean, input_var, out, epsilon);
  return out;
}

void BatchNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                    const Tensor &input_mean, const Tensor &input_var,
                                    Tensor &output, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::BatchNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::BatchNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(!x.shape.empty(), "kernel::BatchNormalization: X must have rank >= 1.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::BatchNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == x.size_bytes(),
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
  // Scratch buffers are drawn from the runtime allocator backing ``output``
  // (when it is allocator-backed) and fall back to inline storage otherwise.
  RawBufferAllocator *allocator = output.has_allocation() ? output.allocation_owner() : nullptr;
  detail::TemporaryTypedBuffer<float> scale_inv_std_buf(static_cast<size_t>(C), allocator,
                                                        "kernel::BatchNormalization scale_inv_std");
  detail::TemporaryTypedBuffer<float> offset_buf(static_cast<size_t>(C), allocator,
                                                 "kernel::BatchNormalization offset");
  float *scale_inv_std = scale_inv_std_buf.data();
  float *offset = offset_buf.data();
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

std::tuple<Tensor, Tensor, Tensor>
BatchNormalization::TrainingForward(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                    const Tensor &input_mean, const Tensor &input_var,
                                    float epsilon, float momentum, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::BatchNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(!x.shape.empty(), "kernel::BatchNormalization: X must have rank >= 1.");

  // Per the opset 9+ spec, when X is rank 1 it is interpreted as N values
  // with C == 1. Otherwise C is the dim at index 1.
  const int64_t N = x.shape[0];
  const int64_t C = x.shape.size() >= 2u ? x.shape[1] : static_cast<int64_t>(1);

  const float *p_in_mean = AsFloat1D(input_mean, C, "input_mean");
  const float *p_in_var = AsFloat1D(input_var, C, "input_var");

  // Number of elements averaged per channel: N * (D1 * ... * Dk).
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }
  const int64_t per_channel = N * spatial;
  EXT_ENFORCE_INVALID(per_channel > 0,
                      "kernel::BatchNormalization: training mode requires X to be non-empty.");

  const float *px = x.AsFloat();

  // Per-channel batch mean and (population) variance, computed over every
  // axis except the channel axis, matching the ONNX reference. The batch
  // statistics are written straight into the allocator-backed result tensors
  // so no extra scratch storage is required.
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  const size_t saved_mean_t_n_bytes = static_cast<size_t>(C) * sizeof(float);
  Tensor saved_mean_t =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {C}, saved_mean_t_n_bytes, allocator);
  const size_t saved_var_t_n_bytes = static_cast<size_t>(C) * sizeof(float);
  Tensor saved_var_t =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {C}, saved_var_t_n_bytes, allocator);
  float *saved_mean = saved_mean_t.AsFloat();
  float *saved_var = saved_var_t.AsFloat();
  if (x.shape.size() == 1u) {
    double sum = 0.0;
    for (int64_t i = 0; i < N; ++i) {
      sum += static_cast<double>(px[i]);
    }
    const double mean = sum / static_cast<double>(per_channel);
    double sq = 0.0;
    for (int64_t i = 0; i < N; ++i) {
      const double d = static_cast<double>(px[i]) - mean;
      sq += d * d;
    }
    saved_mean[0] = static_cast<float>(mean);
    saved_var[0] = static_cast<float>(sq / static_cast<double>(per_channel));
  } else {
    for (int64_t c = 0; c < C; ++c) {
      double sum = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        const int64_t base = (n * C + c) * spatial;
        for (int64_t i = 0; i < spatial; ++i) {
          sum += static_cast<double>(px[base + i]);
        }
      }
      const double mean = sum / static_cast<double>(per_channel);
      double sq = 0.0;
      for (int64_t n = 0; n < N; ++n) {
        const int64_t base = (n * C + c) * spatial;
        for (int64_t i = 0; i < spatial; ++i) {
          const double d = static_cast<double>(px[base + i]) - mean;
          sq += d * d;
        }
      }
      saved_mean[static_cast<size_t>(c)] = static_cast<float>(mean);
      saved_var[static_cast<size_t>(c)] = static_cast<float>(sq / static_cast<double>(per_channel));
    }
  }

  // Normalize Y using the batch statistics via the inference path.
  Tensor y = (*this)(x, scale, bias, saved_mean_t, saved_var_t, epsilon, rt);

  // Update the running estimates: running = input * momentum + saved * (1 - m).
  const size_t running_mean_n_bytes = static_cast<size_t>(C) * sizeof(float);
  Tensor running_mean =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {C}, running_mean_n_bytes, allocator);
  const size_t running_var_n_bytes = static_cast<size_t>(C) * sizeof(float);
  Tensor running_var =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {C}, running_var_n_bytes, allocator);
  float *p_run_mean = running_mean.AsFloat();
  float *p_run_var = running_var.AsFloat();
  for (int64_t c = 0; c < C; ++c) {
    p_run_mean[c] =
        p_in_mean[c] * momentum + saved_mean[static_cast<size_t>(c)] * (1.0f - momentum);
    p_run_var[c] = p_in_var[c] * momentum + saved_var[static_cast<size_t>(c)] * (1.0f - momentum);
  }

  return {std::move(y), std::move(running_mean), std::move(running_var)};
}

void BatchNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 5);
  RequireOutputRange(node, 1, 3);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  const Tensor &input_mean = GetInput(node, 3, rt.tensors());
  const Tensor &input_var = GetInput(node, 4, rt.tensors());
  onnx_kernels::kernel::BatchNormalization k(rt.kernel_ctx());
  if (GetAttributeIntOrDefault(node, "training_mode", 0) != 0) {
    const float momentum = GetAttributeFloatOrDefault(node, "momentum", 0.9f);
    auto [y, running_mean, running_var] =
        k.TrainingForward(x, scale, bias, input_mean, input_var, GetEpsilon(node), momentum);
    SetOutput(node, 0, std::move(y), rt.tensors());
    if (node.output_size() >= 2) {
      SetOutput(node, 1, std::move(running_mean), rt.tensors());
    }
    if (node.output_size() >= 3) {
      SetOutput(node, 2, std::move(running_var), rt.tensors());
    }
    return;
  }
  EXT_ENFORCE_INVALID(node.output_size() == 1,
                      "RunNode: op 'BatchNormalization' only supports a single output "
                      "(running_mean / running_var require training_mode=1).");
  SetOutput(node, 0, k(x, scale, bias, input_mean, input_var, GetEpsilon(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
