// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::FLOAT)};

// Validates that ``t`` is a 1-D FLOAT tensor of length ``c`` and returns its
// data pointer. ``role`` identifies the parameter in error messages.
const float *AsFloat1D(const Tensor &t, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GroupNormalization: ", role, " must be FLOAT.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u, "kernel::GroupNormalization: ", role,
                      " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, "kernel::GroupNormalization: ", role,
                      " size must equal X's channel dimension.");
  return t.AsFloat();
}

} // namespace

GroupNormalization::GroupNormalization(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GroupNormalization", kSupportedElementTypes,
                            kPortableParallelMinimum, kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(GroupNormalization)

Tensor GroupNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                      int64_t num_groups, float epsilon, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GroupNormalization: X must be FLOAT.");
  const size_t out_n_bytes = x.size_bytes();
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes, nullptr);
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
      output.size_bytes() == x.size_bytes(),
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
  const int64_t work_per_group = std::max<int64_t>(1, group_block * 3);
  const int64_t grain =
      std::max<int64_t>(1, tuning().parallel_minimum_elements / work_per_group +
                               (tuning().parallel_minimum_elements % work_per_group != 0));
  ParallelFor(
      N * num_groups, grain,
      [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
          const int64_t g = task % num_groups;
          const int64_t base = task * group_block;
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
      },
      "GroupNormalization");
}

void GroupNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  const int64_t num_groups = GetAttributeIntOrDefault(node, "num_groups", 0);
  SetOutput(node, 0, (*this)(x, scale, bias, num_groups, GetEpsilon(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
