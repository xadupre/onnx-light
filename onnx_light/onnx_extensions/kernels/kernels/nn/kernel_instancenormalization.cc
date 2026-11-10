// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/nn/variance_accumulator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 3> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16)};

// Validates a 1-D parameter of length ``c`` with the input element type.
void CheckParameter(const Tensor &t, int32_t dtype, int64_t c, const char *role) {
  EXT_ENFORCE_INVALID(t.data_type == dtype, "kernel::InstanceNormalization: ", role,
                      " must have the same dtype as X.");
  EXT_ENFORCE_INVALID(t.shape.size() == 1u, "kernel::InstanceNormalization: ", role,
                      " must be rank 1.");
  EXT_ENFORCE_INVALID(t.shape[0] == c, "kernel::InstanceNormalization: ", role,
                      " size must equal X's channel dimension.");
}

} // namespace

InstanceNormalization::InstanceNormalization(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "InstanceNormalization", kSupportedElementTypes,
                            kPortableParallelMinimum, kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(InstanceNormalization)

Tensor InstanceNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                         float epsilon, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          IsHalfPrecision(x.data_type),
                      "kernel::InstanceNormalization: X must be FLOAT, FLOAT16 or BFLOAT16.");
  const size_t out_n_bytes = x.size_bytes();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  (*this)(x, scale, bias, out, epsilon);
  return out;
}

void InstanceNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &bias,
                                       Tensor &output, float epsilon) const {
  const bool half = IsHalfPrecision(x.data_type);
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) || half,
                      "kernel::InstanceNormalization: X must be FLOAT, FLOAT16 or BFLOAT16.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::InstanceNormalization: output must have the same dtype as X.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2u,
                      "kernel::InstanceNormalization: X must have rank >= 2.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::InstanceNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == x.size_bytes(),
      "kernel::InstanceNormalization: output buffer must have the same byte size as X.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  CheckParameter(scale, x.data_type, C, "scale");
  CheckParameter(bias, x.data_type, C, "B");

  // Number of elements per (n, c) slice.
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }

  const auto decode = x.data_type == static_cast<int32_t>(DataType::BFLOAT16) ? Bfloat16BitsToFloat
                                                                              : Float16BitsToFloat;
  const auto encode = x.data_type == static_cast<int32_t>(DataType::BFLOAT16) ? FloatToBfloat16Bits
                                                                              : FloatToFloat16Bits;
  const auto read = [&](const Tensor &t, int64_t i) {
    return half ? decode(reinterpret_cast<const uint16_t *>(t.bytes())[i])
                : reinterpret_cast<const float *>(t.bytes())[i];
  };
  float *py = half ? nullptr : output.AsFloat();
  uint16_t *py_half = half ? reinterpret_cast<uint16_t *>(output.mutable_bytes()) : nullptr;

  // For each (n, c) slice compute mean / var across the spatial dims and
  // then ``y = scale[c] * (x - mean) / sqrt(var + epsilon) + bias[c]``.
  // ``spatial`` may be zero in pathological cases; guard the divisions.
  const int64_t work_per_slice = std::max<int64_t>(1, spatial * 3);
  const int64_t grain =
      std::max<int64_t>(1, tuning().parallel_minimum_elements / work_per_slice +
                               (tuning().parallel_minimum_elements % work_per_slice != 0));
  ParallelFor(
      N * C, grain,
      [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
          const int64_t c = task % C;
          const int64_t base = task * spatial;
          const auto stats = statistics::Accumulate<double>(
              spatial, [&](int64_t i) { return read(x, base + i); },
              tuning().parallel_minimum_elements);
          const double mean = stats.mean;
          const double var = stats.Variance();
          const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + epsilon);
          const float s = read(scale, c) * inv_std;
          const float b = read(bias, c);
          for (int64_t i = 0; i < spatial; ++i) {
            const float y = (read(x, base + i) - static_cast<float>(mean)) * s + b;
            if (half) {
              py_half[base + i] = encode(y);
            } else {
              py[base + i] = y;
            }
          }
        }
      },
      "InstanceNormalization");
}

void InstanceNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt);
  const Tensor &scale = GetInput(node, 1, rt);
  const Tensor &bias = GetInput(node, 2, rt);
  SetOutput(node, 0, (*this)(x, scale, bias, GetEpsilon(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
