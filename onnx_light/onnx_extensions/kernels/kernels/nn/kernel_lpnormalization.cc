// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::FLOAT)};

} // namespace

LpNormalization::LpNormalization(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "LpNormalization", kSupportedElementTypes,
                            kPortableParallelMinimum, kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(LpNormalization)

Tensor LpNormalization::operator()(const Tensor &x, int64_t axis, int64_t p,
                                   RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LpNormalization: x must be FLOAT.");
  EXT_ENFORCE_INVALID(!x.shape.empty(), "kernel::LpNormalization: x must have rank >= 1.");
  EXT_ENFORCE_INVALID(p == 1 || p == 2, "kernel::LpNormalization: p must be 1 or 2.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  int64_t a = axis;
  if (a < 0) {
    a += rank;
  }
  EXT_ENFORCE_INVALID(a >= 0 && a < rank, "kernel::LpNormalization: axis out of range.");

  // Compute outer/dim/inner so that we can iterate slices along the chosen
  // axis. The flat index is outer * (dim * inner) + d * inner + inner_idx.
  int64_t outer = 1;
  for (int64_t i = 0; i < a; ++i) {
    outer *= x.shape[static_cast<size_t>(i)];
  }
  const int64_t dim = x.shape[static_cast<size_t>(a)];
  int64_t inner = 1;
  for (int64_t i = a + 1; i < rank; ++i) {
    inner *= x.shape[static_cast<size_t>(i)];
  }

  const int64_t total = outer * dim * inner;
  const size_t out_n_bytes = static_cast<size_t>(total) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes, nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  const int64_t work_per_slice = std::max<int64_t>(1, dim * 2);
  const int64_t grain =
      std::max<int64_t>(1, tuning().parallel_minimum_elements / work_per_slice +
                               (tuning().parallel_minimum_elements % work_per_slice != 0));
  ParallelFor(
      outer * inner, grain,
      [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
          const int64_t o = task / inner;
          const int64_t s = task % inner;
          double norm = 0.0;
          for (int64_t d = 0; d < dim; ++d) {
            const double v = static_cast<double>(px[(o * dim + d) * inner + s]);
            norm += (p == 1) ? std::abs(v) : v * v;
          }
          if (p == 2) {
            norm = std::sqrt(norm);
          }
          const double inv = norm == 0.0 ? 0.0 : 1.0 / norm;
          for (int64_t d = 0; d < dim; ++d) {
            const int64_t idx = (o * dim + d) * inner + s;
            py[idx] = static_cast<float>(static_cast<double>(px[idx]) * inv);
          }
        }
      },
      "LpNormalization");
  return out;
}

void LpNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", -1);
  const int64_t p = GetAttributeIntOrDefault(node, "p", 2);
  SetOutput(node, 0, (*this)(x, axis, p, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
