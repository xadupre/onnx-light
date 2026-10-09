// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 1> kFloatTypes = {static_cast<int32_t>(DataType::FLOAT)};
constexpr std::array<int32_t, 4> kLpTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

int64_t SliceGrain(int64_t minimum_elements, int64_t spatial) {
  const int64_t work_per_slice = std::max<int64_t>(1, spatial);
  return std::max<int64_t>(1, minimum_elements / work_per_slice +
                                  (minimum_elements % work_per_slice != 0));
}

// Returns the total number of spatial elements per (n, c) slice.
int64_t SpatialCount(const Tensor &x) {
  int64_t count = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    count *= x.shape[i];
  }
  return count;
}

template <typename T>
void GlobalLpPoolLoop(const T *x, T *y, int64_t begin, int64_t end, int64_t spatial, double p) {
  for (int64_t i = begin; i < end; ++i) {
    double log_sum = -std::numeric_limits<double>::infinity();
    for (int64_t s = 0; s < spatial; ++s) {
      const double value = p * std::log(std::abs(static_cast<double>(x[i * spatial + s])));
      if (std::isnan(value)) {
        log_sum = value;
        break;
      }
      // Equal infinities need no subtraction: logaddexp(-inf, -inf) is -inf.
      if (log_sum == value) {
        log_sum += std::log(2.0);
      } else {
        log_sum = std::max(log_sum, value) + std::log1p(std::exp(-std::abs(log_sum - value)));
      }
    }
    y[i] = static_cast<T>(std::exp(log_sum / p));
  }
}

} // namespace

GlobalAveragePool::GlobalAveragePool(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GlobalAveragePool", kFloatTypes, kPortableParallelMinimum,
                            kTuningAbi) {}
GlobalMaxPool::GlobalMaxPool(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GlobalMaxPool", kFloatTypes, kPortableParallelMinimum,
                            kTuningAbi) {}
GlobalLpPool::GlobalLpPool(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GlobalLpPool", kLpTypes, kPortableParallelMinimum, kTuningAbi) {}

void GlobalAveragePool::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("GlobalAveragePool", kFloatTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}
void GlobalMaxPool::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("GlobalMaxPool", kFloatTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}
void GlobalLpPool::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("GlobalLpPool", kLpTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}

// ---------------------------------------------------------------------------
// GlobalAveragePool
// ---------------------------------------------------------------------------

Tensor GlobalAveragePool::operator()(const Tensor &x, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalAveragePool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalAveragePool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(SpatialCount(x) > 0,
                      "kernel::GlobalAveragePool: spatial extent must be non-zero.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  // Build output shape: (N, C, 1, 1, ..., 1).
  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  ParallelFor(
      N * C, SliceGrain(tuning().parallel_minimum_elements, spatial),
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const int64_t base = i * spatial;
          double sum = 0.0;
          for (int64_t s = 0; s < spatial; ++s) {
            sum += static_cast<double>(px[base + s]);
          }
          py[i] = static_cast<float>(sum / static_cast<double>(spatial));
        }
      },
      "GlobalAveragePool");
  return out;
}

// ---------------------------------------------------------------------------
// GlobalMaxPool
// ---------------------------------------------------------------------------

Tensor GlobalMaxPool::operator()(const Tensor &x, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalMaxPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalMaxPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(SpatialCount(x) > 0,
                      "kernel::GlobalMaxPool: spatial extent must be non-zero.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  ParallelFor(
      N * C, SliceGrain(tuning().parallel_minimum_elements, spatial),
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const int64_t base = i * spatial;
          float val = px[base];
          for (int64_t s = 1; s < spatial; ++s) {
            val = std::max(val, px[base + s]);
          }
          py[i] = val;
        }
      },
      "GlobalMaxPool");
  return out;
}

// ---------------------------------------------------------------------------
// GlobalLpPool
// ---------------------------------------------------------------------------

Tensor GlobalLpPool::operator()(const Tensor &x, double p, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE) ||
                          IsHalfPrecision(x.data_type),
                      "kernel::GlobalLpPool: x must be FLOAT, DOUBLE, FLOAT16, or BFLOAT16.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalLpPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(std::isfinite(p) && p > 0,
                      "kernel::GlobalLpPool: p must be finite and strictly positive.");
  if (IsHalfPrecision(x.data_type)) {
    Tensor promoted = PromoteToFloat32(x, rt);
    Tensor result = (*this)(promoted, p);
    return DemoteFromFloat32(result, x.data_type, rt);
  }

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * x.element_size();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, out_shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, out_shape, out_n_bytes, nullptr);
  if (x.data_type == static_cast<int32_t>(DataType::DOUBLE)) {
    ParallelFor(
        N * C, SliceGrain(tuning().parallel_minimum_elements, spatial),
        [&](int64_t begin, int64_t end) {
          GlobalLpPoolLoop(x.AsDouble(), out.AsDouble(), begin, end, spatial, p);
        },
        "GlobalLpPool");
  } else {
    ParallelFor(
        N * C, SliceGrain(tuning().parallel_minimum_elements, spatial),
        [&](int64_t begin, int64_t end) {
          GlobalLpPoolLoop(x.AsFloat(), out.AsFloat(), begin, end, spatial, p);
        },
        "GlobalLpPool");
  }
  return out;
}

void GlobalAveragePool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

void GlobalLpPool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const double p = ctx_.opset.version == 1
                       ? static_cast<double>(GetAttributeFloatOrDefault(node, "p", 2.0f))
                       : static_cast<double>(GetAttributeIntOrDefault(node, "p", 2));
  SetOutput(node, 0, (*this)(x, p, &rt), rt);
}

void GlobalMaxPool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
