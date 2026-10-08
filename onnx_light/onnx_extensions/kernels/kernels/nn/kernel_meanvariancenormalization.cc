// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/nn/variance_accumulator.h"
#include <array>
#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr double kMvnEpsilon = 1e-9;
constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 2> kSupportedElementTypes = {static_cast<int32_t>(DataType::FLOAT),
                                                           static_cast<int32_t>(DataType::DOUBLE)};

Shape NormalizeAxes(const Shape &axes, int64_t rank) {
  Shape normalized;
  normalized.reserve(axes.size());
  Shape seen;
  seen.assign(static_cast<size_t>(rank), 0);
  for (int64_t axis : axes) {
    const int64_t normalized_axis = axis < 0 ? axis + rank : axis;
    EXT_ENFORCE_INVALID(normalized_axis >= 0 && normalized_axis < rank,
                        "kernel::MeanVarianceNormalization: axis out of range.");
    if (!seen[static_cast<size_t>(normalized_axis)]) {
      seen[static_cast<size_t>(normalized_axis)] = 1;
      normalized.push_back(normalized_axis);
    }
  }
  return normalized;
}

int64_t ComputeLane(int64_t idx, const onnx_kernels::Shape &dims,
                    const onnx_kernels::Shape &reduce_mask) {
  const int64_t rank = static_cast<int64_t>(dims.size());
  int64_t lane = 0;
  int64_t lane_multiplier = 1;
  int64_t rem = idx;
  for (int64_t d = rank - 1; d >= 0; --d) {
    const int64_t coord = rem % dims[static_cast<size_t>(d)];
    rem /= dims[static_cast<size_t>(d)];
    if (!reduce_mask[static_cast<size_t>(d)]) {
      lane += coord * lane_multiplier;
      lane_multiplier *= dims[static_cast<size_t>(d)];
    }
  }
  return lane;
}

template <typename T>
void ComputeMvn(const Tensor &x, Tensor &output, const Shape &axes, RawBufferAllocator *allocator,
                int64_t minimum_elements) {
  const onnx_kernels::Shape &dims = x.shape;
  const int64_t rank = static_cast<int64_t>(dims.size());
  const int64_t total = x.element_count();
  if (total == 0) {
    return;
  }

  const Shape normalized_axes = NormalizeAxes(axes, rank);
  Shape reduce_mask;
  reduce_mask.assign(static_cast<size_t>(rank), 0);
  for (int64_t axis : normalized_axes) {
    reduce_mask[static_cast<size_t>(axis)] = 1;
  }

  int64_t reduced_size = 1;
  int64_t lane_count = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (reduce_mask[static_cast<size_t>(i)]) {
      reduced_size *= dims[static_cast<size_t>(i)];
    } else {
      lane_count *= dims[static_cast<size_t>(i)];
    }
  }
  EXT_ENFORCE_INVALID(reduced_size > 0,
                      "kernel::MeanVarianceNormalization: reduced size must be > 0.");

  const T *px = x.As<T>();
  T *py = output.As<T>();
  core::runtime::detail::TemporaryTypedBuffer<statistics::VarianceAccumulator<double>> stats_buf(
      static_cast<size_t>(lane_count), allocator, "kernel::MeanVarianceNormalization statistics");
  auto *stats = stats_buf.data();
  for (int64_t lane = 0; lane < lane_count; ++lane) {
    stats[lane] = {};
  }
  if (lane_count == 1) {
    stats[0] =
        statistics::Accumulate<double>(total, [&](int64_t i) { return px[i]; }, minimum_elements);
  } else {
    for (int64_t idx = 0; idx < total; ++idx) {
      stats[ComputeLane(idx, dims, reduce_mask)].Add(static_cast<double>(px[idx]));
    }
  }

  ParallelFor(
      total, minimum_elements,
      [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t lane = ComputeLane(idx, dims, reduce_mask);
          const auto &stat = stats[lane];
          const double denom = std::sqrt(stat.Variance() + kMvnEpsilon);
          py[idx] = static_cast<T>((static_cast<double>(px[idx]) - stat.mean) / denom);
        }
      },
      "MeanVarianceNormalization");
}

void DispatchMvn(const Tensor &x, Tensor &output, const Shape &axes, RawBufferAllocator *allocator,
                 int64_t minimum_elements) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::MeanVarianceNormalization: X must be FLOAT or DOUBLE.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::MeanVarianceNormalization: output must have the same dtype as X.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::MeanVarianceNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == x.size_bytes(),
      "kernel::MeanVarianceNormalization: output buffer must have the same byte size as X.");

  if (x.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    ComputeMvn<float>(x, output, axes, allocator, minimum_elements);
    return;
  }
  ComputeMvn<double>(x, output, axes, allocator, minimum_elements);
}

} // namespace

MeanVarianceNormalization::MeanVarianceNormalization(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "MeanVarianceNormalization", kSupportedElementTypes,
                            kPortableParallelMinimum, kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(MeanVarianceNormalization)

Tensor MeanVarianceNormalization::operator()(const Tensor &x, const Shape &axes,
                                             RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::MeanVarianceNormalization: X must be FLOAT or DOUBLE.");
  RawBufferAllocator *allocator = rt ? rt->execution_allocator() : nullptr;
  const size_t out_n_bytes = x.size_bytes();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  DispatchMvn(x, out, axes, allocator, tuning().parallel_minimum_elements);
  return out;
}

void MeanVarianceNormalization::operator()(const Tensor &x, Tensor &output,
                                           const Shape &axes) const {
  DispatchMvn(x, output, axes, nullptr, tuning().parallel_minimum_elements);
}

void MeanVarianceNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const std::vector<int64_t> axes = GetAttributeIntsOrDefault(node, "axes", {0, 2, 3});
  SetOutput(node, 0, (*this)(x, axes, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
