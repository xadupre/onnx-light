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

constexpr double kMvnEpsilon = 1e-9;

std::vector<int64_t> NormalizeAxes(const std::vector<int64_t> &axes, int64_t rank) {
  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  std::vector<uint8_t> seen(static_cast<size_t>(rank), 0);
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

int64_t ComputeLane(int64_t idx, const std::vector<int64_t> &dims,
                    const std::vector<uint8_t> &reduce_mask) {
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
void ComputeMvn(const Tensor &x, Tensor &output, const std::vector<int64_t> &axes) {
  const std::vector<int64_t> &dims = x.shape;
  const int64_t rank = static_cast<int64_t>(dims.size());
  const int64_t total = x.element_count();
  if (total == 0) {
    return;
  }

  const std::vector<int64_t> normalized_axes = NormalizeAxes(axes, rank);
  std::vector<uint8_t> reduce_mask(static_cast<size_t>(rank), 0);
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

  std::vector<double> sum(static_cast<size_t>(lane_count), 0.0);
  std::vector<double> sqsum(static_cast<size_t>(lane_count), 0.0);
  std::vector<double> mean(static_cast<size_t>(lane_count), 0.0);

  const T *px = x.As<T>();
  T *py = output.As<T>();

  for (int64_t idx = 0; idx < total; ++idx) {
    const int64_t lane = ComputeLane(idx, dims, reduce_mask);
    sum[static_cast<size_t>(lane)] += static_cast<double>(px[idx]);
  }

  const double reduced_size_d = static_cast<double>(reduced_size);
  for (int64_t lane = 0; lane < lane_count; ++lane) {
    mean[static_cast<size_t>(lane)] = sum[static_cast<size_t>(lane)] / reduced_size_d;
  }

  for (int64_t idx = 0; idx < total; ++idx) {
    const int64_t lane = ComputeLane(idx, dims, reduce_mask);
    const double centered = static_cast<double>(px[idx]) - mean[static_cast<size_t>(lane)];
    sqsum[static_cast<size_t>(lane)] += centered * centered;
  }

  for (int64_t idx = 0; idx < total; ++idx) {
    const int64_t lane = ComputeLane(idx, dims, reduce_mask);
    const double variance = sqsum[static_cast<size_t>(lane)] / reduced_size_d;
    const double denom = std::sqrt(variance + kMvnEpsilon);
    py[idx] =
        static_cast<T>((static_cast<double>(px[idx]) - mean[static_cast<size_t>(lane)]) / denom);
  }
}

} // namespace

Tensor MeanVarianceNormalization::operator()(const Tensor &x,
                                             const std::vector<int64_t> &axes) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::MeanVarianceNormalization: X must be FLOAT or DOUBLE.");
  Tensor out("", x.data_type, x.shape, std::vector<uint8_t>(x.data.size()));
  (*this)(x, out, axes);
  return out;
}

void MeanVarianceNormalization::operator()(const Tensor &x, Tensor &output,
                                           const std::vector<int64_t> &axes) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::MeanVarianceNormalization: X must be FLOAT or DOUBLE.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::MeanVarianceNormalization: output must have the same dtype as X.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::MeanVarianceNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(
      output.data.size() == x.data.size(),
      "kernel::MeanVarianceNormalization: output buffer must have the same byte size as X.");

  if (x.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    ComputeMvn<float>(x, output, axes);
    return;
  }
  ComputeMvn<double>(x, output, axes);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
