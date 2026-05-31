// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Returns the total number of spatial elements per (n, c) slice.
int64_t SpatialCount(const Tensor &x) {
  int64_t count = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    count *= x.shape[i];
  }
  return count;
}

} // namespace

// ---------------------------------------------------------------------------
// GlobalAveragePool
// ---------------------------------------------------------------------------

Tensor GlobalAveragePool::operator()(const Tensor &x) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalAveragePool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalAveragePool: x must have rank >= 2 (N, C, D1, ...).");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  // Build output shape: (N, C, 1, 1, ..., 1).
  std::vector<int64_t> out_shape(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(N * C) * sizeof(float)));

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.data.data());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      double sum = 0.0;
      for (int64_t s = 0; s < spatial; ++s) {
        sum += static_cast<double>(px[base + s]);
      }
      py[n * C + c] = spatial == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(spatial));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GlobalMaxPool
// ---------------------------------------------------------------------------

Tensor GlobalMaxPool::operator()(const Tensor &x) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalMaxPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalMaxPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(SpatialCount(x) > 0,
                      "kernel::GlobalMaxPool: spatial extent must be non-zero.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  std::vector<int64_t> out_shape(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(N * C) * sizeof(float)));

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.data.data());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      float val = px[base];
      for (int64_t s = 1; s < spatial; ++s) {
        val = std::max(val, px[base + s]);
      }
      py[n * C + c] = val;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GlobalLpPool
// ---------------------------------------------------------------------------

Tensor GlobalLpPool::operator()(const Tensor &x, int64_t p) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalLpPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalLpPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(p >= 1, "kernel::GlobalLpPool: p must be >= 1.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  std::vector<int64_t> out_shape(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(N * C) * sizeof(float)));

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.data.data());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      double acc = 0.0;
      for (int64_t s = 0; s < spatial; ++s) {
        acc += std::pow(std::abs(static_cast<double>(px[base + s])), static_cast<double>(p));
      }
      py[n * C + c] = static_cast<float>(std::pow(acc, 1.0 / static_cast<double>(p)));
    }
  }
  return out;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
