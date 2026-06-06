// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor LRN::operator()(const Tensor &x, int64_t size, float alpha, float beta, float bias) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LRN: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::LRN: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(size > 0, "kernel::LRN: size must be strictly positive.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  // Spatial element count per (n, c) slice.
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }
  const int64_t total = N * C * spatial;

  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(total) * sizeof(float)));

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.data.data());

  // For each (n, c, s) compute sum of squares over the channel window then
  // produce the normalized output. The window is centered on c with extent
  // [max(0, c - floor((size-1)/2)), min(C-1, c + ceil((size-1)/2))].
  const int64_t half_lo = (size - 1) / 2; // floor((size - 1) / 2)
  const int64_t half_hi = size / 2;       // ceil((size - 1) / 2)
  const double scale = static_cast<double>(alpha) / static_cast<double>(size);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t c_begin = std::max<int64_t>(0, c - half_lo);
      const int64_t c_end = std::min<int64_t>(C - 1, c + half_hi);
      for (int64_t s = 0; s < spatial; ++s) {
        double sq_sum = 0.0;
        for (int64_t i = c_begin; i <= c_end; ++i) {
          const double v = static_cast<double>(px[(n * C + i) * spatial + s]);
          sq_sum += v * v;
        }
        const double denom =
            std::pow(static_cast<double>(bias) + scale * sq_sum, static_cast<double>(beta));
        py[(n * C + c) * spatial + s] =
            static_cast<float>(static_cast<double>(px[(n * C + c) * spatial + s]) / denom);
      }
    }
  }
  return out;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
