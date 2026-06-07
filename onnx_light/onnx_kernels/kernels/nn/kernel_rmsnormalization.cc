// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Returns ``axis`` normalized to the ``[0, rank)`` range.
int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank,
                      "kernel::RMSNormalization: axis out of range for X's rank.");
  return axis;
}

// Validates that ``scale``'s shape is unidirectionally broadcastable to the
// normalized shape ``X.shape[axis:]``.
void CheckScaleBroadcast(const std::vector<int64_t> &x_shape, int64_t axis,
                         const std::vector<int64_t> &scale_shape) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(scale_shape.size()) <= normalized_rank,
                      "kernel::RMSNormalization: scale rank cannot exceed normalized rank.");
  const int64_t offset = normalized_rank - static_cast<int64_t>(scale_shape.size());
  for (size_t i = 0; i < scale_shape.size(); ++i) {
    const int64_t x_dim = x_shape[static_cast<size_t>(axis + offset) + i];
    const int64_t s_dim = scale_shape[i];
    EXT_ENFORCE_INVALID(s_dim == x_dim || s_dim == 1,
                        "kernel::RMSNormalization: scale shape is not broadcastable to "
                        "X's normalized shape.");
  }
}

} // namespace

Tensor RMSNormalization::operator()(const Tensor &x, const Tensor &scale, int64_t axis,
                                    float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: X must be FLOAT.");
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(x.size_bytes()));
  (*this)(x, scale, out, axis, epsilon);
  return out;
}

void RMSNormalization::operator()(const Tensor &x, const Tensor &scale, Tensor &output,
                                  int64_t axis, float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: scale must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::RMSNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(output.data.size() == x.size_bytes(),
                      "kernel::RMSNormalization: output buffer must have the same byte size as X.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::RMSNormalization: X must have rank >= 1.");
  axis = NormalizeAxis(axis, rank);

  CheckScaleBroadcast(x.shape, axis, scale.shape);

  // Compute the size of the outer block (axes [0, axis)) and the normalized
  // block (axes [axis, rank)).
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= x.shape[static_cast<size_t>(i)];
  }
  int64_t norm_size = 1;
  for (int64_t i = axis; i < rank; ++i) {
    norm_size *= x.shape[static_cast<size_t>(i)];
  }

  // Pre-compute the per-element index into ``scale`` for every position in
  // the normalized block. This is the broadcast resolution: a normalized
  // shape coordinate ``(c_0, ..., c_{normalized_rank-1})`` maps to
  // ``(c_offset, ..., c_{normalized_rank-1})`` in ``scale`` (i.e. the last
  // ``scale_rank`` coordinates), with any ``scale`` dim equal to 1
  // contributing 0 to the index.
  const int64_t normalized_rank = rank - axis;
  const int64_t scale_rank = static_cast<int64_t>(scale.shape.size());
  const int64_t offset = normalized_rank - scale_rank;

  std::vector<int64_t> scale_strides(static_cast<size_t>(scale_rank), 0);
  if (scale_rank > 0) {
    int64_t stride = 1;
    for (int64_t i = scale_rank - 1; i >= 0; --i) {
      const int64_t dim = scale.shape[static_cast<size_t>(i)];
      scale_strides[static_cast<size_t>(i)] = dim == 1 ? 0 : stride;
      stride *= dim;
    }
  }

  std::vector<int64_t> scale_index(static_cast<size_t>(norm_size), 0);
  if (norm_size > 0 && scale_rank > 0) {
    // Walk through the normalized block coordinates in row-major order
    // and accumulate the scale index using ``scale_strides``.
    std::vector<int64_t> coord(static_cast<size_t>(normalized_rank), 0);
    for (int64_t flat = 0; flat < norm_size; ++flat) {
      int64_t si = 0;
      for (int64_t i = offset; i < normalized_rank; ++i) {
        si += coord[static_cast<size_t>(i)] * scale_strides[static_cast<size_t>(i - offset)];
      }
      scale_index[static_cast<size_t>(flat)] = si;

      // Increment ``coord`` in row-major order (last dim varies fastest).
      for (int64_t i = normalized_rank - 1; i >= 0; --i) {
        ++coord[static_cast<size_t>(i)];
        if (coord[static_cast<size_t>(i)] < x.shape[static_cast<size_t>(axis + i)]) {
          break;
        }
        coord[static_cast<size_t>(i)] = 0;
      }
    }
  }

  const float *px = x.AsFloat();
  const float *ps = scale.AsFloat();
  float *py = output.AsFloat();

  // For each outer position, compute the mean of squares over the normalized
  // axes, take the square root and divide ``X`` by it. Then multiply by the
  // broadcasted scale.
  for (int64_t o = 0; o < outer; ++o) {
    const int64_t base = o * norm_size;
    double sqsum = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      const double v = static_cast<double>(px[base + i]);
      sqsum += v * v;
    }
    const double mean = norm_size > 0 ? sqsum / static_cast<double>(norm_size) : 0.0;
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean) + epsilon);
    for (int64_t i = 0; i < norm_size; ++i) {
      py[base + i] = px[base + i] * inv_rms * ps[scale_index[static_cast<size_t>(i)]];
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
