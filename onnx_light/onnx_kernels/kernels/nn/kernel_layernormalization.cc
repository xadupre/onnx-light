// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Returns ``axis`` normalized to the ``[0, rank]`` range. ``rank`` is a
// legal value: it means "reduce nothing", which still requires a degenerate
// loop but matches the upstream semantics.
int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis <= rank,
                      "kernel::LayerNormalization: axis out of range for X's rank.");
  return axis;
}

// Validates that ``param``'s shape is unidirectionally broadcastable to the
// normalized shape ``X.shape[axis:]``.
void CheckParamBroadcast(const std::vector<int64_t> &x_shape, int64_t axis,
                         const std::vector<int64_t> &param_shape, const char *param_name) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(param_shape.size()) <= normalized_rank,
                      "kernel::LayerNormalization: ", param_name,
                      " rank cannot exceed normalized rank.");
  const int64_t offset = normalized_rank - static_cast<int64_t>(param_shape.size());
  for (size_t i = 0; i < param_shape.size(); ++i) {
    const int64_t x_dim = x_shape[static_cast<size_t>(axis + offset) + i];
    const int64_t p_dim = param_shape[i];
    EXT_ENFORCE_INVALID(p_dim == x_dim || p_dim == 1, "kernel::LayerNormalization: ", param_name,
                        " shape is not broadcastable to X's normalized shape.");
  }
}

// Precomputes, for each position ``flat`` in ``[0, norm_size)`` (row-major
// over ``x_shape[axis:]``), the corresponding flat index into a broadcast
// parameter tensor of shape ``param_shape``. Dimensions of ``param_shape``
// equal to 1 contribute 0 to the index.
std::vector<int64_t> BuildBroadcastIndex(const std::vector<int64_t> &x_shape, int64_t axis,
                                         const std::vector<int64_t> &param_shape,
                                         int64_t norm_size) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  const int64_t param_rank = static_cast<int64_t>(param_shape.size());
  const int64_t offset = normalized_rank - param_rank;

  std::vector<int64_t> param_strides(static_cast<size_t>(param_rank), 0);
  if (param_rank > 0) {
    int64_t stride = 1;
    for (int64_t i = param_rank - 1; i >= 0; --i) {
      const int64_t dim = param_shape[static_cast<size_t>(i)];
      param_strides[static_cast<size_t>(i)] = dim == 1 ? 0 : stride;
      stride *= dim;
    }
  }

  std::vector<int64_t> index(static_cast<size_t>(norm_size), 0);
  if (norm_size > 0 && param_rank > 0) {
    std::vector<int64_t> coord(static_cast<size_t>(normalized_rank), 0);
    for (int64_t flat = 0; flat < norm_size; ++flat) {
      int64_t pi = 0;
      for (int64_t i = offset; i < normalized_rank; ++i) {
        pi += coord[static_cast<size_t>(i)] * param_strides[static_cast<size_t>(i - offset)];
      }
      index[static_cast<size_t>(flat)] = pi;

      for (int64_t i = normalized_rank - 1; i >= 0; --i) {
        ++coord[static_cast<size_t>(i)];
        if (coord[static_cast<size_t>(i)] < x_shape[static_cast<size_t>(axis + i)]) {
          break;
        }
        coord[static_cast<size_t>(i)] = 0;
      }
    }
  }
  return index;
}

// Returns the shape ``[d[0], ..., d[axis-1], 1, ..., 1]`` (rank ==
// rank(x_shape)) used for ``Mean`` and ``InvStdDev``.
std::vector<int64_t> ReducedShape(const std::vector<int64_t> &x_shape, int64_t axis) {
  std::vector<int64_t> shape(x_shape);
  for (int64_t i = axis; i < static_cast<int64_t>(shape.size()); ++i) {
    shape[static_cast<size_t>(i)] = 1;
  }
  return shape;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> LayerNormalization::operator()(const Tensor &x,
                                                                  const Tensor &scale,
                                                                  const Tensor &b, int64_t axis,
                                                                  float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: X must be FLOAT.");
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::LayerNormalization: X must have rank >= 1.");
  const int64_t normalized_axis = NormalizeAxis(axis, rank);
  const std::vector<int64_t> reduced_shape = ReducedShape(x.shape, normalized_axis);
  int64_t reduced_elem = 1;
  for (int64_t d : reduced_shape) {
    reduced_elem *= d;
  }
  const size_t reduced_bytes = static_cast<size_t>(reduced_elem) * sizeof(float);

  Tensor y("", static_cast<int32_t>(DataType::FLOAT), x.shape,
           std::vector<uint8_t>(x.size_bytes()));
  Tensor mean("", static_cast<int32_t>(DataType::FLOAT), reduced_shape,
              std::vector<uint8_t>(reduced_bytes));
  Tensor inv_std_dev("", static_cast<int32_t>(DataType::FLOAT), reduced_shape,
                     std::vector<uint8_t>(reduced_bytes));
  (*this)(x, scale, b, y, mean, inv_std_dev, axis, epsilon);
  return {std::move(y), std::move(mean), std::move(inv_std_dev)};
}

void LayerNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &b,
                                    Tensor &y, Tensor &mean, Tensor &inv_std_dev, int64_t axis,
                                    float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Scale must be FLOAT.");
  EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Y must be FLOAT.");
  EXT_ENFORCE_INVALID(mean.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Mean must be FLOAT.");
  EXT_ENFORCE_INVALID(inv_std_dev.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: InvStdDev must be FLOAT.");
  EXT_ENFORCE_INVALID(y.shape == x.shape,
                      "kernel::LayerNormalization: Y must have the same shape as X.");
  EXT_ENFORCE_INVALID(y.data.size() == x.size_bytes(),
                      "kernel::LayerNormalization: Y buffer must have the same byte size as X.");

  const bool has_bias = !b.shape.empty() || b.size_bytes() > 0;
  if (has_bias) {
    EXT_ENFORCE_INVALID(b.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::LayerNormalization: B must be FLOAT.");
  }

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::LayerNormalization: X must have rank >= 1.");
  axis = NormalizeAxis(axis, rank);

  CheckParamBroadcast(x.shape, axis, scale.shape, "Scale");
  if (has_bias) {
    CheckParamBroadcast(x.shape, axis, b.shape, "B");
  }

  const std::vector<int64_t> reduced_shape = ReducedShape(x.shape, axis);
  EXT_ENFORCE_INVALID(mean.shape == reduced_shape,
                      "kernel::LayerNormalization: Mean shape must equal X's reduced shape.");
  EXT_ENFORCE_INVALID(inv_std_dev.shape == reduced_shape,
                      "kernel::LayerNormalization: InvStdDev shape must equal X's reduced shape.");

  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= x.shape[static_cast<size_t>(i)];
  }
  int64_t norm_size = 1;
  for (int64_t i = axis; i < rank; ++i) {
    norm_size *= x.shape[static_cast<size_t>(i)];
  }

  const std::vector<int64_t> scale_index =
      BuildBroadcastIndex(x.shape, axis, scale.shape, norm_size);
  const std::vector<int64_t> bias_index =
      has_bias ? BuildBroadcastIndex(x.shape, axis, b.shape, norm_size) : std::vector<int64_t>();

  const float *px = x.AsFloat();
  const float *ps = scale.AsFloat();
  const float *pb = has_bias ? b.AsFloat() : nullptr;
  float *py = y.AsFloat();
  float *pmean = mean.AsFloat();
  float *pinv = inv_std_dev.AsFloat();

  for (int64_t o = 0; o < outer; ++o) {
    const int64_t base = o * norm_size;
    double sum = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      sum += static_cast<double>(px[base + i]);
    }
    const double m = norm_size > 0 ? sum / static_cast<double>(norm_size) : 0.0;
    double sqdiff = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      const double d = static_cast<double>(px[base + i]) - m;
      sqdiff += d * d;
    }
    const double var = norm_size > 0 ? sqdiff / static_cast<double>(norm_size) : 0.0;
    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + epsilon);
    pmean[o] = static_cast<float>(m);
    pinv[o] = inv;
    for (int64_t i = 0; i < norm_size; ++i) {
      const float normalized = (px[base + i] - static_cast<float>(m)) * inv;
      const int64_t si = scale_index[static_cast<size_t>(i)];
      float v = normalized * ps[si];
      if (has_bias) {
        v += pb[bias_index[static_cast<size_t>(i)]];
      }
      py[base + i] = v;
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
