// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Returns the per-axis normalised coordinate values for an axis of length
// ``dim_size``. The convention matches ``torch.nn.functional.affine_grid``
// and the upstream ONNX reference (``op_affine_grid.py``):
//
//   * align_corners == 1: linearly maps integer indices ``[0, dim_size-1]``
//     to the closed interval ``[-1, +1]``; the corner pixel *centres* are
//     at -1 and +1.
//   * align_corners == 0: maps the centre of pixel ``i`` to
//     ``-1 + (2i+1)/dim_size``; the *outer edges* of the corner pixels are
//     at -1 and +1.
//
// A dim of size 1 collapses to a single coordinate at 0 (matches numpy's
// ``np.arange(start, stop, step)`` behaviour for both conventions, where
// the single produced value is ``start``).
std::vector<float> NormalisedCoords(int64_t dim_size, bool align_corners) {
  std::vector<float> out;
  if (dim_size <= 0) {
    return out;
  }
  out.reserve(static_cast<size_t>(dim_size));
  if (align_corners) {
    if (dim_size == 1) {
      out.push_back(-1.0f);
      return out;
    }
    const float step = 2.0f / static_cast<float>(dim_size - 1);
    for (int64_t i = 0; i < dim_size; ++i) {
      out.push_back(-1.0f + step * static_cast<float>(i));
    }
  } else {
    const float step = 2.0f / static_cast<float>(dim_size);
    const float start = -1.0f + step / 2.0f;
    for (int64_t i = 0; i < dim_size; ++i) {
      out.push_back(start + step * static_cast<float>(i));
    }
  }
  return out;
}

void ValidateInputs(const Tensor &theta, const Tensor &size) {
  EXT_ENFORCE_INVALID(theta.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AffineGrid: theta must be FLOAT.");
  EXT_ENFORCE_INVALID(size.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::AffineGrid: size must be INT64.");
  EXT_ENFORCE_INVALID(
      theta.shape.size() == 3,
      "kernel::AffineGrid: theta must be 3-D (N, 2, 3) for 2D or (N, 3, 4) for 3D.");
  EXT_ENFORCE_INVALID(size.shape.size() == 1,
                      "kernel::AffineGrid: size must be 1-D with 4 (2D) or 5 (3D) entries.");
  EXT_ENFORCE_INVALID(size.shape[0] == 4 || size.shape[0] == 5,
                      "kernel::AffineGrid: size must have 4 (2D) or 5 (3D) entries.");
  if (size.shape[0] == 4) {
    EXT_ENFORCE_INVALID(theta.shape[1] == 2 && theta.shape[2] == 3,
                        "kernel::AffineGrid: theta must be (N, 2, 3) for 2D.");
  } else {
    EXT_ENFORCE_INVALID(theta.shape[1] == 3 && theta.shape[2] == 4,
                        "kernel::AffineGrid: theta must be (N, 3, 4) for 3D.");
  }
  const int64_t *size_data = reinterpret_cast<const int64_t *>(size.data.data());
  EXT_ENFORCE_INVALID(size_data[0] == theta.shape[0],
                      "kernel::AffineGrid: size[0] must equal theta's batch dim N.");
}

// Computes the output shape for an AffineGrid call given a fully validated
// ``size`` input (1-D INT64 of length 4 or 5).
std::vector<int64_t> ComputeOutputShape(const Tensor &size) {
  const int64_t *size_data = reinterpret_cast<const int64_t *>(size.data.data());
  std::vector<int64_t> out_shape;
  out_shape.push_back(size_data[0]); // N
  if (size.shape[0] == 4) {
    out_shape.push_back(size_data[2]); // H
    out_shape.push_back(size_data[3]); // W
    out_shape.push_back(2);
  } else {
    out_shape.push_back(size_data[2]); // D
    out_shape.push_back(size_data[3]); // H
    out_shape.push_back(size_data[4]); // W
    out_shape.push_back(3);
  }
  return out_shape;
}

// Applies an (out_dim x in_dim) affine matrix ``theta`` (read row-major) to
// the homogeneous coordinate ``coords`` of length ``in_dim`` and writes the
// resulting ``out_dim`` values to ``out``.
void ApplyAffine(const float *theta, int64_t out_dim, int64_t in_dim, const float *coords,
                 float *out) {
  for (int64_t k = 0; k < out_dim; ++k) {
    float acc = 0.0f;
    for (int64_t j = 0; j < in_dim; ++j) {
      acc += theta[k * in_dim + j] * coords[j];
    }
    out[k] = acc;
  }
}

} // namespace

Tensor AffineGrid::operator()(const Tensor &theta, const Tensor &size,
                              const Attributes &attrs) const {
  ValidateInputs(theta, size);
  Tensor out;
  out.data_type = static_cast<int32_t>(DataType::FLOAT);
  out.shape = ComputeOutputShape(size);
  int64_t total = 1;
  for (int64_t d : out.shape) {
    total *= d;
  }
  out.data.assign(static_cast<size_t>(total) * sizeof(float), 0);
  (*this)(theta, size, attrs, out);
  return out;
}

void AffineGrid::operator()(const Tensor &theta, const Tensor &size, const Attributes &attrs,
                            Tensor &output) const {
  (void)ctx_;
  ValidateInputs(theta, size);
  const std::vector<int64_t> expected_shape = ComputeOutputShape(size);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AffineGrid: preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::AffineGrid: preallocated output has unexpected shape.");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * sizeof(float),
                      "kernel::AffineGrid: preallocated output buffer has unexpected size.");

  const bool align_corners = attrs.align_corners != 0;
  const float *theta_data = reinterpret_cast<const float *>(theta.data.data());
  float *out_data = reinterpret_cast<float *>(output.data.data());

  if (size.shape[0] == 4) {
    // 2D case. Output indexed as [N, H, W, 2].
    const int64_t N = expected_shape[0];
    const int64_t H = expected_shape[1];
    const int64_t W = expected_shape[2];
    const std::vector<float> y_coords = NormalisedCoords(H, align_corners);
    const std::vector<float> x_coords = NormalisedCoords(W, align_corners);
    // Homogeneous coord per (y, x): [x, y, 1] (matches op_affine_grid.py
    // which prepends y for dim 0 then x for dim 1 and finally takes the
    // dot product with theta rows).
    for (int64_t n = 0; n < N; ++n) {
      const float *t = theta_data + n * 6; // (2 x 3) row-major
      float *out_n = out_data + n * H * W * 2;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          const float coords[3] = {x_coords[w], y_coords[h], 1.0f};
          ApplyAffine(t, /*out_dim=*/2, /*in_dim=*/3, coords, out_n + (h * W + w) * 2);
        }
      }
    }
  } else {
    // 3D case. Output indexed as [N, D, H, W, 3].
    const int64_t N = expected_shape[0];
    const int64_t D = expected_shape[1];
    const int64_t H = expected_shape[2];
    const int64_t W = expected_shape[3];
    const std::vector<float> z_coords = NormalisedCoords(D, align_corners);
    const std::vector<float> y_coords = NormalisedCoords(H, align_corners);
    const std::vector<float> x_coords = NormalisedCoords(W, align_corners);
    for (int64_t n = 0; n < N; ++n) {
      const float *t = theta_data + n * 12; // (3 x 4) row-major
      float *out_n = out_data + n * D * H * W * 3;
      for (int64_t d = 0; d < D; ++d) {
        for (int64_t h = 0; h < H; ++h) {
          for (int64_t w = 0; w < W; ++w) {
            const float coords[4] = {x_coords[w], y_coords[h], z_coords[d], 1.0f};
            ApplyAffine(t, /*out_dim=*/3, /*in_dim=*/4, coords, out_n + ((d * H + h) * W + w) * 3);
          }
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
