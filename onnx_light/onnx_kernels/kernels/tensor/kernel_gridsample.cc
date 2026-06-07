// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// --- mode / padding helpers ------------------------------------------------

enum class Interp { Nearest, Linear, Cubic };
enum class Padding { Zeros, Border, Reflection };

Interp ParseMode(const std::string &mode) {
  if (mode == "linear" || mode == "bilinear") {
    return Interp::Linear;
  }
  if (mode == "nearest") {
    return Interp::Nearest;
  }
  if (mode == "cubic" || mode == "bicubic") {
    return Interp::Cubic;
  }
  throw std::invalid_argument(
      "kernel::GridSample: unknown mode '" + mode +
      "' (expected one of 'linear'/'bilinear', 'nearest', 'cubic'/'bicubic').");
}

Padding ParsePaddingMode(const std::string &pm) {
  if (pm == "zeros") {
    return Padding::Zeros;
  }
  if (pm == "border") {
    return Padding::Border;
  }
  if (pm == "reflection") {
    return Padding::Reflection;
  }
  throw std::invalid_argument("kernel::GridSample: unknown padding_mode '" + pm +
                              "' (expected one of 'zeros', 'border', 'reflection').");
}

// --- de-normalisation / reflection -----------------------------------------

// Maps a normalised coordinate ``n`` to the integer-pixel coordinate space
// ``[0, length-1]`` using the same conventions as the upstream Python
// reference implementation (``op_grid_sample.py``).
double Denormalize(double n, int64_t length, bool align_corners) {
  if (align_corners) {
    return (n + 1.0) / 2.0 * static_cast<double>(length - 1);
  }
  return ((n + 1.0) * static_cast<double>(length) - 1.0) / 2.0;
}

// Reflect ``x`` across the closed border ``[x_min, x_max]`` repeatedly
// until it falls inside the border. Mirrors ``_gs_reflect`` in the
// upstream Python reference implementation.
double Reflect(double x, double x_min, double x_max) {
  double fx = x;
  const double rng = x_max - x_min;
  if (rng <= 0.0) {
    return x_min;
  }
  if (fx < x_min) {
    const double dx = x_min - fx;
    const int64_t n = static_cast<int64_t>(dx / rng);
    const double r = dx - static_cast<double>(n) * rng;
    fx = (n % 2 == 0) ? (x_min + r) : (x_max - r);
  } else if (fx > x_max) {
    const double dx = fx - x_max;
    const int64_t n = static_cast<int64_t>(dx / rng);
    const double r = dx - static_cast<double>(n) * rng;
    fx = (n % 2 == 0) ? (x_max - r) : (x_min + r);
  }
  return fx;
}

// Compute the per-dimension reflection borders. The Python reference uses
// ``[-0.5, dim - 0.5]`` (align_corners=0) or ``[0, dim - 1]``
// (align_corners=1).
void PrepareBorders(const std::vector<int64_t> &dims, bool align_corners, std::vector<double> &lo,
                    std::vector<double> &hi) {
  const size_t r = dims.size();
  lo.resize(r);
  hi.resize(r);
  for (size_t i = 0; i < r; ++i) {
    if (align_corners) {
      lo[i] = 0.0;
      hi[i] = static_cast<double>(dims[i] - 1);
    } else {
      lo[i] = -0.5;
      hi[i] = static_cast<double>(dims[i]) - 0.5;
    }
  }
}

// Cubic-convolution coefficients (alpha = -0.75). Mirrors the upstream
// Python reference (``_gs_get_cubic_coeffs``).
void CubicCoeffs(double x, double coeffs[4]) {
  constexpr double a = -0.75;
  x = std::abs(x);
  coeffs[0] = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
  coeffs[1] = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  coeffs[2] = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
  const double t = 2.0 - x;
  coeffs[3] = ((a * t - 5.0 * a) * t + 8.0 * a) * t - 4.0 * a;
}

// --- pixel access ----------------------------------------------------------

// Returns the value of ``X[n, c, i_0, ..., i_{r-1}]`` honoring
// ``padding_mode``. For ``zeros``, out-of-range indices return 0. For
// ``border``, indices are clamped. For ``reflection``, indices are
// reflected using ``lo[k]`` / ``hi[k]``.
template <typename T>
double PixelAtND(const T *x_data, const std::vector<int64_t> &spatial_dims,
                 const std::vector<int64_t> &spatial_strides, const std::vector<int64_t> &idx,
                 Padding pad, const std::vector<double> &lo, const std::vector<double> &hi) {
  const size_t r = spatial_dims.size();
  int64_t offset = 0;
  for (size_t k = 0; k < r; ++k) {
    int64_t i = idx[k];
    const int64_t d = spatial_dims[k];
    if (pad == Padding::Zeros) {
      if (i < 0 || i >= d) {
        return 0.0;
      }
    } else if (pad == Padding::Border) {
      if (i < 0) {
        i = 0;
      } else if (i >= d) {
        i = d - 1;
      }
    } else { // Reflection
      double rf = Reflect(static_cast<double>(i), lo[k], hi[k]);
      // The upstream Python code casts to int (truncate toward 0). After
      // reflection ``rf`` is within [lo, hi] so it is non-negative for
      // align_corners=1 and may be in [-0.5, d-0.5] for align_corners=0.
      i = static_cast<int64_t>(rf);
      if (i < 0) {
        i = 0;
      } else if (i >= d) {
        i = d - 1;
      }
    }
    offset += i * spatial_strides[k];
  }
  return static_cast<double>(x_data[offset]);
}

// Linear interpolation along the first remaining dim, recursing on the
// remaining dims via dimension folding. ``cur_dim`` is the spatial axis
// being interpolated this level. ``base_idx`` holds the index already
// committed for earlier dims (those will be the same for all recursion
// branches); on entry only dims < cur_dim are filled.
template <typename T>
double LinearInterpND(const T *x_data, const std::vector<int64_t> &spatial_dims,
                      const std::vector<int64_t> &spatial_strides,
                      const std::vector<double> &x_coords, Padding pad,
                      const std::vector<double> &lo, const std::vector<double> &hi,
                      std::vector<int64_t> &base_idx, size_t cur_dim) {
  const size_t r = spatial_dims.size();
  const double xc = x_coords[cur_dim];
  const int64_t x0 = static_cast<int64_t>(std::floor(xc));
  const double frac = xc - static_cast<double>(x0);
  const double c0 = 1.0 - frac;
  const double c1 = frac;
  double result = 0.0;
  for (int k = 0; k < 2; ++k) {
    base_idx[cur_dim] = x0 + k;
    double v;
    if (cur_dim + 1 == r) {
      v = PixelAtND(x_data, spatial_dims, spatial_strides, base_idx, pad, lo, hi);
    } else {
      v = LinearInterpND(x_data, spatial_dims, spatial_strides, x_coords, pad, lo, hi, base_idx,
                         cur_dim + 1);
    }
    result += (k == 0 ? c0 : c1) * v;
  }
  return result;
}

template <typename T>
double CubicInterpND(const T *x_data, const std::vector<int64_t> &spatial_dims,
                     const std::vector<int64_t> &spatial_strides,
                     const std::vector<double> &x_coords, Padding pad,
                     const std::vector<double> &lo, const std::vector<double> &hi,
                     std::vector<int64_t> &base_idx, size_t cur_dim) {
  const size_t r = spatial_dims.size();
  const double xc = x_coords[cur_dim];
  const int64_t x0 = static_cast<int64_t>(std::floor(xc));
  double coeffs[4];
  CubicCoeffs(xc - static_cast<double>(x0), coeffs);
  double result = 0.0;
  for (int k = 0; k < 4; ++k) {
    base_idx[cur_dim] = x0 - 1 + k;
    double v;
    if (cur_dim + 1 == r) {
      v = PixelAtND(x_data, spatial_dims, spatial_strides, base_idx, pad, lo, hi);
    } else {
      v = CubicInterpND(x_data, spatial_dims, spatial_strides, x_coords, pad, lo, hi, base_idx,
                        cur_dim + 1);
    }
    result += coeffs[k] * v;
  }
  return result;
}

// Round half-to-even (banker's rounding) to match the upstream Python
// reference's ``np.rint`` used by the nearest-mode of GridSample.
double NearestEvenRound(double v) { return std::nearbyint(v); }

// --- input validation ------------------------------------------------------

void ValidateInputs(const Tensor &X, const Tensor &grid) {
  EXT_ENFORCE_INVALID(X.shape.size() >= 3,
                      "kernel::GridSample: X must have rank >= 3 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(grid.shape.size() == X.shape.size(),
                      "kernel::GridSample: X and grid must have the same rank.");
  EXT_ENFORCE_INVALID(grid.shape[0] == X.shape[0],
                      "kernel::GridSample: X and grid must agree on the batch dim N.");
  const int64_t r = static_cast<int64_t>(X.shape.size()) - 2;
  EXT_ENFORCE_INVALID(grid.shape.back() == r,
                      "kernel::GridSample: the last dim of grid must equal the number of "
                      "spatial dimensions (rank - 2).");
  EXT_ENFORCE_INVALID(X.data_type == grid.data_type,
                      "kernel::GridSample: X and grid must have the same element type.");
  EXT_ENFORCE_INVALID(X.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          X.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::GridSample: only FLOAT and DOUBLE element types are supported.");
}

std::vector<int64_t> ComputeOutputShape(const Tensor &X, const Tensor &grid) {
  std::vector<int64_t> out;
  out.reserve(X.shape.size());
  out.push_back(X.shape[0]); // N
  out.push_back(X.shape[1]); // C
  for (size_t i = 1; i + 1 < grid.shape.size(); ++i) {
    out.push_back(grid.shape[i]);
  }
  return out;
}

// Iterate over all combinations of ``shape`` indices, writing them to
// ``idx`` and invoking ``fn`` for each. ``idx`` is resized as needed.
template <typename Fn>
void ForEachIndex(const std::vector<int64_t> &shape, std::vector<int64_t> &idx, Fn &&fn) {
  const size_t r = shape.size();
  idx.assign(r, 0);
  if (r == 0) {
    fn();
    return;
  }
  for (int64_t total : shape) {
    if (total <= 0) {
      return;
    }
  }
  while (true) {
    fn();
    // Increment.
    int64_t k = static_cast<int64_t>(r) - 1;
    while (k >= 0) {
      idx[k]++;
      if (idx[k] < shape[k]) {
        break;
      }
      idx[k] = 0;
      k--;
    }
    if (k < 0) {
      return;
    }
  }
}

// Main per-channel computation, templated on the element type ``T``.
template <typename T>
void RunTyped(const Tensor &X, const Tensor &grid, Interp interp, Padding pad, bool align_corners,
              std::vector<uint8_t> &out_bytes, const std::vector<int64_t> &out_shape) {
  const T *x_data = reinterpret_cast<const T *>(X.bytes());
  const T *grid_data = reinterpret_cast<const T *>(grid.bytes());
  T *y_data = reinterpret_cast<T *>(out_bytes.data());

  const int64_t N = X.shape[0];
  const int64_t C = X.shape[1];
  const size_t r = X.shape.size() - 2;

  std::vector<int64_t> spatial_dims(X.shape.begin() + 2, X.shape.end());
  std::vector<int64_t> spatial_strides(r);
  if (r > 0) {
    spatial_strides[r - 1] = 1;
    for (size_t k = r - 1; k > 0; --k) {
      spatial_strides[k - 1] = spatial_strides[k] * spatial_dims[k];
    }
  }
  int64_t spatial_count = 1;
  for (int64_t d : spatial_dims) {
    spatial_count *= d;
  }

  std::vector<int64_t> out_spatial_dims(grid.shape.begin() + 1, grid.shape.end() - 1);
  int64_t out_spatial_count = 1;
  for (int64_t d : out_spatial_dims) {
    out_spatial_count *= d;
  }

  std::vector<double> lo, hi;
  PrepareBorders(spatial_dims, align_corners, lo, hi);

  std::vector<int64_t> ox(out_spatial_dims.size(), 0);
  std::vector<double> x_coords(r, 0.0);
  std::vector<int64_t> work_idx(r, 0);

  for (int64_t n = 0; n < N; ++n) {
    // grid layout: [N, *out_spatial, r]
    const T *grid_n = grid_data + n * out_spatial_count * static_cast<int64_t>(r);
    for (int64_t c = 0; c < C; ++c) {
      const T *x_nc = x_data + (n * C + c) * spatial_count;
      T *y_nc = y_data + (n * C + c) * out_spatial_count;

      int64_t flat_out = 0;
      ForEachIndex(out_spatial_dims, ox, [&]() {
        // Compute flat offset into grid_n for index ox.
        int64_t grid_off = 0;
        int64_t stride = static_cast<int64_t>(r);
        for (size_t k = ox.size(); k > 0; --k) {
          grid_off += ox[k - 1] * stride;
          stride *= out_spatial_dims[k - 1];
        }
        // Read normalised coordinates; reverse order (grid stores
        // coordinates in reverse spatial-dim order — see
        // ``op_grid_sample.py``).
        for (size_t k = 0; k < r; ++k) {
          const double n_val = static_cast<double>(grid_n[grid_off + (r - 1 - k)]);
          x_coords[k] = Denormalize(n_val, spatial_dims[k], align_corners);
        }
        if (interp == Interp::Nearest) {
          for (size_t k = 0; k < r; ++k) {
            x_coords[k] = NearestEvenRound(x_coords[k]);
          }
        }
        // For padding modes that aren't Zeros, clip OOB before lookup.
        for (size_t k = 0; k < r; ++k) {
          const double v = x_coords[k];
          if (v < lo[k] || v > hi[k]) {
            if (pad == Padding::Border) {
              x_coords[k] = std::min(std::max(v, 0.0), static_cast<double>(spatial_dims[k] - 1));
            } else if (pad == Padding::Reflection) {
              x_coords[k] = Reflect(v, lo[k], hi[k]);
            }
          }
        }

        double val = 0.0;
        if (interp == Interp::Nearest) {
          for (size_t k = 0; k < r; ++k) {
            work_idx[k] = static_cast<int64_t>(x_coords[k]);
          }
          val = PixelAtND(x_nc, spatial_dims, spatial_strides, work_idx, pad, lo, hi);
        } else if (interp == Interp::Linear) {
          val = LinearInterpND(x_nc, spatial_dims, spatial_strides, x_coords, pad, lo, hi, work_idx,
                               0);
        } else { // Cubic
          val = CubicInterpND(x_nc, spatial_dims, spatial_strides, x_coords, pad, lo, hi, work_idx,
                              0);
        }
        y_nc[flat_out++] = static_cast<T>(val);
      });
      (void)out_shape; // silence -Wunused-parameter in release builds.
    }
  }
}

} // namespace

Tensor GridSample::operator()(const Tensor &X, const Tensor &grid, const Attributes &attrs) const {
  ValidateInputs(X, grid);
  Tensor out;
  out.data_type = X.data_type;
  out.shape = ComputeOutputShape(X, grid);
  int64_t total = 1;
  for (int64_t d : out.shape) {
    total *= d;
  }
  const size_t elt =
      X.data_type == static_cast<int32_t>(DataType::FLOAT) ? sizeof(float) : sizeof(double);
  out.data.assign(static_cast<size_t>(total) * elt, 0);
  (*this)(X, grid, attrs, out);
  return out;
}

void GridSample::operator()(const Tensor &X, const Tensor &grid, const Attributes &attrs,
                            Tensor &output) const {
  (void)ctx_;
  ValidateInputs(X, grid);
  const std::vector<int64_t> expected_shape = ComputeOutputShape(X, grid);
  EXT_ENFORCE_INVALID(output.data_type == X.data_type,
                      "kernel::GridSample: preallocated output element type does not match X.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::GridSample: preallocated output has unexpected shape.");
  int64_t total = 1;
  for (int64_t d : expected_shape) {
    total *= d;
  }
  const size_t elt =
      X.data_type == static_cast<int32_t>(DataType::FLOAT) ? sizeof(float) : sizeof(double);
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(total) * elt,
                      "kernel::GridSample: preallocated output buffer has unexpected size.");

  const Interp interp = ParseMode(attrs.mode);
  const Padding pad = ParsePaddingMode(attrs.padding_mode);
  const bool align_corners = attrs.align_corners != 0;

  if (total == 0) {
    return;
  }

  if (X.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    RunTyped<float>(X, grid, interp, pad, align_corners, output.data, expected_shape);
  } else {
    RunTyped<double>(X, grid, interp, pad, align_corners, output.data, expected_shape);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
