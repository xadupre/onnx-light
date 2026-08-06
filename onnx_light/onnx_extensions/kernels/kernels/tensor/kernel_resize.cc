// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include "onnx_light_helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Reads the 1-D FLOAT ``scales`` input tensor into an allocator-backed scratch
// buffer. ``expected_length`` is the number of axes the kernel expects scales
// for (``rank`` when ``axes`` is absent, ``axes.size()`` otherwise). ``allocator``
// (when non-null) supplies the scratch storage; otherwise the returned tensor
// falls back to inline storage.
Tensor ReadResizeScales(const Tensor &scales, std::size_t expected_length,
                        RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(scales.data_type == DataType::FLOAT,
                      "kernel::Resize: 'scales' input must be FLOAT.");
  EXT_ENFORCE_INVALID(scales.shape.size() == 1,
                      "kernel::Resize: 'scales' input must be a 1-D tensor.");
  const int64_t n = scales.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == expected_length,
                      "kernel::Resize: 'scales' length must match the number of resized axes.");
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(float);
  Tensor out = MakeOutputTensor(DataType::FLOAT, {n}, n_bytes, allocator);
  if (n > 0) {
    std::memcpy(out.mutable_bytes(), scales.bytes(), n_bytes);
  }
  const float *values = out.As<float>();
  for (int64_t i = 0; i < n; ++i) {
    EXT_ENFORCE_INVALID(values[i] > 0.0f, "kernel::Resize: 'scales' values must be > 0.");
  }
  return out;
}

// Reads the 1-D INT64 ``sizes`` input tensor into an allocator-backed scratch
// buffer. ``allocator`` (when non-null) supplies the scratch storage; otherwise
// the returned tensor falls back to inline storage.
Tensor ReadResizeSizes(const Tensor &sizes, std::size_t expected_length,
                       RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(sizes.data_type == DataType::INT64,
                      "kernel::Resize: 'sizes' input must be INT64.");
  EXT_ENFORCE_INVALID(sizes.shape.size() == 1,
                      "kernel::Resize: 'sizes' input must be a 1-D tensor.");
  const int64_t n = sizes.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == expected_length,
                      "kernel::Resize: 'sizes' length must match the number of resized axes.");
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(int64_t);
  Tensor out = MakeOutputTensor(DataType::INT64, {n}, n_bytes, allocator);
  if (n > 0) {
    std::memcpy(out.mutable_bytes(), sizes.bytes(), n_bytes);
  }
  const int64_t *values = out.As<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    EXT_ENFORCE_INVALID(values[i] > 0, "kernel::Resize: 'sizes' values must be > 0.");
  }
  return out;
}

// Normalises the user-supplied ``axes`` attribute against ``rank``. When the
// attribute is empty, returns ``{0, 1, ..., rank-1}``.
onnx_kernels::Shape NormaliseAxes(const onnx_kernels::Shape &axes, std::size_t rank) {
  onnx_kernels::Shape out;
  if (axes.empty()) {
    out.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      out.push_back(static_cast<int64_t>(i));
    }
    return out;
  }
  out.reserve(axes.size());
  for (int64_t a : axes) {
    int64_t na = a < 0 ? a + static_cast<int64_t>(rank) : a;
    EXT_ENFORCE_INVALID(na >= 0 && na < static_cast<int64_t>(rank),
                        "kernel::Resize: 'axes' value out of range.");
    out.push_back(na);
  }
  return out;
}

// Expands a per-axis ``scales``/``sizes`` array to a length-``rank`` array,
// inserting ``identity`` values on non-resized axes. ``values`` points to
// ``axes.size()`` elements; ``out`` points to storage for ``rank`` elements
// (typically an allocator-backed :cpp:class:`detail::TemporaryTypedBuffer`).
template <typename T>
void ScatterByAxes(const T *values, const onnx_kernels::Shape &axes, std::size_t rank, T identity,
                   T *out) {
  for (std::size_t i = 0; i < rank; ++i) {
    out[i] = identity;
  }
  for (std::size_t i = 0; i < axes.size(); ++i) {
    out[static_cast<std::size_t>(axes[i])] = values[i];
  }
}

// Applies the ``nearest_mode`` rounding rule to convert a (real-valued)
// input coordinate to an integer index. Supports the four modes defined by
// the ONNX Resize spec.
int64_t ApplyNearestMode(double x, const std::string &nearest_mode) {
  if (nearest_mode == "floor") {
    return static_cast<int64_t>(std::floor(x));
  }
  if (nearest_mode == "ceil") {
    return static_cast<int64_t>(std::ceil(x));
  }
  if (nearest_mode == "round_prefer_ceil") {
    // Round half up.
    return static_cast<int64_t>(std::floor(x + 0.5));
  }
  // Default: "round_prefer_floor" — round half down.
  const double f = std::floor(x);
  if (x - f == 0.5) {
    return static_cast<int64_t>(f);
  }
  return static_cast<int64_t>(std::floor(x + 0.5));
}

// Computes the real-valued input coordinate for a given output position
// according to ``coordinate_transformation_mode``. Mirrors the formulas in
// ``onnx/reference/ops/op_resize.py``. ``roi_start``/``roi_end`` are used
// only by ``"tf_crop_and_resize"`` and ignored for every other mode.
double TransformCoord(int64_t out_coord, int64_t in_dim, int64_t out_dim, double scale,
                      const std::string &mode, double roi_start, double roi_end) {
  const double x = static_cast<double>(out_coord);
  if (mode == "asymmetric") {
    return x / scale;
  }
  if (mode == "align_corners") {
    if (out_dim == 1) {
      return 0.0;
    }
    // Upstream ``onnx/reference/ops/op_resize.py::_interpolate_1d_with_x``
    // computes the denominator from ``output_width = scale * input_width``
    // (a float), not the integer ``out_dim``. The two only differ when
    // ``scale * in_dim`` is non-integer (typically downsampling with a
    // fractional scale), but in that case using the integer ``out_dim``
    // lands sample positions exactly on input grid points and breaks
    // bit-exact agreement with the upstream reference.
    return x * static_cast<double>(in_dim - 1) / (static_cast<double>(in_dim) * scale - 1.0);
  }
  if (mode == "pytorch_half_pixel") {
    if (out_dim == 1) {
      return -0.5;
    }
    return (x + 0.5) / scale - 0.5;
  }
  if (mode == "half_pixel_symmetric") {
    const double output_width = static_cast<double>(in_dim) * scale;
    const double adjustment = static_cast<double>(out_dim) / output_width;
    const double center = static_cast<double>(in_dim) / 2.0;
    const double offset = center * (1.0 - adjustment);
    return offset + (x + 0.5) / scale - 0.5;
  }
  if (mode == "tf_crop_and_resize") {
    const double length_original = static_cast<double>(in_dim);
    if (out_dim == 1) {
      return (roi_start + roi_end) * (length_original - 1.0) / 2.0;
    }
    return roi_start * (length_original - 1.0) +
           x * (roi_end - roi_start) * (length_original - 1.0) / static_cast<double>(out_dim - 1);
  }
  // Default: "half_pixel" (the schema default since opset 13).
  EXT_ENFORCE_INVALID(mode == "half_pixel",
                      "kernel::Resize: unsupported coordinate_transformation_mode.");
  return (x + 0.5) / scale - 0.5;
}

// Nearest-neighbor resize for any rank, byte-element-wise copy. Combines the
// per-axis ``scales`` and ``coordinate_transformation_mode`` to compute the
// real-valued input coordinate, then rounds it via ``nearest_mode`` and
// clamps the result to ``[0, in_dim - 1]``. When ``coord_mode`` is
// ``"tf_crop_and_resize"`` and the real-valued coordinate falls outside
// ``[0, in_dim - 1]`` on any axis, the output element is set to
// ``extrapolation_value`` rather than clamped.
void ResizeNearest(const Tensor &input, const float *scales, const onnx_kernels::Shape &out_shape,
                   const std::string &nearest_mode, const std::string &coord_mode,
                   const double *roi_start, const double *roi_end, double extrapolation_value,
                   Tensor &output) {
  const std::size_t elem_size = ElementSize(input.data_type);
  const std::size_t rank = out_shape.size();
  const bool is_tf_crop = coord_mode == "tf_crop_and_resize";

  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }

  onnx_kernels::Shape in_strides;
  in_strides.assign(rank, 0);
  onnx_kernels::Shape out_strides;
  out_strides.assign(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
      out_strides[k - 1] = out_strides[k] * out_shape[k];
    }
  }

  const uint8_t *const in_ptr = input.bytes();
  uint8_t *const out_ptr = output.mutable_bytes();

  // Pre-encoded ``extrapolation_value`` for FLOAT/DOUBLE outputs; for other
  // (whole-byte) types ``tf_crop_and_resize`` extrapolation is not defined
  // by the spec, so we fall back to clamping.
  float extrap_f = static_cast<float>(extrapolation_value);
  double extrap_d = extrapolation_value;

  for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    bool extrapolate = false;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t out_coord = remaining / out_strides[k];
      remaining %= out_strides[k];
      const double x_ori =
          TransformCoord(out_coord, input.shape[k], out_shape[k], static_cast<double>(scales[k]),
                         coord_mode, roi_start[k], roi_end[k]);
      if (is_tf_crop && (x_ori < 0.0 || x_ori > static_cast<double>(input.shape[k] - 1))) {
        extrapolate = true;
        break;
      }
      int64_t in_coord = ApplyNearestMode(x_ori, nearest_mode);
      if (in_coord >= input.shape[k]) {
        in_coord = input.shape[k] - 1;
      }
      if (in_coord < 0) {
        in_coord = 0;
      }
      in_idx += in_coord * in_strides[k];
    }
    if (extrapolate) {
      if (input.data_type == DataType::FLOAT) {
        std::memcpy(out_ptr + static_cast<std::size_t>(out_idx) * elem_size, &extrap_f, elem_size);
      } else if (input.data_type == DataType::DOUBLE) {
        std::memcpy(out_ptr + static_cast<std::size_t>(out_idx) * elem_size, &extrap_d, elem_size);
      } else {
        // For non-floating-point dtypes, zero-fill the element as a
        // best-effort extrapolation (the ONNX spec only defines
        // ``extrapolation_value`` for floating-point outputs).
        std::memset(out_ptr + static_cast<std::size_t>(out_idx) * elem_size, 0, elem_size);
      }
    } else {
      std::memcpy(out_ptr + static_cast<std::size_t>(out_idx) * elem_size,
                  in_ptr + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
    }
  }
}

bool IsNearestMode(const std::string &mode) { return mode == "nearest"; }

bool IsLinearMode(const std::string &mode) { return mode == "linear" || mode == "bilinear"; }

bool IsCubicMode(const std::string &mode) { return mode == "cubic"; }

// Loads element ``idx`` of a floating-point tensor as a double. Only FLOAT and
// DOUBLE are supported here -- linear / cubic resize requires real arithmetic.
double LoadFloat(const Tensor &t, int64_t idx) {
  const uint8_t *const base = t.bytes();
  switch (t.data_type) {
  case DataType::FLOAT: {
    float v;
    std::memcpy(&v, base + static_cast<std::size_t>(idx) * sizeof(float), sizeof(float));
    return static_cast<double>(v);
  }
  case DataType::DOUBLE: {
    double v;
    std::memcpy(&v, base + static_cast<std::size_t>(idx) * sizeof(double), sizeof(double));
    return v;
  }
  default:
    EXT_THROW_INVALID("unsupported data type ", t.data_type, ", ",
                      "kernel::Resize: linear/cubic modes only support FLOAT/DOUBLE input types.");
  }
}

void StoreFloat(Tensor &t, int64_t idx, double value) {
  uint8_t *const base = t.mutable_bytes();
  switch (t.data_type) {
  case DataType::FLOAT: {
    float v = static_cast<float>(value);
    std::memcpy(base + static_cast<std::size_t>(idx) * sizeof(float), &v, sizeof(float));
    return;
  }
  case DataType::DOUBLE: {
    std::memcpy(base + static_cast<std::size_t>(idx) * sizeof(double), &value, sizeof(double));
    return;
  }
  default:
    EXT_THROW_INVALID("unsupported data type ", t.data_type, ", ",
                      "kernel::Resize: linear/cubic modes only support FLOAT/DOUBLE output types.");
  }
}

// 1-D linear interpolation coefficients (2 taps) for the upstream-reference
// fractional position ``ratio`` in ``[0, 1]``.
void LinearCoeffs(double ratio, double coeffs[2]) {
  coeffs[0] = 1.0 - ratio;
  coeffs[1] = ratio;
}

// 1-D cubic interpolation coefficients (4 taps). Mirrors
// ``onnx/reference/ops/op_resize.py::_cubic_coeffs``.
void CubicCoeffs(double ratio, double A, double coeffs[4]) {
  const double r1 = ratio + 1.0;
  const double r2 = ratio;
  const double r3 = 1.0 - ratio;
  const double r4 = (1.0 - ratio) + 1.0;
  coeffs[0] = ((A * r1 - 5.0 * A) * r1 + 8.0 * A) * r1 - 4.0 * A;
  coeffs[1] = ((A + 2.0) * r2 - (A + 3.0)) * r2 * r2 + 1.0;
  coeffs[2] = ((A + 2.0) * r3 - (A + 3.0)) * r3 * r3 + 1.0;
  coeffs[3] = ((A * r4 - 5.0 * A) * r4 + 8.0 * A) * r4 - 4.0 * A;
}

// Anti-aliased 1-D linear interpolation coefficients. Mirrors
// ``onnx/reference/ops/op_resize.py::_linear_coeffs_antialias``. When
// downsampling (``scale < 1``) the triangular kernel is stretched by
// ``scale`` so it spans a wider footprint, low-pass filtering the input. The
// number of taps (returned value) grows as ``scale`` shrinks; the resulting
// coefficients are written to ``coeffs`` and normalised to sum to 1.
// ``coeffs`` must have room for at least ``footprint`` (``2 - 2 * start``)
// entries; every entry in that range is written before use.
int64_t LinearCoeffsAntialias(double ratio, double scale, double *coeffs) {
  if (scale > 1.0) {
    scale = 1.0;
  }
  const int64_t start = static_cast<int64_t>(std::floor(-1.0 / scale) + 1.0);
  const int64_t footprint = 2 - 2 * start;
  double sum = 0.0;
  for (int64_t i = 0; i < footprint; ++i) {
    const double arg = (static_cast<double>(start + i) - ratio) * scale;
    double c = 1.0 - std::abs(arg);
    if (c < 0.0) {
      c = 0.0;
    } else if (c > 1.0) {
      c = 1.0;
    }
    coeffs[static_cast<std::size_t>(i)] = c;
    sum += c;
  }
  for (int64_t i = 0; i < footprint; ++i) {
    coeffs[static_cast<std::size_t>(i)] /= sum;
  }
  return footprint;
}

// Anti-aliased 1-D cubic interpolation coefficients. Mirrors
// ``onnx/reference/ops/op_resize.py::_cubic_coeffs_antialias``. Like
// :cpp:func:`LinearCoeffsAntialias`, the cubic kernel is stretched by
// ``scale`` when downsampling, yielding a variable number of taps.
// ``coeffs`` must have room for at least ``n`` (``i_end - i_start``) entries;
// every entry in that range is written before use.
int64_t CubicCoeffsAntialias(double ratio, double scale, double A, double *coeffs) {
  if (scale > 1.0) {
    scale = 1.0;
  }
  const int64_t i_start = static_cast<int64_t>(std::floor(-2.0 / scale) + 1.0);
  const int64_t i_end = 2 - i_start;
  const int64_t n = i_end - i_start;
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double x = std::abs(scale * (static_cast<double>(i_start + i) - ratio));
    const double x2 = x * x;
    const double x3 = x * x2;
    double c;
    if (x <= 1.0) {
      c = (A + 2.0) * x3 - (A + 3.0) * x2 + 1.0;
    } else if (x < 2.0) {
      c = A * x3 - 5.0 * A * x2 + 8.0 * A * x - 4.0 * A;
    } else {
      c = 0.0;
    }
    coeffs[static_cast<std::size_t>(i)] = c;
    sum += c;
  }
  for (int64_t i = 0; i < n; ++i) {
    coeffs[static_cast<std::size_t>(i)] /= sum;
  }
  return n;
}

// Reproduces ``_get_neighbor_idxes``/``_get_neighbor`` from the upstream
// reference: returns the ``n`` indices nearest to ``x`` in ``[-pad, in_dim +
// pad)`` (preferring smaller indices for ties), sorted ascending. The
// returned indices may be negative or ``>= in_dim``; callers should clamp
// them before indexing the data array, since ``_get_neighbor`` itself does
// "edge"-mode padding.
// ``out`` receives the ``n`` chosen indices (ascending). ``scratch`` is
// caller-supplied working storage with room for at least ``in_dim + 2 *
// ceil(n / 2)`` entries.
void NeighborIndices(double x, int64_t n, int64_t in_dim, int64_t *out, int64_t *scratch) {
  const int64_t pad_width = (n + 1) / 2; // ceil(n / 2)
  const int64_t limit = in_dim + 2 * pad_width;
  const double shifted = x + static_cast<double>(pad_width);
  // Equivalent to: sorted(range(limit), key=lambda i: (abs(shifted - i), i))[:n]
  // For our use cases, ``n`` is 2 or 4, so a small partial-sort suffices.
  for (int64_t i = 0; i < limit; ++i) {
    scratch[static_cast<std::size_t>(i)] = i;
  }
  std::partial_sort(scratch, scratch + n, scratch + limit, [shifted](int64_t a, int64_t b) {
    const double da = std::abs(shifted - static_cast<double>(a));
    const double db = std::abs(shifted - static_cast<double>(b));
    if (da != db) {
      return da < db;
    }
    return a < b;
  });
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = scratch[static_cast<std::size_t>(i)];
  }
  std::sort(out, out + n);
  // Map back to the original (unpadded) index space.
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] -= pad_width;
  }
}

// Interpolates ``data`` (a contiguous 1-D buffer of ``in_dim`` doubles) at
// output position ``out_coord``, using the coordinate transformation ``mode``
// and the per-mode coefficient generator (linear: 2 taps; cubic: 4 taps).
// Mirrors ``_interpolate_1d_with_x`` from the upstream Python reference.
//
// ``is_cubic`` selects which coefficient generator (and tap count) to use.
// A boolean is passed instead of re-checking ``interp_mode`` here because
// this function runs in the inner resize loop.
double Interpolate1D(const double *data, int64_t in_dim, int64_t out_dim, int64_t out_coord,
                     double scale, const std::string &coord_mode, double roi_start, double roi_end,
                     bool is_cubic, double cubic_a, bool exclude_outside, bool antialias,
                     bool use_extrapolation, double extrapolation_value, int64_t *idx_scratch,
                     double *coeff_scratch, int64_t *neighbor_scratch) {
  const double x_ori =
      TransformCoord(out_coord, in_dim, out_dim, scale, coord_mode, roi_start, roi_end);
  if (use_extrapolation && (x_ori < 0.0 || x_ori > static_cast<double>(in_dim - 1))) {
    return extrapolation_value;
  }
  const double x_ori_floor = std::floor(x_ori);
  const bool is_integer = (x_ori - x_ori_floor) == 0.0;
  // When ``x_ori`` is an integer the upstream reference forces ``ratio = 1``
  // (see ``onnx/reference/ops/op_resize.py::_interpolate_1d_with_x``). This
  // pairs with :cpp:func:`NeighborIndices`, which prefers indices smaller
  // than ``x_ori`` for ties: for linear (2 taps) the chosen neighbours are
  // ``[x_ori - 1, x_ori]`` with coefficients ``[0, 1]``, recovering
  // ``data[x_ori]`` exactly. Keeping this convention is required for
  // bit-exact agreement with the upstream Python reference.
  double ratio;
  if (is_integer) {
    ratio = 1.0;
  } else {
    ratio = x_ori - x_ori_floor;
  }

  // ``coeff_scratch`` holds the (possibly variable-length, antialias) taps;
  // ``fixed`` holds the 2-tap linear / 4-tap cubic coefficients used in the
  // non-antialias path, avoiding a heap allocation per element.
  double fixed[4];
  const double *coeffs;
  int64_t n;
  if (antialias) {
    if (is_cubic) {
      n = CubicCoeffsAntialias(ratio, scale, cubic_a, coeff_scratch);
    } else {
      n = LinearCoeffsAntialias(ratio, scale, coeff_scratch);
    }
    coeffs = coeff_scratch;
  } else {
    if (is_cubic) {
      n = 4;
      CubicCoeffs(ratio, cubic_a, fixed);
    } else {
      n = 2;
      LinearCoeffs(ratio, fixed);
    }
    coeffs = fixed;
  }

  NeighborIndices(x_ori, n, in_dim, idx_scratch, neighbor_scratch);

  if (exclude_outside) {
    // ``exclude_outside`` renormalisation must mutate the coefficients, so
    // ensure they live in the (writable) scratch buffer first.
    if (coeffs != coeff_scratch) {
      for (int64_t i = 0; i < n; ++i) {
        coeff_scratch[static_cast<std::size_t>(i)] = coeffs[i];
      }
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      if (idx_scratch[static_cast<std::size_t>(i)] < 0 ||
          idx_scratch[static_cast<std::size_t>(i)] >= in_dim) {
        coeff_scratch[static_cast<std::size_t>(i)] = 0.0;
      }
      sum += coeff_scratch[static_cast<std::size_t>(i)];
    }
    if (sum != 0.0) {
      for (int64_t i = 0; i < n; ++i) {
        coeff_scratch[static_cast<std::size_t>(i)] /= sum;
      }
    }
    coeffs = coeff_scratch;
  }

  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    int64_t idx = idx_scratch[static_cast<std::size_t>(i)];
    // ``_get_neighbor`` returns edge-padded values for out-of-range indices.
    if (idx < 0) {
      idx = 0;
    } else if (idx >= in_dim) {
      idx = in_dim - 1;
    }
    acc += coeffs[i] * data[static_cast<std::size_t>(idx)];
  }
  return acc;
}

// Separable per-axis resize for ``"linear"`` and ``"cubic"`` modes. Processes
// each axis independently using :cpp:func:`Interpolate1D`, writing
// intermediate results into a scratch buffer. The result matches the
// upstream reference's fully-nested 1-D interpolation because the operation
// is a tensor product of per-axis linear combinations.
void ResizeSeparable(const Tensor &input, const float *scales, const onnx_kernels::Shape &out_shape,
                     const std::string &coord_mode, const double *roi_start, const double *roi_end,
                     const std::string &interp_mode, double cubic_a, bool exclude_outside,
                     bool antialias, double extrapolation_value, Tensor &output,
                     RawBufferAllocator *allocator) {
  const std::size_t rank = out_shape.size();
  EXT_ENFORCE_INVALID(input.shape.size() == rank,
                      "kernel::Resize: input rank must equal output rank.");
  // Resolve the interpolation mode once, outside the inner loop, so
  // :cpp:func:`Interpolate1D` can dispatch via a boolean without re-parsing
  // the attribute string for every output element.
  const bool is_cubic = IsCubicMode(interp_mode);
  const bool use_extrapolation = coord_mode == "tf_crop_and_resize";

  // Bound on the working-buffer element count across every axis pass: each
  // pass only rewrites a single dimension, so ``prod max(in, out)`` is an
  // upper bound for every intermediate shape. ``max_in_dim`` bounds the
  // per-axis gathered line and neighbour scratch.
  int64_t bound_elems = 1;
  int64_t max_in_dim = 0;
  for (std::size_t k = 0; k < rank; ++k) {
    const int64_t in_d = input.shape[k];
    const int64_t out_d = out_shape[k];
    bound_elems *= std::max<int64_t>(in_d, out_d);
    max_in_dim = std::max<int64_t>(max_in_dim, in_d);
  }

  // Two ping-pong buffers hold the double-precision working data. We start
  // from a double-precision copy of the input, then interpolate axis-by-axis,
  // swapping ``cur``/``next`` at each step.
  detail::TemporaryTypedBuffer<double> cur_buf(static_cast<std::size_t>(bound_elems), allocator,
                                               "kernel::Resize separable cur");
  detail::TemporaryTypedBuffer<double> next_buf(static_cast<std::size_t>(bound_elems), allocator,
                                                "kernel::Resize separable next");
  double *cur = cur_buf.data();
  double *next = next_buf.data();

  onnx_kernels::Shape cur_shape = input.shape;
  int64_t cur_elems = 1;
  for (int64_t d : cur_shape) {
    cur_elems *= d;
  }
  for (int64_t i = 0; i < cur_elems; ++i) {
    cur[static_cast<std::size_t>(i)] = LoadFloat(input, i);
  }

  // The interpolation kernel uses 2 taps (linear) or 4 taps (cubic); when
  // anti-aliasing while downsampling, the kernel is stretched so the tap
  // count grows. ``coeff_cap`` bounds this across every axis so the scratch
  // buffers can be sized once.
  std::size_t coeff_cap = 4;
  if (antialias) {
    const double kernel_c = is_cubic ? 2.0 : 1.0;
    for (std::size_t k = 0; k < rank; ++k) {
      double s = static_cast<double>(scales[k]);
      if (s > 1.0) {
        s = 1.0;
      }
      if (s <= 0.0) {
        continue;
      }
      const int64_t start = static_cast<int64_t>(std::floor(-kernel_c / s) + 1.0);
      const int64_t cnt = 2 - 2 * start;
      if (cnt > 0 && static_cast<std::size_t>(cnt) > coeff_cap) {
        coeff_cap = static_cast<std::size_t>(cnt);
      }
    }
  }
  // ``NeighborIndices`` needs ``in_dim + 2 * ceil(n / 2)`` scratch entries.
  const std::size_t pad_cap = (coeff_cap + 1) / 2;
  const std::size_t neighbor_cap = static_cast<std::size_t>(max_in_dim) + 2 * pad_cap + 1;

  detail::TemporaryTypedBuffer<int64_t> idx_scratch(coeff_cap, allocator, "kernel::Resize idx");
  detail::TemporaryTypedBuffer<double> coeff_scratch(coeff_cap, allocator, "kernel::Resize coeff");
  detail::TemporaryTypedBuffer<int64_t> neighbor_scratch(neighbor_cap, allocator,
                                                         "kernel::Resize neighbor");
  detail::TemporaryTypedBuffer<double> line_buf(
      static_cast<std::size_t>(std::max<int64_t>(max_in_dim, 1)), allocator, "kernel::Resize line");
  double *line = line_buf.data();

  for (std::size_t axis = 0; axis < rank; ++axis) {
    const int64_t in_dim = cur_shape[axis];
    const int64_t out_dim = out_shape[axis];
    if (in_dim == out_dim && static_cast<double>(scales[axis]) == 1.0) {
      // Skip axes that are not actually being resized.
      continue;
    }
    // Compute outer/inner stride sizes around ``axis``.
    int64_t outer = 1;
    for (std::size_t k = 0; k < axis; ++k) {
      outer *= cur_shape[k];
    }
    int64_t inner = 1;
    for (std::size_t k = axis + 1; k < rank; ++k) {
      inner *= cur_shape[k];
    }
    onnx_kernels::Shape new_shape = cur_shape;
    new_shape[axis] = out_dim;

    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t in = 0; in < inner; ++in) {
        // Gather the 1-D line along ``axis`` at outer position ``o`` and
        // inner position ``in``.
        for (int64_t k = 0; k < in_dim; ++k) {
          line[static_cast<std::size_t>(k)] =
              cur[static_cast<std::size_t>((o * in_dim + k) * inner + in)];
        }
        for (int64_t k = 0; k < out_dim; ++k) {
          const double v =
              Interpolate1D(line, in_dim, out_dim, k, static_cast<double>(scales[axis]), coord_mode,
                            roi_start[axis], roi_end[axis], is_cubic, cubic_a, exclude_outside,
                            antialias, use_extrapolation, extrapolation_value, idx_scratch.data(),
                            coeff_scratch.data(), neighbor_scratch.data());
          next[static_cast<std::size_t>((o * out_dim + k) * inner + in)] = v;
        }
      }
    }
    std::swap(cur, next);
    cur_shape = new_shape;
  }

  EXT_ENFORCE_INVALID(cur_shape == out_shape,
                      "kernel::Resize: separable resize produced an unexpected shape.");
  int64_t final_elems = 1;
  for (int64_t d : cur_shape) {
    final_elems *= d;
  }
  for (int64_t i = 0; i < final_elems; ++i) {
    StoreFloat(output, i, cur[static_cast<std::size_t>(i)]);
  }
}

// Applies ``keep_aspect_ratio_policy`` to a per-axis ``sizes`` request and
// writes the effective target output size for each resized axis into ``out``
// (``requested_len`` entries). ``requested_sizes`` points to ``requested_len``
// elements.
void ApplyKeepAspectRatioPolicy(const int64_t *requested_sizes, std::size_t requested_len,
                                const onnx_kernels::Shape &in_sizes, const std::string &policy,
                                int64_t *out) {
  if (policy == "stretch") {
    for (std::size_t i = 0; i < requested_len; ++i) {
      out[i] = requested_sizes[i];
    }
    return;
  }
  EXT_ENFORCE_INVALID(requested_len == in_sizes.size(),
                      "kernel::Resize: 'sizes' length must match the number of resized axes.");
  // Pick a single scale factor that satisfies the policy across every axis.
  // ``not_larger``: output dim <= sizes[i] for all i (use the minimum ratio).
  // ``not_smaller``: output dim >= sizes[i] for all i (use the maximum ratio).
  double picked = 0.0;
  bool first = true;
  for (std::size_t i = 0; i < requested_len; ++i) {
    EXT_ENFORCE_INVALID(in_sizes[i] > 0,
                        "kernel::Resize: input dim must be > 0 when using 'sizes'.");
    const double ratio = static_cast<double>(requested_sizes[i]) / static_cast<double>(in_sizes[i]);
    if (first) {
      picked = ratio;
      first = false;
    } else if (policy == "not_larger") {
      picked = std::min(picked, ratio);
    } else if (policy == "not_smaller") {
      picked = std::max(picked, ratio);
    } else {
      EXT_THROW_INVALID("kernel::Resize: unsupported keep_aspect_ratio_policy '", policy, "'.");
    }
  }
  for (std::size_t i = 0; i < requested_len; ++i) {
    // ONNX spec: round(sizes[i] * in_sizes[i] / sizes[i]) ... effectively
    // round(picked * in_sizes[i]) since picked is the chosen common ratio.
    out[i] = static_cast<int64_t>(std::llround(picked * static_cast<double>(in_sizes[i])));
    if (out[i] < 1) {
      out[i] = 1;
    }
  }
}

void CheckSupportedAttrs(const Resize::Attributes &attrs) {
  EXT_ENFORCE_INVALID(IsNearestMode(attrs.mode) || IsLinearMode(attrs.mode) ||
                          IsCubicMode(attrs.mode),
                      "kernel::Resize: unsupported interpolation mode '", attrs.mode,
                      "'. Supported modes: 'nearest', 'linear'/'bilinear', 'cubic'.");
}

// Builds per-axis ``roi_start``/``roi_end`` vectors (length ``rank``) from
// the user-supplied ``attrs.roi`` and ``axes``. For axes not listed in
// ``axes`` (or when no ROI was provided), defaults to the full
// ``[0.0, 1.0]`` range so that the ``"tf_crop_and_resize"`` formula
// reduces to identity.
void BuildRoi(const Resize::Attributes &attrs, const onnx_kernels::Shape &axes, std::size_t rank,
              double *roi_start, double *roi_end) {
  for (std::size_t i = 0; i < rank; ++i) {
    roi_start[i] = 0.0;
    roi_end[i] = 1.0;
  }
  if (attrs.coordinate_transformation_mode != "tf_crop_and_resize") {
    return;
  }
  EXT_ENFORCE_INVALID(attrs.roi.size() == 2 * axes.size(),
                      "kernel::Resize: 'roi' length must equal 2 * number of resized axes for "
                      "'tf_crop_and_resize' coordinate_transformation_mode; got roi length ",
                      attrs.roi.size(), ", expected ", 2 * axes.size(), ".");
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t k = static_cast<std::size_t>(axes[i]);
    roi_start[k] = static_cast<double>(attrs.roi[i]);
    roi_end[k] = static_cast<double>(attrs.roi[i + axes.size()]);
  }
}

// Dispatches to the nearest or separable (linear/cubic) implementation
// according to ``attrs.mode``.
void RunResize(const Tensor &X, const float *scales_data, const onnx_kernels::Shape &out_shape,
               const double *roi_start, const double *roi_end, const Resize::Attributes &attrs,
               Tensor &output, RawBufferAllocator *allocator) {
  if (IsNearestMode(attrs.mode)) {
    ResizeNearest(X, scales_data, out_shape, attrs.nearest_mode,
                  attrs.coordinate_transformation_mode, roi_start, roi_end,
                  static_cast<double>(attrs.extrapolation_value), output);
    return;
  }
  ResizeSeparable(X, scales_data, out_shape, attrs.coordinate_transformation_mode, roi_start,
                  roi_end, attrs.mode, static_cast<double>(attrs.cubic_coeff_a),
                  attrs.exclude_outside != 0, attrs.antialias != 0,
                  static_cast<double>(attrs.extrapolation_value), output, allocator);
}

// Builds the ROI scratch and runs the resize into the preallocated
// ``output``. ``allocator`` (when non-null) backs every scratch buffer.
void ComputeResizeFromScales(const Tensor &X, const float *scales_data,
                             const onnx_kernels::Shape &axes, const Resize::Attributes &attrs,
                             Tensor &output, RawBufferAllocator *allocator) {
  const std::size_t rank = X.shape.size();
  detail::TemporaryTypedBuffer<double> roi_start_buf(rank, allocator, "kernel::Resize roi_start");
  detail::TemporaryTypedBuffer<double> roi_end_buf(rank, allocator, "kernel::Resize roi_end");
  BuildRoi(attrs, axes, rank, roi_start_buf.data(), roi_end_buf.data());
  RunResize(X, scales_data, output.shape, roi_start_buf.data(), roi_end_buf.data(), attrs, output,
            allocator);
}

} // namespace

Tensor Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                          RuntimeContext *rt) const {
  CheckSupportedAttrs(attrs);
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  const std::size_t rank = X.shape.size();
  const onnx_kernels::Shape axes = NormaliseAxes(attrs.axes, rank);
  const Tensor scales_in = ReadResizeScales(scales, axes.size(), allocator);
  // Expand to per-axis (rank-length) scales, defaulting non-resized axes to 1.
  detail::TemporaryTypedBuffer<float> scales_buf(rank, allocator, "kernel::Resize scales");
  float *scales_data = scales_buf.data();
  ScatterByAxes<float>(scales_in.As<float>(), axes, rank, 1.0f, scales_data);
  onnx_kernels::Shape out_shape;
  out_shape.assign(rank, 0);
  for (std::size_t k = 0; k < rank; ++k) {
    const double scaled = static_cast<double>(X.shape[k]) * static_cast<double>(scales_data[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  const size_t output_n_bytes = PackedByteSize(X.data_type, total_elements);
  Tensor output = MakeOutputTensor(X.data_type, out_shape, output_n_bytes, allocator);
  ComputeResizeFromScales(X, scales_data, axes, attrs, output, allocator);
  return output;
}

void Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                        Tensor &output) const {
  CheckSupportedAttrs(attrs);
  const std::size_t rank = X.shape.size();
  const onnx_kernels::Shape axes = NormaliseAxes(attrs.axes, rank);
  const Tensor scales_in = ReadResizeScales(scales, axes.size(), nullptr);
  detail::TemporaryTypedBuffer<float> scales_buf(rank, nullptr, "kernel::Resize scales");
  float *scales_data = scales_buf.data();
  ScatterByAxes<float>(scales_in.As<float>(), axes, rank, 1.0f, scales_data);
  onnx_kernels::Shape out_shape;
  out_shape.assign(rank, 0);
  for (std::size_t k = 0; k < rank; ++k) {
    const double scaled = static_cast<double>(X.shape[k]) * static_cast<double>(scales_data[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }

  EXT_ENFORCE_INVALID(output.data_type == X.data_type,
                      "kernel::Resize: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Resize: preallocated output shape mismatch.");

  ComputeResizeFromScales(X, scales_data, axes, attrs, output, nullptr);
}

Tensor Resize::ResizeSizes(const Tensor &X, const Tensor &sizes, const Attributes &attrs,
                           RuntimeContext *rt) const {
  CheckSupportedAttrs(attrs);
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  const std::size_t rank = X.shape.size();
  const onnx_kernels::Shape axes = NormaliseAxes(attrs.axes, rank);
  const Tensor requested = ReadResizeSizes(sizes, axes.size(), allocator);
  // Per-axis input shape restricted to the resized axes, used when computing
  // the effective output sizes under ``keep_aspect_ratio_policy``.
  onnx_kernels::Shape in_axes_shape;
  in_axes_shape.assign(axes.size(), 0);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    in_axes_shape[i] = X.shape[static_cast<std::size_t>(axes[i])];
  }
  detail::TemporaryTypedBuffer<int64_t> effective_buf(axes.size(), allocator,
                                                      "kernel::Resize effective");
  int64_t *effective = effective_buf.data();
  ApplyKeepAspectRatioPolicy(requested.As<int64_t>(), axes.size(), in_axes_shape,
                             attrs.keep_aspect_ratio_policy, effective);
  // Build the full output shape, leaving non-resized axes untouched.
  onnx_kernels::Shape out_shape = X.shape;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    out_shape[static_cast<std::size_t>(axes[i])] = effective[i];
  }
  // Derive per-axis scales from the effective output sizes so we can reuse
  // the nearest path.
  detail::TemporaryTypedBuffer<float> scales_buf(rank, allocator, "kernel::Resize scales");
  float *scales_data = scales_buf.data();
  for (std::size_t k = 0; k < rank; ++k) {
    scales_data[k] = 1.0f;
  }
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t k = static_cast<std::size_t>(axes[i]);
    EXT_ENFORCE_INVALID(X.shape[k] > 0,
                        "kernel::Resize: input dim must be > 0 when using 'sizes'.");
    scales_data[k] = static_cast<float>(effective[i]) / static_cast<float>(X.shape[k]);
  }
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  const size_t output_n_bytes = PackedByteSize(X.data_type, total_elements);
  Tensor output = MakeOutputTensor(X.data_type, out_shape, output_n_bytes, allocator);
  ComputeResizeFromScales(X, scales_data, axes, attrs, output, allocator);
  return output;
}

void Resize::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 1 || node.input_size() > 4), "RunNode: op '",
                      node.op_type(), "' expects between 1 and 4 input(s), got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor *roi = GetOptionalInput(node, 1, rt.tensors());
  const Tensor *scales = GetOptionalInput(node, 2, rt.tensors());
  const Tensor *sizes = GetOptionalInput(node, 3, rt.tensors());
  EXT_ENFORCE_INVALID(!((scales == nullptr) == (sizes == nullptr)),
                      "RunNode: op 'Resize' requires exactly one of 'scales' or 'sizes' to be "
                      "provided.");

  onnx_kernels::kernel::Resize::Attributes attrs;
  attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
  attrs.coordinate_transformation_mode = GetAttributeStringOrDefault(
      node, "coordinate_transformation_mode", attrs.coordinate_transformation_mode);
  attrs.nearest_mode = GetAttributeStringOrDefault(node, "nearest_mode", attrs.nearest_mode);
  attrs.axes = GetAttributeShapeOrDefault(node, "axes", attrs.axes);
  attrs.keep_aspect_ratio_policy =
      GetAttributeStringOrDefault(node, "keep_aspect_ratio_policy", attrs.keep_aspect_ratio_policy);
  attrs.cubic_coeff_a = GetAttributeFloatOrDefault(node, "cubic_coeff_a", attrs.cubic_coeff_a);
  attrs.exclude_outside = GetAttributeIntOrDefault(node, "exclude_outside", attrs.exclude_outside);
  attrs.antialias = GetAttributeIntOrDefault(node, "antialias", attrs.antialias);
  attrs.extrapolation_value =
      GetAttributeFloatOrDefault(node, "extrapolation_value", attrs.extrapolation_value);
  if (roi != nullptr) {
    EXT_ENFORCE_INVALID(!(roi->data_type != DataType::FLOAT),
                        "RunNode: op 'Resize' 'roi' input must be a FLOAT tensor.");
    EXT_ENFORCE_INVALID(!(roi->shape.size() != 1), "RunNode: op 'Resize' 'roi' input must be 1-D.");
    const int64_t n = roi->shape[0];
    attrs.roi.assign(static_cast<std::size_t>(n), 0.0f);
    if (n > 0) {
      std::memcpy(attrs.roi.data(), roi->bytes(), static_cast<std::size_t>(n) * sizeof(float));
    }
  }

  onnx_kernels::kernel::Resize k(rt.kernel_ctx());
  if (scales != nullptr) {
    SetOutput(node, 0, k(x, *scales, attrs, &rt), rt);
  } else {
    SetOutput(node, 0, k.ResizeSizes(x, *sizes, attrs), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
