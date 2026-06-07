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

// Reads the 1-D FLOAT ``scales`` input tensor. ``expected_length`` is the
// number of axes the kernel expects scales for (``rank`` when ``axes`` is
// absent, ``axes.size()`` otherwise).
std::vector<float> ReadResizeScales(const Tensor &scales, std::size_t expected_length) {
  EXT_ENFORCE_INVALID(scales.data_type == DataType::FLOAT,
                      "kernel::Resize: 'scales' input must be FLOAT.");
  EXT_ENFORCE_INVALID(scales.shape.size() == 1,
                      "kernel::Resize: 'scales' input must be a 1-D tensor.");
  const int64_t n = scales.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == expected_length,
                      "kernel::Resize: 'scales' length must match the number of resized axes.");
  std::vector<float> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), scales.bytes(), static_cast<std::size_t>(n) * sizeof(float));
  }
  for (float s : out) {
    EXT_ENFORCE_INVALID(s > 0.0f, "kernel::Resize: 'scales' values must be > 0.");
  }
  return out;
}

// Reads the 1-D INT64 ``sizes`` input tensor.
std::vector<int64_t> ReadResizeSizes(const Tensor &sizes, std::size_t expected_length) {
  EXT_ENFORCE_INVALID(sizes.data_type == DataType::INT64,
                      "kernel::Resize: 'sizes' input must be INT64.");
  EXT_ENFORCE_INVALID(sizes.shape.size() == 1,
                      "kernel::Resize: 'sizes' input must be a 1-D tensor.");
  const int64_t n = sizes.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == expected_length,
                      "kernel::Resize: 'sizes' length must match the number of resized axes.");
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), sizes.bytes(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  for (int64_t s : out) {
    EXT_ENFORCE_INVALID(s > 0, "kernel::Resize: 'sizes' values must be > 0.");
  }
  return out;
}

// Normalises the user-supplied ``axes`` attribute against ``rank``. When the
// attribute is empty, returns ``{0, 1, ..., rank-1}``.
std::vector<int64_t> NormaliseAxes(const std::vector<int64_t> &axes, std::size_t rank) {
  std::vector<int64_t> out;
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
// inserting ``identity`` values on non-resized axes.
template <typename T>
std::vector<T> ScatterByAxes(const std::vector<T> &values, const std::vector<int64_t> &axes,
                             std::size_t rank, T identity) {
  std::vector<T> out(rank, identity);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    out[static_cast<std::size_t>(axes[i])] = values[i];
  }
  return out;
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
// ``onnx/reference/ops/op_resize.py`` for the modes that do not require the
// ``roi`` input.
double TransformCoord(int64_t out_coord, int64_t in_dim, int64_t out_dim, double scale,
                      const std::string &mode) {
  const double x = static_cast<double>(out_coord);
  if (mode == "asymmetric") {
    return x / scale;
  }
  if (mode == "align_corners") {
    if (out_dim == 1) {
      return 0.0;
    }
    return x * static_cast<double>(in_dim - 1) / static_cast<double>(out_dim - 1);
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
  // Default: "half_pixel" (the schema default since opset 13).
  EXT_ENFORCE_INVALID(mode == "half_pixel",
                      "kernel::Resize: unsupported coordinate_transformation_mode.");
  return (x + 0.5) / scale - 0.5;
}

// Nearest-neighbor resize for any rank, byte-element-wise copy. Combines the
// per-axis ``scales`` and ``coordinate_transformation_mode`` to compute the
// real-valued input coordinate, then rounds it via ``nearest_mode`` and
// clamps the result to ``[0, in_dim - 1]``.
void ResizeNearest(const Tensor &input, const std::vector<float> &scales,
                   const std::vector<int64_t> &out_shape, const std::string &nearest_mode,
                   const std::string &coord_mode, Tensor &output) {
  const std::size_t elem_size = ElementSize(input.data_type);
  const std::size_t rank = out_shape.size();

  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }

  std::vector<int64_t> in_strides(rank, 0);
  std::vector<int64_t> out_strides(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
      out_strides[k - 1] = out_strides[k] * out_shape[k];
    }
  }

  const uint8_t *const in_ptr = input.bytes();
  uint8_t *const out_ptr = output.data.data();

  for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t out_coord = remaining / out_strides[k];
      remaining %= out_strides[k];
      const double x_ori = TransformCoord(out_coord, input.shape[k], out_shape[k],
                                          static_cast<double>(scales[k]), coord_mode);
      int64_t in_coord = ApplyNearestMode(x_ori, nearest_mode);
      if (in_coord >= input.shape[k]) {
        in_coord = input.shape[k] - 1;
      }
      if (in_coord < 0) {
        in_coord = 0;
      }
      in_idx += in_coord * in_strides[k];
    }
    std::memcpy(out_ptr + static_cast<std::size_t>(out_idx) * elem_size,
                in_ptr + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

bool IsNearestMode(const std::string &mode) { return mode == "nearest"; }

// Applies ``keep_aspect_ratio_policy`` to a per-axis ``sizes`` request and
// returns the effective target output size for each resized axis.
std::vector<int64_t> ApplyKeepAspectRatioPolicy(const std::vector<int64_t> &requested_sizes,
                                                const std::vector<int64_t> &in_sizes,
                                                const std::string &policy) {
  if (policy == "stretch") {
    return requested_sizes;
  }
  EXT_ENFORCE_INVALID(requested_sizes.size() == in_sizes.size(),
                      "kernel::Resize: 'sizes' length must match the number of resized axes.");
  // Pick a single scale factor that satisfies the policy across every axis.
  // ``not_larger``: output dim <= sizes[i] for all i (use the minimum ratio).
  // ``not_smaller``: output dim >= sizes[i] for all i (use the maximum ratio).
  double picked = 0.0;
  bool first = true;
  for (std::size_t i = 0; i < requested_sizes.size(); ++i) {
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
      throw std::invalid_argument("kernel::Resize: unsupported keep_aspect_ratio_policy '" +
                                  policy + "'.");
    }
  }
  std::vector<int64_t> out(requested_sizes.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    // ONNX spec: round(sizes[i] * in_sizes[i] / sizes[i]) ... effectively
    // round(picked * in_sizes[i]) since picked is the chosen common ratio.
    out[i] = static_cast<int64_t>(std::llround(picked * static_cast<double>(in_sizes[i])));
    if (out[i] < 1) {
      out[i] = 1;
    }
  }
  return out;
}

void CheckSupportedAttrs(const Resize::Attributes &attrs) {
  if (!IsNearestMode(attrs.mode)) {
    throw std::invalid_argument("kernel::Resize: only 'nearest' interpolation mode is supported "
                                "in this reference implementation; got '" +
                                attrs.mode + "'.");
  }
  if (attrs.coordinate_transformation_mode == "tf_crop_and_resize") {
    throw std::invalid_argument(
        "kernel::Resize: 'tf_crop_and_resize' coordinate_transformation_mode is not supported "
        "in this reference implementation.");
  }
}

} // namespace

Tensor Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs) const {
  const std::size_t rank = X.shape.size();
  const std::vector<int64_t> axes = NormaliseAxes(attrs.axes, rank);
  const std::vector<float> scales_in = ReadResizeScales(scales, axes.size());
  // Expand to per-axis (rank-length) scales, defaulting non-resized axes to 1.
  const std::vector<float> scales_vec = ScatterByAxes<float>(scales_in, axes, rank, 1.0f);
  std::vector<int64_t> out_shape(rank);
  for (std::size_t k = 0; k < rank; ++k) {
    const double scaled = static_cast<double>(X.shape[k]) * static_cast<double>(scales_vec[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  Tensor output("", X.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(X.data_type, total_elements)));
  (*this)(X, scales, attrs, output);
  return output;
}

void Resize::operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                        Tensor &output) const {
  CheckSupportedAttrs(attrs);
  const std::size_t rank = X.shape.size();
  const std::vector<int64_t> axes = NormaliseAxes(attrs.axes, rank);
  const std::vector<float> scales_in = ReadResizeScales(scales, axes.size());
  const std::vector<float> scales_vec = ScatterByAxes<float>(scales_in, axes, rank, 1.0f);
  std::vector<int64_t> out_shape(rank);
  for (std::size_t k = 0; k < rank; ++k) {
    const double scaled = static_cast<double>(X.shape[k]) * static_cast<double>(scales_vec[k]);
    out_shape[k] = static_cast<int64_t>(std::floor(scaled));
  }

  EXT_ENFORCE_INVALID(output.data_type == X.data_type,
                      "kernel::Resize: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Resize: preallocated output shape mismatch.");

  ResizeNearest(X, scales_vec, out_shape, attrs.nearest_mode, attrs.coordinate_transformation_mode,
                output);
}

Tensor Resize::ResizeSizes(const Tensor &X, const Tensor &sizes, const Attributes &attrs) const {
  CheckSupportedAttrs(attrs);
  const std::size_t rank = X.shape.size();
  const std::vector<int64_t> axes = NormaliseAxes(attrs.axes, rank);
  const std::vector<int64_t> requested = ReadResizeSizes(sizes, axes.size());
  // Per-axis input shape restricted to the resized axes, used when computing
  // the effective output sizes under ``keep_aspect_ratio_policy``.
  std::vector<int64_t> in_axes_shape(axes.size());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    in_axes_shape[i] = X.shape[static_cast<std::size_t>(axes[i])];
  }
  const std::vector<int64_t> effective =
      ApplyKeepAspectRatioPolicy(requested, in_axes_shape, attrs.keep_aspect_ratio_policy);
  // Build the full output shape, leaving non-resized axes untouched.
  std::vector<int64_t> out_shape(X.shape.begin(), X.shape.end());
  for (std::size_t i = 0; i < axes.size(); ++i) {
    out_shape[static_cast<std::size_t>(axes[i])] = effective[i];
  }
  // Derive per-axis scales from the effective output sizes so we can reuse
  // the nearest path.
  std::vector<float> scales_vec(rank, 1.0f);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t k = static_cast<std::size_t>(axes[i]);
    EXT_ENFORCE_INVALID(X.shape[k] > 0,
                        "kernel::Resize: input dim must be > 0 when using 'sizes'.");
    scales_vec[k] = static_cast<float>(effective[i]) / static_cast<float>(X.shape[k]);
  }
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  Tensor output("", X.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(X.data_type, total_elements)));
  ResizeNearest(X, scales_vec, out_shape, attrs.nearest_mode, attrs.coordinate_transformation_mode,
                output);
  return output;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
