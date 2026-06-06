// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Round half to even (IEEE 754), matching the rounding rule used by the
// reference ONNX ``DynamicQuantizeLinear`` implementation.
inline float RoundHalfToEven(float v) { return std::nearbyint(v); }

inline float SaturateUint8(float v) {
  if (v < 0.0f) {
    return 0.0f;
  }
  if (v > 255.0f) {
    return 255.0f;
  }
  return v;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> DynamicQuantizeLinear::operator()(const Tensor &x) const {
  Tensor y("", static_cast<int32_t>(DataType::UINT8), x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count())));
  Tensor y_scale("", static_cast<int32_t>(DataType::FLOAT), /*shape=*/{},
                 std::vector<uint8_t>(sizeof(float), 0));
  Tensor y_zero_point("", static_cast<int32_t>(DataType::UINT8), /*shape=*/{},
                      std::vector<uint8_t>(1, 0));
  (*this)(x, y, y_scale, y_zero_point);
  return std::tuple<Tensor, Tensor, Tensor>(std::move(y), std::move(y_scale),
                                            std::move(y_zero_point));
}

void DynamicQuantizeLinear::operator()(const Tensor &x, Tensor &y, Tensor &y_scale,
                                       Tensor &y_zero_point) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DynamicQuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::DynamicQuantizeLinear: y must be UINT8.");
  EXT_ENFORCE_INVALID(y.shape == x.shape,
                      "kernel::DynamicQuantizeLinear: y shape must match x shape.");
  EXT_ENFORCE_INVALID(y.data.size() == static_cast<size_t>(x.element_count()),
                      "kernel::DynamicQuantizeLinear: y buffer has unexpected size.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DynamicQuantizeLinear: y_scale must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.shape.empty() && y_scale.data.size() == sizeof(float),
                      "kernel::DynamicQuantizeLinear: y_scale must be a scalar FLOAT.");
  EXT_ENFORCE_INVALID(y_zero_point.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::DynamicQuantizeLinear: y_zero_point must be UINT8.");
  EXT_ENFORCE_INVALID(y_zero_point.shape.empty() && y_zero_point.data.size() == 1,
                      "kernel::DynamicQuantizeLinear: y_zero_point must be a scalar UINT8.");

  const int64_t n = x.element_count();
  EXT_ENFORCE_INVALID(n > 0, "kernel::DynamicQuantizeLinear: x must be non-empty.");

  const float *px = x.AsFloat();
  // Range adjusted to include 0 (matches the spec: maximum(0, max(x)) and
  // minimum(0, min(x))).
  float x_min = 0.0f;
  float x_max = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float v = px[i];
    if (v < x_min) {
      x_min = v;
    }
    if (v > x_max) {
      x_max = v;
    }
  }

  // qmin=0, qmax=255 for uint8; the spec divides by (qmax - qmin) = 255.
  const float scale = (x_max - x_min) / 255.0f;

  // intermediate_zero_point = qmin - x_min / scale = -x_min / scale (qmin = 0).
  // Both the Python reference (``round(...)``) and ``np.round(...)`` use
  // banker's rounding (round half to even), so we mirror that here for
  // ``y_zero_point`` as well as for ``y`` below.
  float zp_f;
  if (scale == 0.0f) {
    // All values equal (and equal to 0 after the [0, max]/[min, 0] adjustment);
    // the spec leaves this case undefined but onnxruntime / numpy reference
    // both produce y_scale=0, y_zero_point=0, y=0.
    zp_f = 0.0f;
  } else {
    zp_f = RoundHalfToEven(-x_min / scale);
  }
  zp_f = SaturateUint8(zp_f);
  const uint8_t zp = static_cast<uint8_t>(zp_f);

  // Write scalar outputs.
  float *p_scale = reinterpret_cast<float *>(y_scale.data.data());
  p_scale[0] = scale;
  y_zero_point.data[0] = zp;

  // Quantize x.
  uint8_t *py = y.data.data();
  if (scale == 0.0f) {
    std::fill(py, py + n, zp);
    return;
  }
  const float zp_float = static_cast<float>(zp);
  for (int64_t i = 0; i < n; ++i) {
    float v = RoundHalfToEven(px[i] / scale) + zp_float;
    v = SaturateUint8(v);
    py[i] = static_cast<uint8_t>(v);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
