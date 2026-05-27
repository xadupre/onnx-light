// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// IEEE 754 round-half-to-even (banker's rounding), matching the rule
// specified by the ONNX QuantizeLinear operator.
inline float RoundHalfToEven(float v) {
  float rounded = std::nearbyint(v);
  return rounded;
}

inline void RequireScalar(const Tensor &t, const char *name) {
  // A scalar is either a 0-D tensor (shape == {}) or a 1-D tensor with a
  // single element. The latter is what the ONNX spec uses for per-axis
  // quantization along a degenerate axis but is also commonly produced by
  // tooling for the per-tensor case.
  const int64_t n = t.element_count();
  EXT_ENFORCE_INVALID(n == 1, std::string("kernel::QuantizeLinear: ") + name +
                                  " must be a scalar (per-tensor quantization).");
}

template <typename ZP>
void QuantizeLoop(const Tensor &x, float y_scale, ZP y_zero_point, Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.data.data());
  const int64_t n = x.element_count();
  constexpr float kMin = static_cast<float>(std::numeric_limits<ZP>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<ZP>::max());
  const float zp = static_cast<float>(y_zero_point);
  for (int64_t i = 0; i < n; ++i) {
    float v = RoundHalfToEven(px[i] / y_scale) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    py[i] = static_cast<ZP>(v);
  }
}

} // namespace

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale) const {
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::UINT8), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count())));
  (*this)(x, y_scale, out);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  RequireScalar(y_scale, "y_scale");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::UINT8),
                      "kernel::QuantizeLinear: default output (no y_zero_point) must be UINT8.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::QuantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = y_scale.AsFloat()[0];
  QuantizeLoop<uint8_t>(x, scale, /*y_zero_point=*/0, output);
}

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale,
                                  const Tensor &y_zero_point) const {
  Tensor out("", y_zero_point.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) *
                                  ElementSize(y_zero_point.data_type)));
  (*this)(x, y_scale, y_zero_point, out);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, const Tensor &y_zero_point,
                                Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  RequireScalar(y_scale, "y_scale");
  RequireScalar(y_zero_point, "y_zero_point");
  EXT_ENFORCE_INVALID(output.data_type == y_zero_point.data_type,
                      "kernel::QuantizeLinear: output data_type must match y_zero_point.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::QuantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = y_scale.AsFloat()[0];
  switch (output.data_type) {
  case static_cast<int32_t>(TensorProto::DataType::UINT8):
    QuantizeLoop<uint8_t>(x, scale, static_cast<uint8_t>(y_zero_point.data[0]), output);
    break;
  case static_cast<int32_t>(TensorProto::DataType::INT8):
    QuantizeLoop<int8_t>(x, scale, static_cast<int8_t>(y_zero_point.data[0]), output);
    break;
  default:
    throw std::invalid_argument(
        "kernel::QuantizeLinear: only UINT8 and INT8 outputs are supported.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
