// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/quantization/include_quantization_kernels.h"
#include "onnx_backend_test/kernels/tensor/cast_float8.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

inline void RequireScalar(const Tensor &t, const char *name) {
  // A scalar is either a 0-D tensor (shape == {}) or a 1-D tensor with a
  // single element. Both are accepted for the per-tensor case to mirror
  // QuantizeLinear's behaviour.
  const int64_t n = t.element_count();
  EXT_ENFORCE_INVALID(n == 1, std::string("kernel::DequantizeLinear: ") + name +
                                  " must be a scalar (per-tensor dequantization).");
}

template <typename XT>
void DequantizeLoop(const Tensor &x, float x_scale, XT x_zero_point, Tensor &output) {
  const XT *px = reinterpret_cast<const XT *>(x.data.data());
  float *py = output.AsFloat();
  const int64_t n = x.element_count();
  const float zp = static_cast<float>(x_zero_point);
  for (int64_t i = 0; i < n; ++i) {
    py[i] = (static_cast<float>(px[i]) - zp) * x_scale;
  }
}

// Dispatch table for float8 → float32 bit-level conversion. Each entry
// matches one of the four ONNX float8 element types and points at the
// saturating ``Float8*BitsToFloat`` decoder declared in ``cast_float8.h``.
using Float8Decoder = float (*)(std::uint8_t) noexcept;

inline Float8Decoder Float8DecoderFor(int32_t dtype) noexcept {
  switch (dtype) {
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
    return &Float8E4M3FNBitsToFloat;
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
    return &Float8E4M3FNUZBitsToFloat;
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
    return &Float8E5M2BitsToFloat;
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ):
    return &Float8E5M2FNUZBitsToFloat;
  default:
    return nullptr;
  }
}

inline void DequantizeFloat8Loop(const Tensor &x, float x_scale, float x_zero_point,
                                 Float8Decoder decode, Tensor &output) {
  const std::uint8_t *px = x.data.data();
  float *py = output.AsFloat();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    py[i] = (decode(px[i]) - x_zero_point) * x_scale;
  }
}

} // namespace

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale) const {
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, x_scale, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale, Tensor &output) const {
  EXT_ENFORCE_INVALID(x_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: x_scale must be FLOAT.");
  RequireScalar(x_scale, "x_scale");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: output (no x_zero_point) must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = x_scale.AsFloat()[0];
  switch (x.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    DequantizeLoop<uint8_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    DequantizeLoop<int8_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::UINT16):
    DequantizeLoop<uint16_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::INT16):
    DequantizeLoop<int16_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::INT32):
    DequantizeLoop<int32_t>(x, scale, /*x_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ):
    DequantizeFloat8Loop(x, scale, /*x_zero_point=*/0.0f, Float8DecoderFor(x.data_type), output);
    break;
  default:
    throw std::invalid_argument("kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16, "
                                "INT32, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2 and "
                                "FLOAT8E5M2FNUZ inputs are supported.");
  }
}

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                    const Tensor &x_zero_point) const {
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, x_scale, x_zero_point, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                  const Tensor &x_zero_point, Tensor &output) const {
  EXT_ENFORCE_INVALID(x_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: x_scale must be FLOAT.");
  RequireScalar(x_scale, "x_scale");
  RequireScalar(x_zero_point, "x_zero_point");
  EXT_ENFORCE_INVALID(x.data_type == x_zero_point.data_type,
                      "kernel::DequantizeLinear: x_zero_point data_type must match x.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = x_scale.AsFloat()[0];
  switch (x.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    DequantizeLoop<uint8_t>(x, scale, static_cast<uint8_t>(x_zero_point.data[0]), output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    DequantizeLoop<int8_t>(x, scale, static_cast<int8_t>(x_zero_point.data[0]), output);
    break;
  case static_cast<int32_t>(DataType::UINT16): {
    uint16_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(uint16_t));
    DequantizeLoop<uint16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT16): {
    int16_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(int16_t));
    DequantizeLoop<int16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT32): {
    int32_t zp;
    std::memcpy(&zp, x_zero_point.data.data(), sizeof(int32_t));
    DequantizeLoop<int32_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ): {
    const Float8Decoder decode = Float8DecoderFor(x.data_type);
    const float zp = decode(x_zero_point.data[0]);
    DequantizeFloat8Loop(x, scale, zp, decode, output);
    break;
  }
  default:
    throw std::invalid_argument("kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16, "
                                "INT32, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2 and "
                                "FLOAT8E5M2FNUZ inputs are supported.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
