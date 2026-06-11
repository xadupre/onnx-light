// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/tensor/cast_float8.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr int32_t kFloat32ExponentBias = 127;
constexpr int32_t kFloat16ExponentBias = 15;

// IEEE-754 binary16 ↔ binary32 conversions (saturating, round-half-to-even).
// Mirrors the local helpers in other kernels (e.g. ``kernel_mod.cc``,
// ``kernel_causal_conv_with_state.cc``) so this translation unit stays
// self-contained.
uint16_t FloatToFloat16Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t e32 = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    return static_cast<uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const int32_t e = e32 - kFloat32ExponentBias + kFloat16ExponentBias;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t m = (m32 | 0x00800000u) >> static_cast<uint32_t>(1 - e);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = m & 0x00000fffu;
    uint16_t h = static_cast<uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<uint16_t>(h + 1);
    }
    return h;
  }
  const uint32_t low = m32 & 0x1fffu;
  uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<uint16_t>(h + 1);
  }
  return h;
}

float Float16BitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1u;
  const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
  const uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      uint32_t m = mant;
      int32_t e = -1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      f = (sign << 31) | (static_cast<uint32_t>(e + kFloat32ExponentBias + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) |
        (static_cast<uint32_t>(exp - kFloat16ExponentBias + kFloat32ExponentBias) << 23) |
        (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline void RequireScalar(const Tensor &t, const char *name) {
  // A scalar is either a 0-D tensor (shape == {}) or a 1-D tensor with a
  // single element. Both are accepted for the per-tensor case to mirror
  // QuantizeLinear's behaviour.
  const int64_t n = t.element_count();
  EXT_ENFORCE_INVALID(n == 1, std::string("kernel::DequantizeLinear: ") + name +
                                  " must be a scalar (per-tensor dequantization).");
}

inline bool IsSupportedScaleDType(int32_t dtype) {
  return dtype == static_cast<int32_t>(DataType::FLOAT) ||
         dtype == static_cast<int32_t>(DataType::FLOAT16);
}

// Decodes the scalar ``x_scale`` to a float32 regardless of whether it is
// stored as FLOAT or FLOAT16.
inline float ReadScalarScale(const Tensor &x_scale) {
  if (x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)) {
    uint16_t bits;
    std::memcpy(&bits, x_scale.bytes(), sizeof(uint16_t));
    return Float16BitsToFloat(bits);
  }
  return x_scale.AsFloat()[0];
}

template <typename XT>
void DequantizeLoop(const Tensor &x, float x_scale, XT x_zero_point, Tensor &output) {
  const XT *px = reinterpret_cast<const XT *>(x.bytes());
  const int64_t n = x.element_count();
  const float zp = static_cast<float>(x_zero_point);
  if (output.data_type == static_cast<int32_t>(DataType::FLOAT16)) {
    uint16_t *py = reinterpret_cast<uint16_t *>(output.data.data());
    for (int64_t i = 0; i < n; ++i) {
      py[i] = FloatToFloat16Bits((static_cast<float>(px[i]) - zp) * x_scale);
    }
    return;
  }
  float *py = output.AsFloat();
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
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  if (output.data_type == static_cast<int32_t>(DataType::FLOAT16)) {
    uint16_t *py = reinterpret_cast<uint16_t *>(output.data.data());
    for (int64_t i = 0; i < n; ++i) {
      py[i] = FloatToFloat16Bits((decode(px[i]) - x_zero_point) * x_scale);
    }
    return;
  }
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    py[i] = (decode(px[i]) - x_zero_point) * x_scale;
  }
}

} // namespace

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale) const {
  // The output element type matches ``x_scale``'s element type (FLOAT or
  // FLOAT16). Both encodings occupy known fixed-size storage so the buffer
  // can be sized up-front.
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  Tensor out("", x_scale.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * elem_size));
  (*this)(x, x_scale, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale, Tensor &output) const {
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  RequireScalar(x_scale, "x_scale");
  EXT_ENFORCE_INVALID(
      output.data_type == x_scale.data_type,
      "kernel::DequantizeLinear: output (no x_zero_point) dtype must match x_scale.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = ReadScalarScale(x_scale);
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
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  Tensor out("", x_scale.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * elem_size));
  (*this)(x, x_scale, x_zero_point, out);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                  const Tensor &x_zero_point, Tensor &output) const {
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  RequireScalar(x_scale, "x_scale");
  RequireScalar(x_zero_point, "x_zero_point");
  EXT_ENFORCE_INVALID(x.data_type == x_zero_point.data_type,
                      "kernel::DequantizeLinear: x_zero_point data_type must match x.");
  EXT_ENFORCE_INVALID(output.data_type == x_scale.data_type,
                      "kernel::DequantizeLinear: output dtype must match x_scale.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = ReadScalarScale(x_scale);
  switch (x.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    DequantizeLoop<uint8_t>(x, scale, static_cast<uint8_t>(x_zero_point.bytes()[0]), output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    DequantizeLoop<int8_t>(x, scale, static_cast<int8_t>(x_zero_point.bytes()[0]), output);
    break;
  case static_cast<int32_t>(DataType::UINT16): {
    uint16_t zp;
    std::memcpy(&zp, x_zero_point.bytes(), sizeof(uint16_t));
    DequantizeLoop<uint16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT16): {
    int16_t zp;
    std::memcpy(&zp, x_zero_point.bytes(), sizeof(int16_t));
    DequantizeLoop<int16_t>(x, scale, zp, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT32): {
    int32_t zp;
    std::memcpy(&zp, x_zero_point.bytes(), sizeof(int32_t));
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
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
