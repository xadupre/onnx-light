// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_float8.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/cast_sub_byte.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

inline void RequireScalar(const Tensor &t, const char *name) {
  // A scalar is either a 0-D tensor (shape == {}) or a 1-D tensor with a
  // single element. Both are accepted for the per-tensor case to mirror
  // QuantizeLinear's behaviour.
  const int64_t n = t.element_count();
  EXT_ENFORCE_INVALID(n == 1, "kernel::DequantizeLinear: ", name,
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

// Dequantize INT4-packed input to float output.
void DequantizeInt4Loop(const Tensor &x, float x_scale, float zp, bool is_signed, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const std::uint8_t nibble = Read4BitElement(px, i);
    const float val = is_signed ? static_cast<float>(Int4NibbleToInt8(nibble))
                                : static_cast<float>(Uint4NibbleToUint8(nibble));
    py[i] = (val - zp) * x_scale;
  }
}

// Dequantize INT2-packed input to float output.
void DequantizeInt2Loop(const Tensor &x, float x_scale, float zp, bool is_signed, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const std::uint8_t bits = Read2BitElement(px, i);
    const float val = is_signed ? static_cast<float>(Int2BitsToInt8(bits))
                                : static_cast<float>(Uint2BitsToUint8(bits));
    py[i] = (val - zp) * x_scale;
  }
}

// Dequantize FLOAT4E2M1-packed input to float output.
void DequantizeFloat4E2M1Loop(const Tensor &x, float x_scale, float zp, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const float val = Float4E2M1NibbleToFloat(Read4BitElement(px, i));
    py[i] = (val - zp) * x_scale;
  }
}

// Reads the zero-point nibble from a sub-byte packed tensor (ZP has 1 element).
inline float ReadSubByteScalarZP(const Tensor &x_zero_point) {
  const int32_t dtype = x_zero_point.data_type;
  if (dtype == static_cast<int32_t>(DataType::INT4)) {
    return static_cast<float>(Int4NibbleToInt8(Read4BitElement(x_zero_point.bytes(), 0)));
  } else if (dtype == static_cast<int32_t>(DataType::UINT4)) {
    return static_cast<float>(Uint4NibbleToUint8(Read4BitElement(x_zero_point.bytes(), 0)));
  } else if (dtype == static_cast<int32_t>(DataType::INT2)) {
    return static_cast<float>(Int2BitsToInt8(Read2BitElement(x_zero_point.bytes(), 0)));
  } else if (dtype == static_cast<int32_t>(DataType::UINT2)) {
    return static_cast<float>(Uint2BitsToUint8(Read2BitElement(x_zero_point.bytes(), 0)));
  } else if (dtype == static_cast<int32_t>(DataType::FLOAT4E2M1)) {
    return Float4E2M1NibbleToFloat(Read4BitElement(x_zero_point.bytes(), 0));
  }
  return 0.0f;
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
  case static_cast<int32_t>(DataType::INT4):
    DequantizeInt4Loop(x, scale, /*zp=*/0.0f, /*is_signed=*/true, output);
    break;
  case static_cast<int32_t>(DataType::UINT4):
    DequantizeInt4Loop(x, scale, /*zp=*/0.0f, /*is_signed=*/false, output);
    break;
  case static_cast<int32_t>(DataType::INT2):
    DequantizeInt2Loop(x, scale, /*zp=*/0.0f, /*is_signed=*/true, output);
    break;
  case static_cast<int32_t>(DataType::UINT2):
    DequantizeInt2Loop(x, scale, /*zp=*/0.0f, /*is_signed=*/false, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT4E2M1):
    DequantizeFloat4E2M1Loop(x, scale, /*zp=*/0.0f, output);
    break;
  default:
    EXT_THROW_INVALID(
        "unsupported data type ", x.data_type, ", ",
        "kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16, INT32, FLOAT8E4M3FN, "
        "FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, INT4, UINT4, INT2, UINT2 and FLOAT4E2M1 "
        "inputs are supported.");
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
  case static_cast<int32_t>(DataType::INT4):
    DequantizeInt4Loop(x, scale, ReadSubByteScalarZP(x_zero_point), /*is_signed=*/true, output);
    break;
  case static_cast<int32_t>(DataType::UINT4):
    DequantizeInt4Loop(x, scale, ReadSubByteScalarZP(x_zero_point), /*is_signed=*/false, output);
    break;
  case static_cast<int32_t>(DataType::INT2):
    DequantizeInt2Loop(x, scale, ReadSubByteScalarZP(x_zero_point), /*is_signed=*/true, output);
    break;
  case static_cast<int32_t>(DataType::UINT2):
    DequantizeInt2Loop(x, scale, ReadSubByteScalarZP(x_zero_point), /*is_signed=*/false, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT4E2M1):
    DequantizeFloat4E2M1Loop(x, scale, ReadSubByteScalarZP(x_zero_point), output);
    break;
  default:
    EXT_THROW_INVALID(
        "unsupported data type ", x.data_type, ", ",
        "kernel::DequantizeLinear: only UINT8, INT8, UINT16, INT16, INT32, FLOAT8E4M3FN, "
        "FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, INT4, UINT4, INT2, UINT2 and FLOAT4E2M1 "
        "inputs are supported.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
