// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/cast_sub_byte.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
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
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
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

// Computes the stride of elements inner to ``axis`` for ``shape``.
inline int64_t ComputeInnerStride(const onnx_kernels::Shape &shape, int64_t axis) {
  int64_t stride = 1;
  for (int64_t d = axis + 1; d < static_cast<int64_t>(shape.size()); ++d) {
    stride *= shape[static_cast<std::size_t>(d)];
  }
  return stride;
}

// Computes, for every element of ``x``, the flat index of the scale (and
// zero-point) value that governs it. For per-axis dequantization the scale is a
// 1-D tensor indexed by the coordinate along ``axis``. For blocked
// dequantization the scale has the same rank as ``x`` and a coarser ``axis``
// dimension; the block size is derived per dimension as ``x_shape[d] /
// scale_shape[d]`` (it is not passed in), so consecutive elements along ``axis``
// share one scale. This mirrors the upstream ``np.repeat`` expansion of the
// scale tensor.
void ComputeScaleIndex(const Tensor &x, const Tensor &x_scale, int64_t axis, int64_t *scale_index) {
  const onnx_kernels::Shape &x_shape = x.shape;
  const std::size_t rank = x_shape.size();
  const int64_t n = x.element_count();

  if (x_scale.shape.size() == rank) {
    // Blocked: scale shape matches x rank, divides x element-wise (only ``axis``
    // is coarser in practice). The scale flat index is obtained by dividing each
    // coordinate by the per-dimension repeat factor.
    const onnx_kernels::Shape &s_shape = x_scale.shape;
    onnx_kernels::Shape repeats;
    repeats.assign(rank, 0);
    onnx_kernels::Shape s_strides;
    s_strides.assign(rank, 0);
    int64_t stride = 1;
    for (std::size_t d = rank; d-- > 0;) {
      s_strides[d] = stride;
      stride *= s_shape[d];
      repeats[d] = s_shape[d] != 0 ? x_shape[d] / s_shape[d] : 1;
    }
    onnx_kernels::Shape coord;
    coord.assign(rank, 0);
    for (int64_t i = 0; i < n; ++i) {
      int64_t si = 0;
      for (std::size_t d = 0; d < rank; ++d) {
        si += (coord[d] / repeats[d]) * s_strides[d];
      }
      scale_index[static_cast<std::size_t>(i)] = si;
      for (std::size_t d = rank; d-- > 0;) {
        if (++coord[d] < x_shape[d]) {
          break;
        }
        coord[d] = 0;
      }
    }
    return;
  }

  // Per-axis: 1-D scale indexed by the coordinate along ``axis``.
  const int64_t inner_stride = ComputeInnerStride(x_shape, axis);
  const int64_t axis_size = x_shape[static_cast<std::size_t>(axis)];
  for (int64_t i = 0; i < n; ++i) {
    scale_index[static_cast<std::size_t>(i)] = (i / inner_stride) % axis_size;
  }
}

// Per-block dequantization for whole-byte integer input types.
template <typename XT>
void DequantizeBlockLoop(const Tensor &x, const float *scales, const XT *zp_data,
                         const int64_t *scale_index, Tensor &output) {
  const XT *px = reinterpret_cast<const XT *>(x.bytes());
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    py[i] = (static_cast<float>(px[i]) - static_cast<float>(zp_data[si])) * scales[si];
  }
}

// Per-block dequantization for float8 byte-per-element input types.
void DequantizeBlockFloat8Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                               Float8Decoder decode, const int64_t *scale_index, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const float zp = decode(zp_bytes[si]);
    py[i] = (decode(px[i]) - zp) * scales[si];
  }
}

// Per-block dequantization for INT4/UINT4 sub-byte packed input.
void DequantizeBlockInt4Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                             bool is_signed, const int64_t *scale_index, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const std::uint8_t zp_nibble = Read4BitElement(zp_bytes, si);
    const float zp = is_signed ? static_cast<float>(Int4NibbleToInt8(zp_nibble))
                               : static_cast<float>(Uint4NibbleToUint8(zp_nibble));
    const std::uint8_t nibble = Read4BitElement(px, i);
    const float val = is_signed ? static_cast<float>(Int4NibbleToInt8(nibble))
                                : static_cast<float>(Uint4NibbleToUint8(nibble));
    py[i] = (val - zp) * scales[si];
  }
}

// Per-block dequantization for INT2/UINT2 sub-byte packed input.
void DequantizeBlockInt2Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                             bool is_signed, const int64_t *scale_index, Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const std::uint8_t zp_bits = Read2BitElement(zp_bytes, si);
    const float zp = is_signed ? static_cast<float>(Int2BitsToInt8(zp_bits))
                               : static_cast<float>(Uint2BitsToUint8(zp_bits));
    const std::uint8_t bits = Read2BitElement(px, i);
    const float val = is_signed ? static_cast<float>(Int2BitsToInt8(bits))
                                : static_cast<float>(Uint2BitsToUint8(bits));
    py[i] = (val - zp) * scales[si];
  }
}

// Per-block dequantization for FLOAT4E2M1 sub-byte packed input.
void DequantizeBlockFloat4E2M1Loop(const Tensor &x, const float *scales,
                                   const std::uint8_t *zp_bytes, const int64_t *scale_index,
                                   Tensor &output) {
  const std::uint8_t *px = x.bytes();
  const int64_t n = x.element_count();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const float zp = Float4E2M1NibbleToFloat(Read4BitElement(zp_bytes, si));
    const float val = Float4E2M1NibbleToFloat(Read4BitElement(px, i));
    py[i] = (val - zp) * scales[si];
  }
}

} // namespace

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                    RuntimeContext *rt) const {
  // The output element type matches ``x_scale``'s element type (FLOAT or
  // FLOAT16). Both encodings occupy known fixed-size storage so the buffer
  // can be sized up-front.
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, x_scale.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x_scale.data_type, x.shape, out_n_bytes, nullptr);
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
      output.size_bytes() == static_cast<size_t>(x.element_count()) * output.element_size(),
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
                                    const Tensor &x_zero_point, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, x_scale.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x_scale.data_type, x.shape, out_n_bytes, nullptr);
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
      output.size_bytes() == static_cast<size_t>(x.element_count()) * output.element_size(),
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
    const float zp = decode(x_zero_point.bytes()[0]);
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

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                    const Tensor &x_zero_point, int64_t axis,
                                    RuntimeContext *rt) const {
  if (x_scale.element_count() == 1) {
    return (*this)(x, x_scale, x_zero_point, rt);
  }
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, x_scale.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x_scale.data_type, x.shape, out_n_bytes, nullptr);
  (*this)(x, x_scale, x_zero_point, axis, out, rt);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale,
                                  const Tensor &x_zero_point, int64_t axis, Tensor &output,
                                  RuntimeContext *rt) const {
  if (x_scale.element_count() == 1) {
    return (*this)(x, x_scale, x_zero_point, output);
  }
  EXT_ENFORCE_INVALID(x_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: x_scale must be FLOAT for per-axis "
                      "dequantization.");
  EXT_ENFORCE_INVALID(x.data_type == x_zero_point.data_type,
                      "kernel::DequantizeLinear: x_zero_point data_type must match x.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::DequantizeLinear: output dtype must be FLOAT for per-axis "
                      "dequantization.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::DequantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(x.element_count()) * output.element_size(),
      "kernel::DequantizeLinear preallocated output buffer has unexpected size in bytes.");
  EXT_ENFORCE_INVALID(axis >= 0 && axis < static_cast<int64_t>(x.shape.size()),
                      "kernel::DequantizeLinear: axis out of range.");
  // Two layouts share this code path: per-axis (1-D scale indexed by the
  // coordinate along ``axis``) and blocked (scale has the same rank as ``x`` and
  // a coarser ``axis`` dimension). ``x_scale`` and ``x_zero_point`` must share
  // the same shape in both layouts.
  EXT_ENFORCE_INVALID(
      x_scale.shape == x_zero_point.shape,
      "kernel::DequantizeLinear: x_scale and x_zero_point must have the same shape.");
  if (x_scale.shape.size() == x.shape.size()) {
    // Blocked: every scale dimension must divide the matching ``x`` dimension,
    // and only ``axis`` may differ from the corresponding ``x`` dimension.
    for (std::size_t d = 0; d < x.shape.size(); ++d) {
      const int64_t s_dim = x_scale.shape[d];
      EXT_ENFORCE_INVALID(s_dim > 0 && x.shape[d] % s_dim == 0,
                          "kernel::DequantizeLinear: blocked x_scale dimension must divide the "
                          "matching x dimension.");
      EXT_ENFORCE_INVALID(static_cast<int64_t>(d) == axis || s_dim == x.shape[d],
                          "kernel::DequantizeLinear: blocked x_scale may only differ from x along "
                          "the quantization axis.");
    }
  } else {
    const int64_t axis_size = x.shape[static_cast<std::size_t>(axis)];
    EXT_ENFORCE_INVALID(
        x_scale.element_count() == axis_size,
        "kernel::DequantizeLinear: x_scale element count must equal axis dimension.");
  }
  RawBufferAllocator *allocator =
      rt ? rt->execution_allocator()
         : (output.has_allocation() ? output.allocation_owner() : nullptr);
  detail::TemporaryTypedBuffer<int64_t> scale_index_buf(static_cast<std::size_t>(x.element_count()),
                                                        allocator,
                                                        "kernel::DequantizeLinear: scale-index");
  ComputeScaleIndex(x, x_scale, axis, scale_index_buf.data());
  const int64_t *idx = scale_index_buf.data();
  const float *scales = x_scale.AsFloat();
  const std::uint8_t *zp_bytes = x_zero_point.bytes();

  switch (x.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    DequantizeBlockLoop<uint8_t>(x, scales, reinterpret_cast<const uint8_t *>(zp_bytes), idx,
                                 output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    DequantizeBlockLoop<int8_t>(x, scales, reinterpret_cast<const int8_t *>(zp_bytes), idx, output);
    break;
  case static_cast<int32_t>(DataType::UINT16): {
    const int64_t n_zp = x_zero_point.element_count();
    detail::TemporaryTypedBuffer<uint16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                  "kernel::DequantizeLinear: zero-point");
    zp_vec.CopyFromBytes(zp_bytes);
    DequantizeBlockLoop<uint16_t>(x, scales, zp_vec.data(), idx, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT16): {
    const int64_t n_zp = x_zero_point.element_count();
    detail::TemporaryTypedBuffer<int16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                 "kernel::DequantizeLinear: zero-point");
    zp_vec.CopyFromBytes(zp_bytes);
    DequantizeBlockLoop<int16_t>(x, scales, zp_vec.data(), idx, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT32): {
    const int64_t n_zp = x_zero_point.element_count();
    detail::TemporaryTypedBuffer<int32_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                 "kernel::DequantizeLinear: zero-point");
    zp_vec.CopyFromBytes(zp_bytes);
    DequantizeBlockLoop<int32_t>(x, scales, zp_vec.data(), idx, output);
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ):
    DequantizeBlockFloat8Loop(x, scales, zp_bytes, Float8DecoderFor(x.data_type), idx, output);
    break;
  case static_cast<int32_t>(DataType::INT4):
    DequantizeBlockInt4Loop(x, scales, zp_bytes, /*is_signed=*/true, idx, output);
    break;
  case static_cast<int32_t>(DataType::UINT4):
    DequantizeBlockInt4Loop(x, scales, zp_bytes, /*is_signed=*/false, idx, output);
    break;
  case static_cast<int32_t>(DataType::INT2):
    DequantizeBlockInt2Loop(x, scales, zp_bytes, /*is_signed=*/true, idx, output);
    break;
  case static_cast<int32_t>(DataType::UINT2):
    DequantizeBlockInt2Loop(x, scales, zp_bytes, /*is_signed=*/false, idx, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT4E2M1):
    DequantizeBlockFloat4E2M1Loop(x, scales, zp_bytes, idx, output);
    break;
  default:
    EXT_THROW_INVALID(
        "unsupported data type ", x.data_type, ", ",
        "kernel::DequantizeLinear (per-axis): only UINT8, INT8, UINT16, INT16, INT32, "
        "FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, INT4, UINT4, INT2, UINT2 "
        "and FLOAT4E2M1 inputs are supported.");
  }
}

Tensor DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale, int64_t axis,
                                    RuntimeContext *rt) const {
  if (x_scale.element_count() == 1) {
    return (*this)(x, x_scale, rt);
  }
  EXT_ENFORCE_INVALID(IsSupportedScaleDType(x_scale.data_type),
                      "kernel::DequantizeLinear: x_scale must be FLOAT or FLOAT16.");
  const size_t elem_size = x_scale.data_type == static_cast<int32_t>(DataType::FLOAT16)
                               ? sizeof(uint16_t)
                               : sizeof(float);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, x_scale.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x_scale.data_type, x.shape, out_n_bytes, nullptr);
  (*this)(x, x_scale, axis, out, rt);
  return out;
}

void DequantizeLinear::operator()(const Tensor &x, const Tensor &x_scale, int64_t axis,
                                  Tensor &output, RuntimeContext *rt) const {
  if (x_scale.element_count() == 1) {
    return (*this)(x, x_scale, output);
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < static_cast<int64_t>(x.shape.size()),
                      "kernel::DequantizeLinear: axis out of range.");
  // A zero-filled buffer decodes to a zero point of 0 for every supported
  // input element type (whole-byte integers, float8 and sub-byte packed
  // types), so the per-axis/blocked dequantization can reuse the
  // explicit-zero-point overload. The zero point mirrors ``x_scale``'s shape so
  // both per-axis (1-D) and blocked (N-D) layouts are handled.
  const int64_t scale_count = x_scale.element_count();
  const size_t zero_zero_point_n_bytes = PackedByteSize(x.data_type, scale_count);
  Tensor zero_zero_point =
      rt ? rt->MakeTemporaryTensor(x.data_type, x_scale.shape, zero_zero_point_n_bytes)
         : MakeOutputTensor(x.data_type, x_scale.shape, zero_zero_point_n_bytes, nullptr);
  // Allocator storage is no longer zero-initialised, so clear the synthetic
  // zero-point buffer explicitly: every supported element type decodes a zero
  // byte pattern to a zero point of 0.
  std::fill(zero_zero_point.mutable_bytes(),
            zero_zero_point.mutable_bytes() + zero_zero_point.size_bytes(), uint8_t{0u});
  (*this)(x, x_scale, zero_zero_point, axis, output, rt);
}

void DequantizeLinear::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 2);
  EXT_ENFORCE_INVALID(!(node.input_size() > 3),
                      "RunNode: op 'DequantizeLinear' expects 2 or 3 inputs, got ",
                      node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &x_scale = GetInput(node, 1, rt.tensors());
  const Tensor *x_zero_point = GetOptionalInput(node, 2, rt.tensors());
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", 1);
  onnx_kernels::kernel::DequantizeLinear k(rt.kernel_ctx());
  if (x_zero_point != nullptr) {
    SetOutput(node, 0, k(x, x_scale, *x_zero_point, axis, &rt), rt);
  } else {
    SetOutput(node, 0, k(x, x_scale, axis, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
