// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"

#include "onnx_core/runtime/kernels/cast_float8.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/cast_sub_byte.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
  EXT_ENFORCE_INVALID(n == 1, "kernel::QuantizeLinear: ", name,
                      " must be a scalar (per-tensor quantization).");
}

template <typename ZP>
void QuantizeLoop(const Tensor &x, float y_scale, ZP y_zero_point, Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.mutable_bytes());
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

template <typename ZP> ZP ReadScalarZeroPoint(const Tensor &y_zero_point) {
  ZP value{};
  std::memcpy(&value, y_zero_point.bytes(), sizeof(ZP));
  return value;
}

// Helper to decode float8 bits to float for QuantizeLinear zero-point.
inline float Float8BitsToFloat(std::uint8_t byte, int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT8E4M3FN:
    return Float8E4M3FNBitsToFloat(byte);
  case DataType::FLOAT8E4M3FNUZ:
    return Float8E4M3FNUZBitsToFloat(byte);
  case DataType::FLOAT8E5M2:
    return Float8E5M2BitsToFloat(byte);
  case DataType::FLOAT8E5M2FNUZ:
    return Float8E5M2FNUZBitsToFloat(byte);
  default:
    return 0.0f;
  }
}

// Reads the scalar float zero-point from a FLOAT8-typed tensor (1-element).
inline float ReadFloat8ScalarZP(const Tensor &y_zero_point) {
  const std::uint8_t byte = y_zero_point.bytes()[0];
  return Float8BitsToFloat(byte, y_zero_point.data_type);
}

// Helper to encode a float to float8 for QuantizeLinear.
inline std::uint8_t FloatToFloat8(float v, int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT8E4M3FN:
    return FloatToFloat8E4M3FNBits(v);
  case DataType::FLOAT8E4M3FNUZ:
    return FloatToFloat8E4M3FNUZBits(v);
  case DataType::FLOAT8E5M2:
    return FloatToFloat8E5M2Bits(v);
  case DataType::FLOAT8E5M2FNUZ:
    return FloatToFloat8E5M2FNUZBits(v);
  default:
    EXT_THROW_INVALID("unsupported data type ", dtype, ", ",
                      "kernel::QuantizeLinear: unsupported float8 dtype.");
  }
}

// Per-axis quantization loop: each element selects its scale/ZP by its
// position along ``axis``.  Works for whole-byte integer output types.
template <typename ZP>
void QuantizeAxisLoop(const Tensor &x, const float *scales, const ZP *zp_data, int64_t inner_stride,
                      int64_t axis_size, Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.mutable_bytes());
  const int64_t n = x.element_count();
  constexpr float kMin = static_cast<float>(std::numeric_limits<ZP>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<ZP>::max());
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const float zp = static_cast<float>(zp_data[axis_idx]);
    float v = RoundHalfToEven(px[i] / scales[axis_idx]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    py[i] = static_cast<ZP>(v);
  }
}

// Per-axis quantization loop with zero zero-point (symmetric).
template <typename ZP>
void QuantizeAxisLoopSymmetric(const Tensor &x, const float *scales, int64_t inner_stride,
                               int64_t axis_size, Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.mutable_bytes());
  const int64_t n = x.element_count();
  constexpr float kMin = static_cast<float>(std::numeric_limits<ZP>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<ZP>::max());
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    float v = RoundHalfToEven(px[i] / scales[axis_idx]);
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    py[i] = static_cast<ZP>(v);
  }
}

// Per-axis quantization for 4-bit (INT4/UINT4) packed output.
void QuantizeAxisInt4Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                          int32_t out_dtype, int64_t inner_stride, int64_t axis_size,
                          Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  const bool is_signed = (static_cast<DataType>(out_dtype) == DataType::INT4);
  const float kMin = is_signed ? -8.0f : 0.0f;
  const float kMax = is_signed ? 7.0f : 15.0f;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const std::uint8_t zp_nibble = Read4BitElement(zp_bytes, axis_idx);
    const float zp = is_signed ? static_cast<float>(Int4NibbleToInt8(zp_nibble))
                               : static_cast<float>(Uint4NibbleToUint8(zp_nibble));
    float v = RoundHalfToEven(px[i] / scales[axis_idx]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    const std::uint8_t nibble = static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x0Fu;
    Write4BitElement(py, i, nibble);
  }
}

// Per-axis quantization for 2-bit (INT2/UINT2) packed output.
void QuantizeAxisInt2Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                          int32_t out_dtype, int64_t inner_stride, int64_t axis_size,
                          Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  const bool is_signed = (static_cast<DataType>(out_dtype) == DataType::INT2);
  const float kMin = is_signed ? -2.0f : 0.0f;
  const float kMax = is_signed ? 1.0f : 3.0f;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const std::uint8_t zp_bits = Read2BitElement(zp_bytes, axis_idx);
    const float zp = is_signed ? static_cast<float>(Int2BitsToInt8(zp_bits))
                               : static_cast<float>(Uint2BitsToUint8(zp_bits));
    float v = RoundHalfToEven(px[i] / scales[axis_idx]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    const std::uint8_t bits = static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x03u;
    Write2BitElement(py, i, bits);
  }
}

// Per-axis quantization for FLOAT4E2M1 packed output.
void QuantizeAxisFloat4E2M1Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                                int64_t inner_stride, int64_t axis_size, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const float zp = Float4E2M1NibbleToFloat(Read4BitElement(zp_bytes, axis_idx));
    const float v = px[i] / scales[axis_idx] + zp;
    Write4BitElement(py, i, FloatRoundToFloat4E2M1Nibble(v));
  }
}

void QuantizeAxisFloat6Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                            int64_t inner_stride, int64_t axis_size, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const auto dtype = static_cast<DataType>(output.data_type);
  std::memset(py, 0, output.size_bytes());
  for (int64_t i = 0; i < x.element_count(); ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const float zp = Float6BitsToFloat(Read6BitElement(zp_bytes, axis_idx), dtype);
    Write6BitElement(py, i, FloatToFloat6Bits(px[i] / scales[axis_idx] + zp, dtype));
  }
}

// Per-axis quantization for float8 byte-per-element output.
void QuantizeAxisFloat8Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                            int32_t out_dtype, int64_t inner_stride, int64_t axis_size,
                            Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t axis_idx = (i / inner_stride) % axis_size;
    const float zp = Float8BitsToFloat(zp_bytes[axis_idx], out_dtype);
    const float v = px[i] / scales[axis_idx] + zp;
    py[i] = FloatToFloat8(v, out_dtype);
  }
}

// Computes the stride of elements inner to ``axis`` for ``shape``.
inline int64_t ComputeInnerStride(const onnx_kernels::Shape &shape, int64_t axis) {
  int64_t stride = 1;
  for (int64_t d = axis + 1; d < static_cast<int64_t>(shape.size()); ++d) {
    stride *= shape[static_cast<std::size_t>(d)];
  }
  return stride;
}

// Computes, for every element of ``x``, the flat index of the scale/ZP value
// that governs it. Supports both per-axis (1-D scale) and blocked (N-D scale).
//
// GCC 13 emits a false-positive -Wfree-nonheap-object when it inlines the
// destructor of the local ``coord`` vector and its interval analysis concludes
// the allocated pointer might be non-heap.  The code is correct; suppress the
// spurious diagnostic for GCC >= 12.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
void ComputeScaleIndex(const Tensor &x, const Tensor &y_scale, int64_t axis, int64_t *scale_index) {
  const onnx_kernels::Shape &x_shape = x.shape;
  const std::size_t rank = x_shape.size();
  const int64_t n = x.element_count();

  if (y_scale.shape.size() == rank) {
    // Blocked: scale shape matches x rank, each scale dim divides x dim.
    const onnx_kernels::Shape &s_shape = y_scale.shape;
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
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

// Per-block quantization for whole-byte integer output types.
template <typename ZP>
void QuantizeBlockLoop(const Tensor &x, const float *scales, const ZP *zp_data,
                       const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.mutable_bytes());
  const int64_t n = x.element_count();
  constexpr float kMin = static_cast<float>(std::numeric_limits<ZP>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<ZP>::max());
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const float zp = static_cast<float>(zp_data[si]);
    float v = RoundHalfToEven(px[i] / scales[si]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    py[i] = static_cast<ZP>(v);
  }
}

// Per-block quantization for whole-byte integer output types with zero zero-point (symmetric).
template <typename ZP>
void QuantizeBlockLoopSymmetric(const Tensor &x, const float *scales, const int64_t *scale_index,
                                Tensor &output) {
  const float *px = x.AsFloat();
  ZP *py = reinterpret_cast<ZP *>(output.mutable_bytes());
  const int64_t n = x.element_count();
  constexpr float kMin = static_cast<float>(std::numeric_limits<ZP>::min());
  constexpr float kMax = static_cast<float>(std::numeric_limits<ZP>::max());
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    float v = RoundHalfToEven(px[i] / scales[si]);
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    py[i] = static_cast<ZP>(v);
  }
}

// Per-block quantization for INT2/UINT2 packed output.
void QuantizeBlockInt2Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                           int32_t out_dtype, const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  const bool is_signed = (static_cast<DataType>(out_dtype) == DataType::INT2);
  const float kMin = is_signed ? -2.0f : 0.0f;
  const float kMax = is_signed ? 1.0f : 3.0f;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const std::uint8_t zp_bits = Read2BitElement(zp_bytes, si);
    const float zp = is_signed ? static_cast<float>(Int2BitsToInt8(zp_bits))
                               : static_cast<float>(Uint2BitsToUint8(zp_bits));
    float v = RoundHalfToEven(px[i] / scales[si]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    Write2BitElement(py, i, static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x03u);
  }
}

// Per-block quantization for FLOAT4E2M1 packed output.
void QuantizeBlockFloat4E2M1Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                                 const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const float zp = Float4E2M1NibbleToFloat(Read4BitElement(zp_bytes, si));
    const float v = px[i] / scales[si] + zp;
    Write4BitElement(py, i, FloatRoundToFloat4E2M1Nibble(v));
  }
}

void QuantizeBlockFloat6Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                             const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const auto dtype = static_cast<DataType>(output.data_type);
  std::memset(py, 0, output.size_bytes());
  for (int64_t i = 0; i < x.element_count(); ++i) {
    const int64_t si = scale_index[i];
    const float zp = Float6BitsToFloat(Read6BitElement(zp_bytes, si), dtype);
    Write6BitElement(py, i, FloatToFloat6Bits(px[i] / scales[si] + zp, dtype));
  }
}

// Per-block quantization for INT4/UINT4 packed output.
void QuantizeBlockInt4Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                           int32_t out_dtype, const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  const bool is_signed = (static_cast<DataType>(out_dtype) == DataType::INT4);
  const float kMin = is_signed ? -8.0f : 0.0f;
  const float kMax = is_signed ? 7.0f : 15.0f;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const std::uint8_t zp_nibble = Read4BitElement(zp_bytes, si);
    const float zp = is_signed ? static_cast<float>(Int4NibbleToInt8(zp_nibble))
                               : static_cast<float>(Uint4NibbleToUint8(zp_nibble));
    float v = RoundHalfToEven(px[i] / scales[si]) + zp;
    if (v < kMin) {
      v = kMin;
    } else if (v > kMax) {
      v = kMax;
    }
    Write4BitElement(py, i, static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x0Fu);
  }
}

// Per-block quantization for float8 byte-per-element output.
void QuantizeBlockFloat8Loop(const Tensor &x, const float *scales, const std::uint8_t *zp_bytes,
                             int32_t out_dtype, const int64_t *scale_index, Tensor &output) {
  const float *px = x.AsFloat();
  std::uint8_t *py = output.mutable_bytes();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t si = scale_index[i];
    const float zp = Float8BitsToFloat(zp_bytes[si], out_dtype);
    const float v = px[i] / scales[si] + zp;
    py[i] = FloatToFloat8(v, out_dtype);
  }
}

} // namespace

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale,
                                  RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count());
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::UINT8), x.shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::UINT8), x.shape, out_n_bytes, nullptr);
  (*this)(x, y_scale, out);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  RequireScalar(y_scale, "y_scale");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == PackedByteSize(output.data_type, x.element_count()),
      "kernel::QuantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = y_scale.AsFloat()[0];
  switch (output.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    QuantizeLoop<uint8_t>(x, scale, /*y_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    QuantizeLoop<int8_t>(x, scale, /*y_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::UINT16):
    QuantizeLoop<uint16_t>(x, scale, /*y_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::INT16):
    QuantizeLoop<int16_t>(x, scale, /*y_zero_point=*/0, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT4E2M1): {
    // No y_zero_point: do not add a zero point so that signed zero is
    // preserved (adding 0.0f to -0.0f would flip the sign to +0.0f).
    const int64_t n = x.element_count();
    const float *pxf = x.AsFloat();
    std::uint8_t *py = output.mutable_bytes();
    for (int64_t i = 0; i < n; ++i) {
      Write4BitElement(py, i, FloatRoundToFloat4E2M1Nibble(pxf[i] / scale));
    }
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT6E2M3):
  case static_cast<int32_t>(DataType::FLOAT6E3M2): {
    const auto dtype = static_cast<DataType>(output.data_type);
    const float *px = x.AsFloat();
    std::memset(output.mutable_bytes(), 0, output.size_bytes());
    for (int64_t i = 0; i < x.element_count(); ++i)
      Write6BitElement(output.mutable_bytes(), i, FloatToFloat6Bits(px[i] / scale, dtype));
    break;
  }
  default:
    EXT_THROW_INVALID(
        "unsupported data type ", output.data_type, ", ",
        "kernel::QuantizeLinear: only UINT8, INT8, UINT16, INT16 and FLOAT4E2M1 outputs are "
        "supported (no-zero-point overload).");
  }
}

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale,
                                  const Tensor &y_zero_point, RuntimeContext *rt) const {
  const size_t out_n_bytes = PackedByteSize(y_zero_point.data_type, x.element_count());
  Tensor out = rt ? rt->MakeOutputTensor(0, y_zero_point.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(y_zero_point.data_type, x.shape, out_n_bytes, nullptr);
  (*this)(x, y_scale, y_zero_point, out);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, const Tensor &y_zero_point,
                                Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  RequireScalar(y_scale, "y_scale");
  RequireScalar(y_zero_point, "y_zero_point");
  EXT_ENFORCE_INVALID(output.data_type == y_zero_point.data_type,
                      "kernel::QuantizeLinear: output data_type must match y_zero_point.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == PackedByteSize(output.data_type, x.element_count()),
      "kernel::QuantizeLinear preallocated output buffer has unexpected size in bytes.");
  const float scale = y_scale.AsFloat()[0];
  switch (output.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    QuantizeLoop<uint8_t>(x, scale, ReadScalarZeroPoint<uint8_t>(y_zero_point), output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    QuantizeLoop<int8_t>(x, scale, ReadScalarZeroPoint<int8_t>(y_zero_point), output);
    break;
  case static_cast<int32_t>(DataType::UINT16):
    QuantizeLoop<uint16_t>(x, scale, ReadScalarZeroPoint<uint16_t>(y_zero_point), output);
    break;
  case static_cast<int32_t>(DataType::INT16):
    QuantizeLoop<int16_t>(x, scale, ReadScalarZeroPoint<int16_t>(y_zero_point), output);
    break;
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ): {
    const float zp = ReadFloat8ScalarZP(y_zero_point);
    std::uint8_t *py = output.mutable_bytes();
    const float *pxf = x.AsFloat();
    const int64_t n = x.element_count();
    for (int64_t i = 0; i < n; ++i) {
      py[i] = FloatToFloat8(pxf[i] / scale + zp, output.data_type);
    }
    break;
  }
  case static_cast<int32_t>(DataType::INT4):
  case static_cast<int32_t>(DataType::UINT4): {
    const int64_t n = x.element_count();
    const float *pxf = x.AsFloat();
    std::uint8_t *py = output.mutable_bytes();
    const bool is_signed = (static_cast<DataType>(output.data_type) == DataType::INT4);
    const float kMin = is_signed ? -8.0f : 0.0f;
    const float kMax = is_signed ? 7.0f : 15.0f;
    const std::uint8_t zp_nibble = Read4BitElement(y_zero_point.bytes(), 0);
    const float zp = is_signed ? static_cast<float>(Int4NibbleToInt8(zp_nibble))
                               : static_cast<float>(Uint4NibbleToUint8(zp_nibble));
    for (int64_t i = 0; i < n; ++i) {
      float v = RoundHalfToEven(pxf[i] / scale) + zp;
      if (v < kMin)
        v = kMin;
      else if (v > kMax)
        v = kMax;
      Write4BitElement(py, i, static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x0Fu);
    }
    break;
  }
  case static_cast<int32_t>(DataType::INT2):
  case static_cast<int32_t>(DataType::UINT2): {
    const int64_t n = x.element_count();
    const float *pxf = x.AsFloat();
    std::uint8_t *py = output.mutable_bytes();
    const bool is_signed = (static_cast<DataType>(output.data_type) == DataType::INT2);
    const float kMin = is_signed ? -2.0f : 0.0f;
    const float kMax = is_signed ? 1.0f : 3.0f;
    const std::uint8_t zp_bits = Read2BitElement(y_zero_point.bytes(), 0);
    const float zp = is_signed ? static_cast<float>(Int2BitsToInt8(zp_bits))
                               : static_cast<float>(Uint2BitsToUint8(zp_bits));
    for (int64_t i = 0; i < n; ++i) {
      float v = RoundHalfToEven(pxf[i] / scale) + zp;
      if (v < kMin)
        v = kMin;
      else if (v > kMax)
        v = kMax;
      Write2BitElement(py, i, static_cast<std::uint8_t>(static_cast<int32_t>(v)) & 0x03u);
    }
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT4E2M1): {
    const int64_t n = x.element_count();
    const float *pxf = x.AsFloat();
    std::uint8_t *py = output.mutable_bytes();
    const float zp = Float4E2M1NibbleToFloat(Read4BitElement(y_zero_point.bytes(), 0));
    for (int64_t i = 0; i < n; ++i) {
      Write4BitElement(py, i, FloatRoundToFloat4E2M1Nibble(pxf[i] / scale + zp));
    }
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT6E2M3):
  case static_cast<int32_t>(DataType::FLOAT6E3M2): {
    const auto dtype = static_cast<DataType>(output.data_type);
    const float zp = Float6BitsToFloat(Read6BitElement(y_zero_point.bytes(), 0), dtype);
    const float *px = x.AsFloat();
    std::memset(output.mutable_bytes(), 0, output.size_bytes());
    for (int64_t i = 0; i < x.element_count(); ++i)
      Write6BitElement(output.mutable_bytes(), i, FloatToFloat6Bits(px[i] / scale + zp, dtype));
    break;
  }
  default:
    EXT_THROW_INVALID(
        "unsupported data type ", output.data_type, ", ",
        "kernel::QuantizeLinear: only UINT8, INT8, UINT16, INT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, "
        "FLOAT8E5M2, FLOAT8E5M2FNUZ, INT4, UINT4, INT2, UINT2 and FLOAT4E2M1 outputs are "
        "supported.");
  }
}

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale,
                                  const Tensor &y_zero_point, int64_t axis,
                                  RuntimeContext *rt) const {
  // Scalar scale: delegate to the per-tensor overload.
  if (y_scale.element_count() == 1) {
    return (*this)(x, y_scale, y_zero_point, rt);
  }
  const size_t out_n_bytes = PackedByteSize(y_zero_point.data_type, x.element_count());
  Tensor out = rt ? rt->MakeOutputTensor(0, y_zero_point.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(y_zero_point.data_type, x.shape, out_n_bytes, nullptr);
  (*this)(x, y_scale, y_zero_point, axis, out, rt);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, const Tensor &y_zero_point,
                                int64_t axis, Tensor &output, RuntimeContext *rt) const {
  // Scalar scale: delegate to the per-tensor in-place overload.
  if (y_scale.element_count() == 1) {
    return (*this)(x, y_scale, y_zero_point, output);
  }
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == y_zero_point.data_type,
                      "kernel::QuantizeLinear: output data_type must match y_zero_point.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == PackedByteSize(output.data_type, x.element_count()),
      "kernel::QuantizeLinear preallocated output buffer has unexpected size in bytes.");
  EXT_ENFORCE_INVALID(axis >= 0 && axis < static_cast<int64_t>(x.shape.size()),
                      "kernel::QuantizeLinear: axis out of range.");
  EXT_ENFORCE_INVALID(y_scale.shape == y_zero_point.shape,
                      "kernel::QuantizeLinear: y_scale and y_zero_point must have the same shape.");

  // Detect blocked vs per-axis layout.
  const bool blocked = (y_scale.shape.size() == x.shape.size());
  if (blocked) {
    for (std::size_t d = 0; d < x.shape.size(); ++d) {
      const int64_t s_dim = y_scale.shape[d];
      EXT_ENFORCE_INVALID(s_dim > 0 && x.shape[d] % s_dim == 0,
                          "kernel::QuantizeLinear: blocked y_scale dimension must divide the "
                          "matching x dimension.");
      EXT_ENFORCE_INVALID(static_cast<int64_t>(d) == axis || s_dim == x.shape[d],
                          "kernel::QuantizeLinear: blocked y_scale may only differ from x along "
                          "the quantization axis.");
    }
  } else {
    const int64_t axis_size = x.shape[static_cast<std::size_t>(axis)];
    EXT_ENFORCE_INVALID(y_scale.element_count() == axis_size,
                        "kernel::QuantizeLinear: y_scale element count must equal axis dimension.");
  }

  const float *scales = y_scale.AsFloat();
  const std::uint8_t *zp_bytes = y_zero_point.bytes();
  RawBufferAllocator *allocator =
      rt ? rt->execution_allocator()
         : (output.has_allocation() ? output.allocation_owner() : nullptr);

  if (!blocked) {
    // Per-axis: use QuantizeAxisLoop (no scale_index buffer needed).
    // For UINT16/INT16 the ZP values must be widened into a typed buffer;
    // all other types can read ZP bytes directly.
    const int64_t inner_stride = ComputeInnerStride(x.shape, axis);
    const int64_t axis_size = x.shape[static_cast<std::size_t>(axis)];
    switch (output.data_type) {
    case static_cast<int32_t>(DataType::UINT8):
      QuantizeAxisLoop<uint8_t>(x, scales, reinterpret_cast<const uint8_t *>(zp_bytes),
                                inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::INT8):
      QuantizeAxisLoop<int8_t>(x, scales, reinterpret_cast<const int8_t *>(zp_bytes), inner_stride,
                               axis_size, output);
      return;
    case static_cast<int32_t>(DataType::UINT16): {
      const int64_t n_zp = y_zero_point.element_count();
      detail::TemporaryTypedBuffer<uint16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                    "kernel::QuantizeLinear: zero-point");
      zp_vec.CopyFromBytes(zp_bytes);
      QuantizeAxisLoop<uint16_t>(x, scales, zp_vec.data(), inner_stride, axis_size, output);
      return;
    }
    case static_cast<int32_t>(DataType::INT16): {
      const int64_t n_zp = y_zero_point.element_count();
      detail::TemporaryTypedBuffer<int16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                   "kernel::QuantizeLinear: zero-point");
      zp_vec.CopyFromBytes(zp_bytes);
      QuantizeAxisLoop<int16_t>(x, scales, zp_vec.data(), inner_stride, axis_size, output);
      return;
    }
    case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
    case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
    case static_cast<int32_t>(DataType::FLOAT8E5M2):
    case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ):
      QuantizeAxisFloat8Loop(x, scales, zp_bytes, output.data_type, inner_stride, axis_size,
                             output);
      return;
    case static_cast<int32_t>(DataType::INT4):
    case static_cast<int32_t>(DataType::UINT4):
      QuantizeAxisInt4Loop(x, scales, zp_bytes, output.data_type, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::INT2):
    case static_cast<int32_t>(DataType::UINT2):
      QuantizeAxisInt2Loop(x, scales, zp_bytes, output.data_type, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::FLOAT4E2M1):
      QuantizeAxisFloat4E2M1Loop(x, scales, zp_bytes, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::FLOAT6E2M3):
    case static_cast<int32_t>(DataType::FLOAT6E3M2):
      QuantizeAxisFloat6Loop(x, scales, zp_bytes, inner_stride, axis_size, output);
      return;
    default:
      EXT_THROW_INVALID("unsupported data type ", output.data_type, ", ",
                        "kernel::QuantizeLinear (per-axis): unsupported output dtype.");
    }
  }

  // Blocked quantization: compute a flat scale index for every element, then
  // dispatch to the block loop.  For UINT16/INT16 the ZP values must be
  // widened into a typed buffer to avoid strict-aliasing violations.
  detail::TemporaryTypedBuffer<int64_t> scale_index_buf(static_cast<std::size_t>(x.element_count()),
                                                        allocator,
                                                        "kernel::QuantizeLinear: scale-index");
  ComputeScaleIndex(x, y_scale, axis, scale_index_buf.data());
  const int64_t *idx = scale_index_buf.data();

  switch (output.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    QuantizeBlockLoop<uint8_t>(x, scales, reinterpret_cast<const uint8_t *>(zp_bytes), idx, output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    QuantizeBlockLoop<int8_t>(x, scales, reinterpret_cast<const int8_t *>(zp_bytes), idx, output);
    break;
  case static_cast<int32_t>(DataType::UINT16): {
    const int64_t n_zp = y_zero_point.element_count();
    detail::TemporaryTypedBuffer<uint16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                  "kernel::QuantizeLinear: zero-point");
    zp_vec.CopyFromBytes(zp_bytes);
    QuantizeBlockLoop<uint16_t>(x, scales, zp_vec.data(), idx, output);
    break;
  }
  case static_cast<int32_t>(DataType::INT16): {
    const int64_t n_zp = y_zero_point.element_count();
    detail::TemporaryTypedBuffer<int16_t> zp_vec(static_cast<std::size_t>(n_zp), allocator,
                                                 "kernel::QuantizeLinear: zero-point");
    zp_vec.CopyFromBytes(zp_bytes);
    QuantizeBlockLoop<int16_t>(x, scales, zp_vec.data(), idx, output);
    break;
  }
  case static_cast<int32_t>(DataType::FLOAT8E4M3FN):
  case static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ):
  case static_cast<int32_t>(DataType::FLOAT8E5M2):
  case static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ):
    QuantizeBlockFloat8Loop(x, scales, zp_bytes, output.data_type, idx, output);
    break;
  case static_cast<int32_t>(DataType::INT4):
  case static_cast<int32_t>(DataType::UINT4):
    QuantizeBlockInt4Loop(x, scales, zp_bytes, output.data_type, idx, output);
    break;
  case static_cast<int32_t>(DataType::INT2):
  case static_cast<int32_t>(DataType::UINT2):
    QuantizeBlockInt2Loop(x, scales, zp_bytes, output.data_type, idx, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT4E2M1):
    QuantizeBlockFloat4E2M1Loop(x, scales, zp_bytes, idx, output);
    break;
  case static_cast<int32_t>(DataType::FLOAT6E2M3):
  case static_cast<int32_t>(DataType::FLOAT6E3M2):
    QuantizeBlockFloat6Loop(x, scales, zp_bytes, idx, output);
    break;
  default:
    EXT_THROW_INVALID("unsupported data type ", output.data_type, ", ",
                      "kernel::QuantizeLinear (blocked): unsupported output dtype.");
  }
}

Tensor QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, int64_t axis,
                                  int32_t output_dtype, RuntimeContext *rt) const {
  const size_t out_n_bytes = PackedByteSize(output_dtype, x.element_count());
  Tensor out = rt ? rt->MakeOutputTensor(0, output_dtype, x.shape, out_n_bytes)
                  : MakeOutputTensor(output_dtype, x.shape, out_n_bytes, nullptr);
  (*this)(x, y_scale, axis, output_dtype, out, rt);
  return out;
}

void QuantizeLinear::operator()(const Tensor &x, const Tensor &y_scale, int64_t axis,
                                int32_t output_dtype, Tensor &output, RuntimeContext *rt) const {
  if (y_scale.element_count() == 1) {
    return (*this)(x, y_scale, output);
  }
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: x must be FLOAT.");
  EXT_ENFORCE_INVALID(y_scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::QuantizeLinear: y_scale must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == output_dtype,
                      "kernel::QuantizeLinear: output data_type must match output_dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::QuantizeLinear preallocated output shape must match x shape.");
  EXT_ENFORCE_INVALID(axis >= 0 && axis < static_cast<int64_t>(x.shape.size()),
                      "kernel::QuantizeLinear: axis out of range.");

  const bool blocked = (y_scale.shape.size() == x.shape.size());
  if (blocked) {
    for (std::size_t d = 0; d < x.shape.size(); ++d) {
      const int64_t s_dim = y_scale.shape[d];
      EXT_ENFORCE_INVALID(s_dim > 0 && x.shape[d] % s_dim == 0,
                          "kernel::QuantizeLinear: blocked y_scale dimension must divide the "
                          "matching x dimension.");
      EXT_ENFORCE_INVALID(static_cast<int64_t>(d) == axis || s_dim == x.shape[d],
                          "kernel::QuantizeLinear: blocked y_scale may only differ from x along "
                          "the quantization axis.");
    }
  } else {
    const int64_t axis_size = x.shape[static_cast<std::size_t>(axis)];
    EXT_ENFORCE_INVALID(y_scale.element_count() == axis_size,
                        "kernel::QuantizeLinear: y_scale element count must equal axis dimension.");
  }

  RawBufferAllocator *allocator =
      rt ? rt->execution_allocator()
         : (output.has_allocation() ? output.allocation_owner() : nullptr);
  const float *scales = y_scale.AsFloat();

  if (!blocked) {
    // Per-axis symmetric: zero-point is always zero, no temporary buffer needed.
    const int64_t inner_stride = ComputeInnerStride(x.shape, axis);
    const int64_t axis_size = x.shape[static_cast<std::size_t>(axis)];
    switch (output.data_type) {
    case static_cast<int32_t>(DataType::UINT8):
      QuantizeAxisLoopSymmetric<uint8_t>(x, scales, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::INT8):
      QuantizeAxisLoopSymmetric<int8_t>(x, scales, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::UINT16):
      QuantizeAxisLoopSymmetric<uint16_t>(x, scales, inner_stride, axis_size, output);
      return;
    case static_cast<int32_t>(DataType::INT16):
      QuantizeAxisLoopSymmetric<int16_t>(x, scales, inner_stride, axis_size, output);
      return;
    default:
      EXT_THROW_INVALID("unsupported data type ", output.data_type, ", ",
                        "kernel::QuantizeLinear (symmetric per-axis): unsupported output dtype.");
    }
  }

  // Blocked symmetric: compute a flat scale index, then dispatch.
  detail::TemporaryTypedBuffer<int64_t> scale_index_buf(static_cast<std::size_t>(x.element_count()),
                                                        allocator,
                                                        "kernel::QuantizeLinear: scale-index");
  ComputeScaleIndex(x, y_scale, axis, scale_index_buf.data());
  const int64_t *idx = scale_index_buf.data();

  // Symmetric case: zero-point is always zero, no temporary buffer needed.
  switch (output.data_type) {
  case static_cast<int32_t>(DataType::UINT8):
    QuantizeBlockLoopSymmetric<uint8_t>(x, scales, idx, output);
    break;
  case static_cast<int32_t>(DataType::INT8):
    QuantizeBlockLoopSymmetric<int8_t>(x, scales, idx, output);
    break;
  case static_cast<int32_t>(DataType::UINT16):
    QuantizeBlockLoopSymmetric<uint16_t>(x, scales, idx, output);
    break;
  case static_cast<int32_t>(DataType::INT16):
    QuantizeBlockLoopSymmetric<int16_t>(x, scales, idx, output);
    break;
  default:
    EXT_THROW_INVALID("unsupported data type ", output.data_type, ", ",
                      "kernel::QuantizeLinear (symmetric blocked): unsupported output dtype.");
  }
}

void QuantizeLinear::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 2);
  EXT_ENFORCE_INVALID(!(node.input_size() > 3),
                      "RunNode: op 'QuantizeLinear' expects 2 or 3 inputs, got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y_scale = GetInput(node, 1, rt.tensors());
  const Tensor *y_zero_point = GetOptionalInput(node, 2, rt.tensors());
  int64_t axis = GetAttributeIntOrDefault(node, "axis", 1);
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  if (axis < 0) {
    axis += rank;
  }
  const int64_t output_dtype = GetAttributeIntOrDefault(node, "output_dtype", 0);
  onnx_kernels::kernel::QuantizeLinear k(rt.kernel_ctx());
  if (y_zero_point != nullptr) {
    SetOutput(node, 0, k(x, y_scale, *y_zero_point, axis, &rt), rt);
  } else if (output_dtype != 0) {
    SetOutput(node, 0, k(x, y_scale, axis, static_cast<int32_t>(output_dtype), &rt), rt);
  } else {
    SetOutput(node, 0, k(x, y_scale, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
