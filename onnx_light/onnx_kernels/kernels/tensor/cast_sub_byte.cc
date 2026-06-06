// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/cast_sub_byte.h"

#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Truncates ``v`` toward zero. Non-finite inputs (NaN, +/-inf) are handled
// by the caller before reaching this helper for the integer narrowing
// paths, so we only need finite-value semantics here. ``std::trunc``
// preserves the input sign, so callers can clamp the result directly.
inline double TruncTowardZero(float v) noexcept { return static_cast<double>(std::trunc(v)); }

inline std::uint8_t ClampToInt4(double t) noexcept {
  if (t <= -8.0)
    return 0x8; // -8 as a 4-bit signed value (two's complement low nibble)
  if (t >= 7.0)
    return 0x7;
  // Map negative values into [-8, -1] → [0x8, 0xF].
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x0F);
}

inline std::uint8_t ClampToUint4(double t) noexcept {
  if (t <= 0.0)
    return 0x0;
  if (t >= 15.0)
    return 0xF;
  return static_cast<std::uint8_t>(static_cast<unsigned>(t) & 0x0Fu);
}

inline std::uint8_t ClampToInt2(double t) noexcept {
  // Int2 representable range is [-2, 1] (two's complement on 2 bits).
  if (t <= -2.0)
    return 0x2; // -2
  if (t >= 1.0)
    return 0x1;
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x03);
}

inline std::uint8_t ClampToUint2(double t) noexcept {
  if (t <= 0.0)
    return 0x0;
  if (t >= 3.0)
    return 0x3;
  return static_cast<std::uint8_t>(static_cast<unsigned>(t) & 0x03u);
}

} // namespace

// ---------------------------------------------------------------------------
// 4-bit element conversions.
// ---------------------------------------------------------------------------

std::uint8_t FloatToInt4Nibble(float v) noexcept {
  if (std::isnan(v)) {
    // Match ``ml_dtypes.int4`` round-toward-zero conversion of NaN to 0.
    return 0x0;
  }
  return ClampToInt4(TruncTowardZero(v));
}

std::uint8_t FloatToUint4Nibble(float v) noexcept {
  if (std::isnan(v)) {
    return 0x0;
  }
  return ClampToUint4(TruncTowardZero(v));
}

std::int8_t Int4NibbleToInt8(std::uint8_t nibble) noexcept {
  const std::uint8_t low = static_cast<std::uint8_t>(nibble & 0x0F);
  // Sign-extend the 4-bit two's complement value into an 8-bit int.
  if (low & 0x8) {
    return static_cast<std::int8_t>(static_cast<int>(low) - 16);
  }
  return static_cast<std::int8_t>(low);
}

std::uint8_t Uint4NibbleToUint8(std::uint8_t nibble) noexcept {
  return static_cast<std::uint8_t>(nibble & 0x0F);
}

// ---------------------------------------------------------------------------
// 2-bit element conversions.
// ---------------------------------------------------------------------------

std::uint8_t FloatToInt2Bits(float v) noexcept {
  if (std::isnan(v)) {
    return 0x0;
  }
  return ClampToInt2(TruncTowardZero(v));
}

std::uint8_t FloatToUint2Bits(float v) noexcept {
  if (std::isnan(v)) {
    return 0x0;
  }
  return ClampToUint2(TruncTowardZero(v));
}

std::int8_t Int2BitsToInt8(std::uint8_t bits) noexcept {
  const std::uint8_t low = static_cast<std::uint8_t>(bits & 0x03);
  if (low & 0x2) {
    return static_cast<std::int8_t>(static_cast<int>(low) - 4);
  }
  return static_cast<std::int8_t>(low);
}

std::uint8_t Uint2BitsToUint8(std::uint8_t bits) noexcept {
  return static_cast<std::uint8_t>(bits & 0x03);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
