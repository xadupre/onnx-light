// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_sub_byte.h"

#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Truncates ``v`` toward zero. Non-finite inputs (NaN, +/-inf) are handled
// by the caller before reaching this helper for the integer narrowing
// paths, so we only need finite-value semantics here. ``std::trunc``
// preserves the input sign, so callers can mask the result directly.
inline double TruncTowardZero(float v) noexcept { return static_cast<double>(std::trunc(v)); }

inline std::uint8_t WrapToInt4(double t) noexcept {
  // ONNX Cast uses wrapping (modular) semantics for narrowing to sub-byte
  // integer types: truncate toward zero, then take the low N bits.
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x0F);
}

inline std::uint8_t WrapToUint4(double t) noexcept {
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x0Fu);
}

inline std::uint8_t WrapToInt2(double t) noexcept {
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x03);
}

inline std::uint8_t WrapToUint2(double t) noexcept {
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x03u);
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
  return WrapToInt4(TruncTowardZero(v));
}

std::uint8_t FloatToUint4Nibble(float v) noexcept {
  if (std::isnan(v)) {
    return 0x0;
  }
  return WrapToUint4(TruncTowardZero(v));
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
  return WrapToInt2(TruncTowardZero(v));
}

std::uint8_t FloatToUint2Bits(float v) noexcept {
  if (std::isnan(v)) {
    return 0x0;
  }
  return WrapToUint2(TruncTowardZero(v));
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
