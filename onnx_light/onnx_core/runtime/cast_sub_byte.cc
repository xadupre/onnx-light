// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_sub_byte.h"

#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

// Truncates ``v`` toward zero. Non-finite inputs (NaN, +/-inf) are handled
// by the caller before reaching this helper for the integer narrowing
// paths, so we only need finite-value semantics here. ``std::trunc``
// preserves the input sign, so callers can clamp the result directly.
inline double TruncTowardZero(float v) noexcept { return static_cast<double>(std::trunc(v)); }

// Guard value for int64 conversion: float32 max (~3.4e38) exceeds INT64_MAX
// (~9.2e18), so large finite floats must be excluded before casting to
// int64_t to avoid undefined behaviour. Values beyond this threshold are
// exact multiples of high powers of 2, so their low nibble/bit-pair is 0.
static constexpr double kSafeInt64Range = 9.2e18;

inline std::uint8_t ClampToInt4(double t) noexcept {
  // Match ml_dtypes.int4 wrapping semantics: truncate toward zero, keep low
  // 4 bits.
  if (t >= kSafeInt64Range || t <= -kSafeInt64Range)
    return 0x0;
  const int64_t v = static_cast<int64_t>(t);
  return static_cast<std::uint8_t>(static_cast<uint64_t>(v) & 0x0FU);
}

inline std::uint8_t ClampToUint4(double t) noexcept {
  // Match ml_dtypes.uint4 wrapping semantics: truncate toward zero, keep low
  // 4 bits.
  if (t >= kSafeInt64Range || t <= -kSafeInt64Range)
    return 0x0;
  const int64_t v = static_cast<int64_t>(t);
  return static_cast<std::uint8_t>(static_cast<uint64_t>(v) & 0x0FU);
}

inline std::uint8_t ClampToInt2(double t) noexcept {
  // Match `ml_dtypes.int2` cast semantics used by ONNX backend tests:
  // truncate toward zero, then keep the low 2 bits (two's complement wrap).
  const int v = static_cast<int>(t);
  return static_cast<std::uint8_t>(v & 0x03);
}

inline std::uint8_t ClampToUint2(double t) noexcept {
  // Match ml_dtypes.uint2 wrapping semantics: truncate toward zero, keep low
  // 2 bits.
  if (t >= kSafeInt64Range || t <= -kSafeInt64Range)
    return 0x0;
  const int64_t v = static_cast<int64_t>(t);
  return static_cast<std::uint8_t>(static_cast<uint64_t>(v) & 0x03U);
}

} // namespace

// ---------------------------------------------------------------------------
// 4-bit element conversions.
// ---------------------------------------------------------------------------

std::uint8_t FloatToInt4Nibble(float v) noexcept {
  if (!std::isfinite(v)) {
    // Match ml_dtypes.int4 conversion of non-finite inputs to 0.
    return 0x0;
  }
  return ClampToInt4(TruncTowardZero(v));
}

std::uint8_t FloatToUint4Nibble(float v) noexcept {
  if (!std::isfinite(v)) {
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
  if (!std::isfinite(v)) {
    return 0x0;
  }
  return ClampToInt2(TruncTowardZero(v));
}

std::uint8_t FloatToUint2Bits(float v) noexcept {
  if (!std::isfinite(v)) {
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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
