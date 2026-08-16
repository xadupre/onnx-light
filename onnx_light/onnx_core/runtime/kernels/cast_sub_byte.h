// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Sub-byte (INT4 / UINT4 / INT2 / UINT2) ↔ wider-type conversion routines used
// by the backend test ``Cast`` reference kernel.
//
// The element-level conversions implement the saturating semantics of the
// ONNX ``Cast`` operator with its default ``saturate=1`` attribute: source
// values outside the destination 4-bit / 2-bit representable range are
// clamped to the nearest representable bound. Floating-point inputs are
// rounded toward zero (mirroring C++ ``static_cast<int>(float)``).
//
// The packing layout matches the ONNX TensorProto wire format used for
// INT4/UINT4/INT2/UINT2: two 4-bit elements per byte (low nibble first) and
// four 2-bit elements per byte (least significant pair first), with the
// trailing nibble/bit-pair zero-padded when the element count is not a
// multiple of the packing factor.

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

// ---------------------------------------------------------------------------
// Element-level saturating conversions to the unpacked 4-bit value carried in
// the low nibble of a uint8_t. The high nibble of the returned byte is zero.
// ---------------------------------------------------------------------------
std::uint8_t FloatToInt4Nibble(float v) noexcept;
std::uint8_t FloatToUint4Nibble(float v) noexcept;
std::int8_t Int4NibbleToInt8(std::uint8_t nibble) noexcept;
std::uint8_t Uint4NibbleToUint8(std::uint8_t nibble) noexcept;

// ---------------------------------------------------------------------------
// Element-level saturating conversions to the unpacked 2-bit value carried in
// the low two bits of a uint8_t. The upper six bits of the returned byte are
// zero.
// ---------------------------------------------------------------------------
std::uint8_t FloatToInt2Bits(float v) noexcept;
std::uint8_t FloatToUint2Bits(float v) noexcept;
std::int8_t Int2BitsToInt8(std::uint8_t bits) noexcept;
std::uint8_t Uint2BitsToUint8(std::uint8_t bits) noexcept;

// ---------------------------------------------------------------------------
// Packed sub-byte element read/write helpers shared across kernels.
//
// The packing layout matches the ONNX TensorProto convention:
//   * INT4 / UINT4 / FLOAT4E2M1 — two 4-bit elements per byte, low nibble
//     holds the even-indexed element (flat index 2*i), high nibble holds the
//     odd-indexed element (flat index 2*i + 1).
//   * INT2 / UINT2 — four 2-bit elements per byte, packed least-significant
//     pair first (flat index 4*i in bits 0–1, 4*i+1 in bits 2–3, etc.).
// ---------------------------------------------------------------------------
inline std::uint8_t Read4BitElement(const std::uint8_t *data, int64_t i) noexcept {
  const std::uint8_t byte = data[i / 2];
  return static_cast<std::uint8_t>((i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F));
}

inline void Write4BitElement(std::uint8_t *data, int64_t i, std::uint8_t nibble) noexcept {
  std::uint8_t &byte = data[i / 2];
  if (i % 2 == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xF0) | (nibble & 0x0F));
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0F) | ((nibble & 0x0F) << 4));
  }
}

inline std::uint8_t Read2BitElement(const std::uint8_t *data, int64_t i) noexcept {
  const std::uint8_t byte = data[i / 4];
  const int shift = static_cast<int>((i % 4) * 2);
  return static_cast<std::uint8_t>((byte >> shift) & 0x03);
}

inline void Write2BitElement(std::uint8_t *data, int64_t i, std::uint8_t bits) noexcept {
  std::uint8_t &byte = data[i / 4];
  const int shift = static_cast<int>((i % 4) * 2);
  const std::uint8_t mask = static_cast<std::uint8_t>(0x03u << shift);
  byte = static_cast<std::uint8_t>((byte & ~mask) | (((bits & 0x03) << shift) & mask));
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
