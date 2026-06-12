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

#include "onnx_kernels/simple_tensor.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

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

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
