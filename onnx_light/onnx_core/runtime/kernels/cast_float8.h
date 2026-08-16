// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Float8 ↔ float32 conversion routines used by the backend test ``Cast``
// reference kernel.
//
// The bit-level rounding/saturation logic is adapted from the
// ``include/onnxruntime/core/common/float8.h`` header in the Microsoft
// ONNX Runtime project (Copyright (c) Microsoft Corporation, licensed
// under the MIT License). Only the host (non-CUDA) saturating paths are
// reproduced because that matches the behaviour exercised by the upstream
// ONNX ``Cast`` node tests when the ``saturate`` attribute keeps its
// default value of ``1``.

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

// Saturating conversion from IEEE 754 binary32 to FLOAT8E4M3FN, returning
// the resulting 8 raw bits. Non-finite inputs round to the format-specific
// canonical NaN bit pattern (0x7F / 0xFF) for NaN, and to the largest
// finite magnitude (0x7E / 0xFE) for +/-infinity (saturate semantics).
std::uint8_t FloatToFloat8E4M3FNBits(float v) noexcept;
// Inverse of ``FloatToFloat8E4M3FNBits``.
float Float8E4M3FNBitsToFloat(std::uint8_t bits) noexcept;

// Saturating conversion to FLOAT8E4M3FNUZ (no infinities, single NaN
// pattern 0x80, no negative zero).
std::uint8_t FloatToFloat8E4M3FNUZBits(float v) noexcept;
float Float8E4M3FNUZBitsToFloat(std::uint8_t bits) noexcept;

// Saturating conversion to FLOAT8E5M2 (IEEE-754-like, with +/-infinity
// and NaN). Saturation maps +/-infinity inputs to the largest finite
// magnitudes 0x7B / 0xFB.
std::uint8_t FloatToFloat8E5M2Bits(float v) noexcept;
float Float8E5M2BitsToFloat(std::uint8_t bits) noexcept;

// Saturating conversion to FLOAT8E5M2FNUZ (no infinities, single NaN
// pattern 0x80, no negative zero).
std::uint8_t FloatToFloat8E5M2FNUZBits(float v) noexcept;
float Float8E5M2FNUZBitsToFloat(std::uint8_t bits) noexcept;

// Saturating conversion to FLOAT8E8M0 (8-bit unsigned exponent, no mantissa,
// no sign). The encoded byte stores the biased exponent ``E`` such that the
// decoded value is ``2^(E - 127)``; ``E == 0xFF`` denotes ``NaN``. Negative
// inputs (the spec calls this case "undefined") and ``NaN`` inputs map to
// the canonical ``NaN`` bit pattern ``0xFF``. With saturate semantics
// (the only mode this reference kernel implements) ``+infinity`` and finite
// values above ``2^127`` are clamped to the largest finite magnitude
// (``0xFE``, i.e. ``2^127``), and positive values below ``2^-127`` are
// clamped to ``0x00`` (i.e. ``2^-127``). The default ``round_mode`` for
// FLOAT8E8M0 is ``"up"``: positive values that are not exact powers of two
// are rounded up to the next representable power of two.
std::uint8_t FloatToFloat8E8M0Bits(float v) noexcept;
float Float8E8M0BitsToFloat(std::uint8_t bits) noexcept;

// -------------------------------------------------------------------------
// Non-saturating variants.
//
// When ``saturate=0``, values that overflow the destination range (or
// +/-infinity for FN/FNUZ types) become NaN (for types without infinity)
// or +/-infinity (for E5M2 which supports it).
// -------------------------------------------------------------------------
std::uint8_t FloatToFloat8E4M3FNBitsNoSaturate(float v) noexcept;
std::uint8_t FloatToFloat8E4M3FNUZBitsNoSaturate(float v) noexcept;
std::uint8_t FloatToFloat8E5M2BitsNoSaturate(float v) noexcept;
std::uint8_t FloatToFloat8E5M2FNUZBitsNoSaturate(float v) noexcept;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
