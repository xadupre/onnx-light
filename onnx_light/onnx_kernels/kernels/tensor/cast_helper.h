// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared bit-level numeric conversion helpers used by reference kernels and
// backend test case generators. The float16 ↔ float32 routines implement the
// IEEE-754 binary16 saturating, round-half-to-even conversion used across
// kernels (Mod, Range, CausalConvWithState, DequantizeLinear, ...) and the
// backend test case builders that need to encode FLOAT16 tensors from
// float32 samples (Attention, QLinearMatMul, ...).

#pragma once

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

// Saturating, round-half-to-even conversion of an IEEE-754 binary32 value to
// the raw 16-bit pattern of its IEEE-754 binary16 counterpart. Overflow
// rounds to +/-infinity, underflow to +/-0, and NaN inputs are preserved as
// quiet NaNs with the input's sign.
std::uint16_t FloatToFloat16Bits(float f) noexcept;

// Inverse of :cpp:func:`FloatToFloat16Bits`. Decodes the raw 16-bit pattern
// of an IEEE-754 binary16 value into its IEEE-754 binary32 counterpart.
float Float16BitsToFloat(std::uint16_t h) noexcept;

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
