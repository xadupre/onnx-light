// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared bit-level numeric conversion helpers and small Tensor builders used
// by reference kernels and backend test case generators. The float16 ↔
// float32 routines implement the IEEE-754 binary16 saturating,
// round-half-to-even conversion used across kernels (Mod, Range,
// CausalConvWithState, DequantizeLinear, ...) and the backend test case
// builders that need to encode FLOAT16 tensors from float32 samples
// (Attention, QLinearMatMul, ...). The bfloat16 conversion and sub-byte
// packing helpers are similarly shared between the Range kernel and the
// QuantizeLinear / DequantizeLinear backend test cases.

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

// ---------------------------------------------------------------------------
// Bit-level numeric conversions
// ---------------------------------------------------------------------------

// Saturating, round-half-to-even conversion of an IEEE-754 binary32 value to
// the raw 16-bit pattern of its IEEE-754 binary16 counterpart. Overflow
// rounds to +/-infinity, underflow to +/-0, and NaN inputs are preserved as
// quiet NaNs with the input's sign.
std::uint16_t FloatToFloat16Bits(float f) noexcept;

// Inverse of :cpp:func:`FloatToFloat16Bits`. Decodes the raw 16-bit pattern
// of an IEEE-754 binary16 value into its IEEE-754 binary32 counterpart.
float Float16BitsToFloat(std::uint16_t h) noexcept;

// Round-to-nearest-even ``float`` -> ``bfloat16`` encoder. Matches the
// behaviour of the upstream ``onnx.helper`` bfloat16 helpers. NaN inputs are
// preserved as quiet NaNs.
std::uint16_t FloatToBfloat16Bits(float f) noexcept;

// Inverse of :cpp:func:`FloatToBfloat16Bits`. Decodes the raw 16-bit
// ``bfloat16`` pattern (the upper 16 bits of an IEEE-754 binary32 value)
// into its IEEE-754 binary32 counterpart.
float Bfloat16BitsToFloat(std::uint16_t b) noexcept;

// Encodes a single ``float`` into the 4-bit FLOAT4E2M1 nibble (2 exponent
// bits, 1 mantissa bit). The format supports the values
// ``{+/-0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}``; values outside
// this set raise ``std::invalid_argument``.
std::uint8_t FloatToFloat4E2M1Nibble(float v);

// Decodes a single FLOAT4E2M1 nibble to a ``float32`` value. The nibble is
// interpreted using the same bit layout as :func:`FloatToFloat4E2M1Nibble`.
float Float4E2M1NibbleToFloat(std::uint8_t nibble) noexcept;

// Encodes a ``float`` to the nearest representable FLOAT4E2M1 nibble, using
// round-half-to-even tie-breaking. Values outside the representable range
// ``[-6, 6]`` are saturated to ``+/-6``; ``NaN`` maps to ``+0``.
std::uint8_t FloatRoundToFloat4E2M1Nibble(float v) noexcept;

// Packs ``values`` (one element per ``int8_t`` entry, range checked by the
// caller) into a 4-bit-per-element little-endian buffer matching the ONNX
// sub-byte layout (low nibble first per byte, trailing slot zero-padded).
std::vector<std::uint8_t> Pack4Bit(const std::vector<std::int8_t> &values);

// Packs ``values`` (one element per ``int8_t`` entry) into a 2-bit-per-
// element little-endian buffer matching the ONNX sub-byte layout (lowest
// pair first per byte, trailing slots zero-padded).
std::vector<std::uint8_t> Pack2Bit(const std::vector<std::int8_t> &values);

// ---------------------------------------------------------------------------
// Tensor builders
// ---------------------------------------------------------------------------

// Builds a FLOAT16 tensor of the requested shape from a flattened list of
// float32 sample values rounded via :cpp:func:`FloatToFloat16Bits`.
Tensor MakeFloat16Tensor(const std::string &name, const Shape &shape,
                         const std::vector<float> &values, RawBufferAllocator *allocator = nullptr);

// Builds a FLOAT16 scalar tensor from a ``float`` sample value.
Tensor MakeFloat16Scalar(const std::string &name, float value,
                         RawBufferAllocator *allocator = nullptr);

// Builds a BFLOAT16 tensor of the requested shape from a flattened list of
// float32 sample values rounded via :cpp:func:`FloatToBfloat16Bits`.
Tensor MakeBfloat16Tensor(const std::string &name, const Shape &shape,
                          const std::vector<float> &values,
                          RawBufferAllocator *allocator = nullptr);

// Builds a BFLOAT16 scalar tensor from a ``float`` sample value.
Tensor MakeBfloat16Scalar(const std::string &name, float value,
                          RawBufferAllocator *allocator = nullptr);

// Encodes a FLOAT tensor as a FLOAT16 tensor by round-tripping every
// element through :cpp:func:`FloatToFloat16Bits`. The caller-provided
// ``name`` becomes the tensor name on the resulting ``Tensor``.
Tensor FloatToFloat16Tensor(const std::string &name, const Tensor &f,
                            RawBufferAllocator *allocator = nullptr);

// Round-trips every element of a FLOAT tensor through
// :cpp:func:`FloatToFloat16Bits` / :cpp:func:`Float16BitsToFloat` and
// returns a fresh FLOAT tensor reflecting the FP16 storage precision. Used
// to simulate the input-side rounding that happens when FLOAT16 tensors
// are fed into a backend that internally promotes to FLOAT.
Tensor RoundToFloat16(const Tensor &f, RawBufferAllocator *allocator = nullptr);

// Builds a UINT16 scalar tensor carrying the supplied zero-point value.
Tensor Uint16ZeroPoint(std::uint16_t value, RawBufferAllocator *allocator = nullptr);

// Builds an INT16 scalar tensor carrying the supplied zero-point value.
Tensor Int16ZeroPoint(std::int16_t value, RawBufferAllocator *allocator = nullptr);

// Builds a 1-D float8 tensor from the float32 sample values in ``values``.
// ``encode`` is the saturating ``FloatToFloat8*Bits``.
Tensor MakeFloat8Tensor(DataType dtype, const Shape &shape, const std::vector<float> &values,
                        std::uint8_t (*encode)(float) noexcept,
                        RawBufferAllocator *allocator = nullptr);

// Builds a sub-byte tensor with the supplied ``dtype`` and ``shape`` from a
// flattened list of element values. ``bits`` must be 4 or 2.
Tensor MakeSubByteTensor(DataType dtype, const Shape &shape, const std::vector<std::int8_t> &values,
                         int bits, RawBufferAllocator *allocator = nullptr);

// Builds a FLOAT4E2M1 tensor of the requested shape from a flattened list
// of float32 sample values; every input must be representable in FLOAT4E2M1
// (see :cpp:func:`FloatToFloat4E2M1Nibble`).
Tensor MakeFloat4E2M1Tensor(const Shape &shape, const std::vector<float> &values,
                            RawBufferAllocator *allocator = nullptr);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
