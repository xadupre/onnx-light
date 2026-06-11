// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/cast_helper.h"

#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr std::int32_t kFloat32ExponentBias = 127;
constexpr std::int32_t kFloat16ExponentBias = 15;
} // namespace

std::uint16_t FloatToFloat16Bits(float f) noexcept {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const std::int32_t e32 = static_cast<std::int32_t>((u >> 23) & 0xffu);
  const std::uint32_t m32 = u & 0x007fffffu;
  if (e32 == 0xff) {
    // Inf / NaN: preserve sign; collapse the mantissa to a quiet-NaN marker
    // when it was non-zero.
    return static_cast<std::uint16_t>(sign | 0x7c00u | (m32 != 0 ? 0x0200u : 0u));
  }
  const std::int32_t e = e32 - kFloat32ExponentBias + kFloat16ExponentBias;
  if (e >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u); // overflow -> +/-inf
  }
  if (e <= 0) {
    if (e < -10) {
      return static_cast<std::uint16_t>(sign); // too small -> +/-0
    }
    const std::uint32_t m = (m32 | 0x00800000u) >> static_cast<std::uint32_t>(1 - e);
    const std::uint32_t round_bit = (m >> 12) & 1u;
    const std::uint32_t sticky = m & 0x00000fffu;
    std::uint16_t h = static_cast<std::uint16_t>(sign | (m >> 13));
    if (round_bit && (sticky != 0 || (h & 1))) {
      h = static_cast<std::uint16_t>(h + 1);
    }
    return h;
  }
  const std::uint32_t low = m32 & 0x1fffu;
  std::uint16_t h =
      static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10) | (m32 >> 13));
  if (low > 0x1000u || (low == 0x1000u && (h & 1u))) {
    h = static_cast<std::uint16_t>(h + 1); // mantissa carry naturally bumps exponent
  }
  return h;
}

float Float16BitsToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) >> 15) & 0x1u;
  const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
  const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3ffu;
  std::uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      std::uint32_t m = mant;
      std::int32_t e = -1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      f = (sign << 31) | (static_cast<std::uint32_t>(e + kFloat32ExponentBias + 1) << 23) |
          (m << 13);
    }
  } else if (exp == 0x1fu) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) |
        (static_cast<std::uint32_t>(exp - kFloat16ExponentBias + kFloat32ExponentBias) << 23) |
        (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
