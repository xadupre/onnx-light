// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Float8 bit-level rounding/saturation logic adapted from
// ``include/onnxruntime/core/common/float8.h`` in the Microsoft ONNX
// Runtime project (Copyright (c) Microsoft Corporation, MIT License).

#include "onnx_core/runtime/kernels/cast_float8.h"

#include <bit>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

inline std::uint32_t BitcastFloatToU32(float v) noexcept { return std::bit_cast<std::uint32_t>(v); }

inline float BitcastU32ToFloat(std::uint32_t b) noexcept { return std::bit_cast<float>(b); }

} // namespace

// ---------------------------------------------------------------------------
// FLOAT8E4M3FN — bias 7, max finite magnitude 448, NaN at S.1111.111.
// ---------------------------------------------------------------------------
std::uint8_t FloatToFloat8E4M3FNBits(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  std::uint8_t val = static_cast<std::uint8_t>((b & 0x80000000u) >> 24); // sign
  if ((b & 0x7FFFFFFFu) == 0x7F800000u) {
    // +/- infinity: saturate to largest finite magnitude.
    val |= 126;
  } else if ((b & 0x7F800000u) == 0x7F800000u) {
    // NaN.
    val |= 0x7F;
  } else {
    const std::uint8_t e = static_cast<std::uint8_t>((b & 0x7F800000u) >> 23);
    const std::uint32_t m = b & 0x007FFFFFu;
    if (e != 0) {
      if (e < 117) {
        // Underflow to zero.
      } else if (e < 121) {
        // Subnormal in the destination format.
        const auto d = 120 - e;
        if (d < 3) {
          val |= 1u << (2 - d);
          val |= m >> (21 + d);
        } else if (m > 0) {
          val |= 1;
        }
        const auto mask = 1u << (20 + d);
        if ((m & mask) && ((val & 1u) || ((m & (mask - 1)) > 0) ||
                           ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
          val += 1;
        }
      } else if (e < 136) {
        // Normalized in the destination format.
        const auto ex = e - 120;
        if (ex == 0) {
          val |= 0x4;
          val |= m >> 21;
        } else {
          val |= ex << 3;
          val |= m >> 20;
          if ((val & 0x7F) == 0x7F) {
            val &= 0xFE;
          }
        }
        if ((m & 0x80000u) && ((m & 0x100000u) || (m & 0x7FFFFu))) {
          if ((val & 0x7F) < 0x7E) {
            val += 1;
          }
        }
      } else {
        // Overflow: saturate.
        val |= 126;
      }
    }
  }
  return val;
}

float Float8E4M3FNBitsToFloat(std::uint8_t val) noexcept {
  std::uint32_t res;
  if (val == 255) {
    res = 0xFFC00000u;
  } else if (val == 127) {
    res = 0x7FC00000u;
  } else {
    std::uint32_t expo = (val & 0x78u) >> 3;
    std::uint32_t mant = val & 0x07u;
    const std::uint32_t sign = val & 0x80u;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7Fu - 7u;
        if ((mant & 0x4u) == 0) {
          mant &= 0x3u;
          mant <<= 1;
          expo -= 1;
        }
        if ((mant & 0x4u) == 0) {
          mant &= 0x3u;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x3u) << 21;
        res |= expo << 23;
      }
    } else {
      res |= mant << 20;
      expo -= 0x7u;
      expo += 0x7Fu;
      res |= expo << 23;
    }
  }
  return BitcastU32ToFloat(res);
}

// ---------------------------------------------------------------------------
// FLOAT8E4M3FNUZ — bias 8, no infinities, single NaN pattern 0x80,
// no negative zero.
// ---------------------------------------------------------------------------
std::uint8_t FloatToFloat8E4M3FNUZBits(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  std::uint8_t val = static_cast<std::uint8_t>((b & 0x80000000u) >> 24); // sign
  if ((b & 0x7FFFFFFFu) == 0x7F800000u) {
    // +/- infinity: saturate to largest finite magnitude.
    val |= 0x7F;
  } else if ((b & 0x7F800000u) == 0x7F800000u) {
    // NaN.
    val = 0x80;
  } else {
    const std::uint8_t e = static_cast<std::uint8_t>((b & 0x7F800000u) >> 23);
    const std::uint32_t m = b & 0x007FFFFFu;
    if (e < 116) {
      val = 0;
    } else if (e < 120) {
      const auto d = 119 - e;
      if (d < 3) {
        val |= 1u << (2 - d);
        val |= m >> (21 + d);
      } else if (m > 0) {
        val |= 1;
      } else {
        val = 0;
      }
      const auto mask = 1u << (20 + d);
      if ((m & mask) && ((val & 1u) || ((m & (mask - 1)) > 0) ||
                         ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
        val += 1;
      }
    } else if (e < 135) {
      const auto ex = e - 119;
      if (ex == 0) {
        val |= 0x4;
        val |= m >> 21;
      } else {
        val |= ex << 3;
        val |= m >> 20;
      }
      if ((m & 0x80000u) && ((m & 0x100000u) || (m & 0x7FFFFu))) {
        if ((val & 0x7F) < 0x7F) {
          val += 1;
        }
      }
    } else {
      val |= 0x7F;
    }
  }
  return val;
}

float Float8E4M3FNUZBitsToFloat(std::uint8_t val) noexcept {
  std::uint32_t res;
  if (val == 0x80) {
    res = 0xFFC00000u;
  } else {
    std::uint32_t expo = (val & 0x78u) >> 3;
    std::uint32_t mant = val & 0x07u;
    const std::uint32_t sign = val & 0x80u;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7Fu - 8u;
        if ((mant & 0x4u) == 0) {
          mant &= 0x3u;
          mant <<= 1;
          expo -= 1;
        }
        if ((mant & 0x4u) == 0) {
          mant &= 0x3u;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x3u) << 21;
        res |= expo << 23;
      }
    } else {
      res |= mant << 20;
      expo -= 8u;
      expo += 0x7Fu;
      res |= expo << 23;
    }
  }
  return BitcastU32ToFloat(res);
}

// ---------------------------------------------------------------------------
// FLOAT8E5M2 — bias 15, IEEE-754-like with +/-infinity and NaN.
// Saturating mode maps +/-infinity to the largest finite magnitudes
// 0x7B / 0xFB.
// ---------------------------------------------------------------------------
std::uint8_t FloatToFloat8E5M2Bits(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  std::uint8_t val = static_cast<std::uint8_t>((b & 0x80000000u) >> 24); // sign
  if ((b & 0x7FFFFFFFu) == 0x7F800000u) {
    // +/- infinity: saturate to largest finite magnitude.
    val |= 0x7B;
  } else if ((b & 0x7F800000u) == 0x7F800000u) {
    // NaN. Use the 0x7E mantissa pattern (low bit cleared) to match the
    // canonical encoding ONNX's ``numpy_helper.saturate_cast`` produces
    // via ``ml_dtypes`` for ``float8_e5m2``.
    val |= 0x7E;
  } else {
    const std::uint32_t e = (b & 0x7F800000u) >> 23;
    const std::uint32_t m = b & 0x007FFFFFu;
    if (e != 0) {
      if (e < 110) {
        // Underflow to zero.
      } else if (e < 113) {
        const auto d = 112 - e;
        if (d < 2) {
          val |= 1u << (1 - d);
          val |= m >> (22 + d);
        } else if (m > 0) {
          val |= 1;
        }
        const auto mask = 1u << (21 + d);
        if ((m & mask) && ((val & 1u) || ((m & (mask - 1)) > 0) ||
                           ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
          val += 1;
        }
      } else if (e < 143) {
        const auto ex = e - 112;
        val |= ex << 2;
        val |= m >> 21;
        if ((m & 0x100000u) && ((m & 0xFFFFFu) || (m & 0x200000u))) {
          if ((val & 0x7F) < 0x7B) {
            val += 1;
          } else {
            val |= 0x7B;
          }
        }
      } else {
        val |= 0x7B;
      }
    }
  }
  return val;
}

float Float8E5M2BitsToFloat(std::uint8_t val) noexcept {
  std::uint32_t res;
  if (val >= 253) {
    res = 0xFFC00000u;
  } else if (val >= 125 && val <= 127) {
    res = 0x7FC00000u;
  } else if (val == 252) {
    res = 0xFF800000u;
  } else if (val == 124) {
    res = 0x7F800000u;
  } else {
    std::uint32_t expo = (val & 0x7Cu) >> 2;
    std::uint32_t mant = val & 0x03u;
    const std::uint32_t sign = val & 0x80u;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7Fu - 15u;
        if ((mant & 0x2u) == 0) {
          mant &= 0x1u;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x1u) << 22;
        res |= expo << 23;
      }
    } else {
      res |= mant << 21;
      expo -= 15u;
      expo += 0x7Fu;
      res |= expo << 23;
    }
  }
  return BitcastU32ToFloat(res);
}

// ---------------------------------------------------------------------------
// FLOAT8E5M2FNUZ — bias 16, no infinities, single NaN pattern 0x80,
// no negative zero.
// ---------------------------------------------------------------------------
std::uint8_t FloatToFloat8E5M2FNUZBits(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  std::uint8_t val = static_cast<std::uint8_t>((b & 0x80000000u) >> 24); // sign
  if ((b & 0x7FFFFFFFu) == 0x7F800000u) {
    // +/- infinity: saturate to largest finite magnitude.
    val |= 0x7F;
  } else if ((b & 0x7F800000u) == 0x7F800000u) {
    val = 0x80;
  } else {
    const std::uint32_t e = (b & 0x7F800000u) >> 23;
    const std::uint32_t m = b & 0x007FFFFFu;
    if (e < 109) {
      val = 0;
    } else if (e < 112) {
      const auto d = 111 - e;
      if (d < 2) {
        val |= 1u << (1 - d);
        val |= m >> (22 + d);
      } else if (m > 0) {
        val |= 1;
      } else {
        val = 0;
      }
      const auto mask = 1u << (21 + d);
      if ((m & mask) && ((val & 1u) || ((m & (mask - 1)) > 0) ||
                         ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
        val += 1;
      }
    } else if (e < 143) {
      const auto ex = e - 111;
      val |= ex << 2;
      val |= m >> 21;
      if ((m & 0x100000u) && ((m & 0xFFFFFu) || (m & 0x200000u))) {
        if ((val & 0x7F) < 0x7F) {
          val += 1;
        }
      }
    } else {
      val |= 0x7F;
    }
  }
  return val;
}

float Float8E5M2FNUZBitsToFloat(std::uint8_t val) noexcept {
  std::uint32_t res;
  if (val == 0x80) {
    res = 0xFFC00000u;
  } else {
    std::uint32_t expo = (val & 0x7Cu) >> 2;
    std::uint32_t mant = val & 0x03u;
    const std::uint32_t sign = val & 0x80u;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7Fu - 16u;
        if ((mant & 0x2u) == 0) {
          mant &= 0x1u;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x1u) << 22;
        res |= expo << 23;
      }
    } else {
      res |= mant << 21;
      expo -= 16u;
      expo += 0x7Fu;
      res |= expo << 23;
    }
  }
  return BitcastU32ToFloat(res);
}

// ---------------------------------------------------------------------------
// FLOAT8E8M0 — 8-bit unsigned biased exponent, no mantissa, no sign.
//
// Bit layout (one byte): ``EEEEEEEE``. Decoded value: ``2^(E - 127)`` for
// ``E`` in ``[0, 254]``, and ``NaN`` when ``E == 255 (0xFF)``. The format
// cannot represent zero, negative numbers or non-power-of-two values, so the
// reference encoder implements the spec's default behaviour
// (``round_mode="up"`` and ``saturate=1``):
//
// * ``NaN`` input or negative input (incl. ``-infinity`` and ``-0``)
//   → ``0xFF`` (canonical NaN). The spec calls negative inputs
//   "undefined"; we follow the ml_dtypes convention of mapping them to NaN
//   so the encoder is deterministic and total.
// * ``+infinity`` or finite positive values strictly above ``2^127``
//   → ``0xFE`` (largest finite magnitude, saturation).
// * Positive finite values at or below ``2^-127`` (including ``+0``)
//   → ``0x00`` (smallest finite magnitude, saturation).
// * Other positive finite values are rounded *up* to the nearest power of
//   two: ``2^E`` → ``E + 127``; values in the open interval
//   ``(2^E, 2^(E+1))`` round to ``2^(E+1)``, i.e. ``E + 128``.
// ---------------------------------------------------------------------------
std::uint8_t FloatToFloat8E8M0Bits(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  const std::uint32_t sign = b & 0x80000000u;
  const std::uint32_t exp = (b & 0x7F800000u) >> 23;
  const std::uint32_t mant = b & 0x007FFFFFu;
  // NaN -> canonical NaN.
  if (exp == 0xFFu && mant != 0u) {
    return 0xFFu;
  }
  // Negative inputs (incl. -infinity and -0): spec leaves this case
  // undefined; map to NaN for a deterministic, total encoder.
  if (sign != 0u) {
    return 0xFFu;
  }
  // +infinity -> saturate to largest finite (2^127, i.e. bits=254).
  if (exp == 0xFFu) {
    return 0xFEu;
  }
  // Finite positive values, including +0 and subnormals.
  // Treat +0 / subnormals (exp == 0) as below 2^-127 -> bits=0.
  if (exp == 0u) {
    return 0x00u;
  }
  // Normal range. exp is the IEEE-754 biased exponent (1..254 -> unbiased
  // -126..127). Round mode is "up": exact powers of two keep their
  // exponent; anything with a non-zero mantissa rounds up by 1.
  std::uint32_t bits = exp;
  if (mant != 0u) {
    bits += 1u;
  }
  // Saturate above 2^127: bits=255 is reserved for NaN.
  if (bits >= 0xFFu) {
    return 0xFEu;
  }
  return static_cast<std::uint8_t>(bits);
}

float Float8E8M0BitsToFloat(std::uint8_t bits) noexcept {
  if (bits == 0xFFu) {
    // Canonical NaN: build a quiet NaN with sign 0.
    return BitcastU32ToFloat(0x7FC00000u);
  }
  // Value = 2^(bits - 127). Construct the float by placing ``bits`` in
  // the IEEE-754 binary32 exponent field with sign=0 and mantissa=0.
  return BitcastU32ToFloat(static_cast<std::uint32_t>(bits) << 23);
}

// ---------------------------------------------------------------------------
// Non-saturating variants.
// ---------------------------------------------------------------------------

std::uint8_t FloatToFloat8E4M3FNBitsNoSaturate(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  const std::uint8_t sign = static_cast<std::uint8_t>((b & 0x80000000u) >> 24);
  // +/- infinity or NaN -> NaN (sign-preserving: 0x7F or 0xFF).
  if ((b & 0x7F800000u) == 0x7F800000u) {
    return sign | 0x7F;
  }
  // For finite values, delegate to the saturating path and check for overflow.
  std::uint8_t val = FloatToFloat8E4M3FNBits(v);
  // The saturating path maps overflow to 0x7E/0xFE (max finite). In
  // non-saturating mode, overflow must become NaN instead.
  if ((val & 0x7F) == 0x7E) {
    // Check if the original was actually at max magnitude (448.0) or
    // below: if so it's legitimate, otherwise it overflowed.
    const std::uint32_t abs_bits = b & 0x7FFFFFFFu;
    // 448.0 = 0x43E00000
    if (abs_bits > 0x43E00000u) {
      return sign | 0x7F; // NaN
    }
  }
  return val;
}

std::uint8_t FloatToFloat8E4M3FNUZBitsNoSaturate(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  // +/- infinity or NaN -> NaN (0x80).
  if ((b & 0x7F800000u) == 0x7F800000u) {
    return 0x80;
  }
  // For finite values, delegate to the saturating path and check for overflow.
  std::uint8_t val = FloatToFloat8E4M3FNUZBits(v);
  // The saturating path maps overflow to sign|0x7F (max finite). In
  // non-saturating mode, overflow must become NaN (0x80).
  if ((val & 0x7F) == 0x7F && val != 0x80) {
    const std::uint32_t abs_bits = b & 0x7FFFFFFFu;
    // Max finite for E4M3FNUZ is 240.0 = 0x43700000.
    if (abs_bits > 0x43700000u) {
      return 0x80; // NaN
    }
  }
  return val;
}

std::uint8_t FloatToFloat8E5M2BitsNoSaturate(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  const std::uint8_t sign = static_cast<std::uint8_t>((b & 0x80000000u) >> 24);
  // NaN -> NaN (sign-preserving).
  if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0) {
    return sign | 0x7E; // canonical NaN pattern
  }
  // +/- infinity -> +/- infinity (E5M2 has infinity representation).
  if ((b & 0x7FFFFFFFu) == 0x7F800000u) {
    return sign | 0x7C; // infinity in E5M2
  }
  // For finite values, delegate to saturating path and check for overflow.
  std::uint8_t val = FloatToFloat8E5M2Bits(v);
  // The saturating path maps overflow to 0x7B/0xFB (max finite). In
  // non-saturating mode, overflow must become +/-infinity.
  if ((val & 0x7F) == 0x7B) {
    const std::uint32_t abs_bits = b & 0x7FFFFFFFu;
    // Max finite for E5M2 is 57344.0 = 0x47600000.
    if (abs_bits > 0x47600000u) {
      return sign | 0x7C; // infinity
    }
  }
  return val;
}

std::uint8_t FloatToFloat8E5M2FNUZBitsNoSaturate(float v) noexcept {
  const std::uint32_t b = BitcastFloatToU32(v);
  // +/- infinity or NaN -> NaN (0x80).
  if ((b & 0x7F800000u) == 0x7F800000u) {
    return 0x80;
  }
  // For finite values, delegate to saturating path and check for overflow.
  std::uint8_t val = FloatToFloat8E5M2FNUZBits(v);
  // The saturating path maps overflow to sign|0x7F (max finite). In
  // non-saturating mode, overflow must become NaN (0x80).
  if ((val & 0x7F) == 0x7F && val != 0x80) {
    const std::uint32_t abs_bits = b & 0x7FFFFFFFu;
    // Max finite for E5M2FNUZ is 57344.0 = 0x47600000.
    if (abs_bits > 0x47600000u) {
      return 0x80; // NaN
    }
  }
  return val;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
