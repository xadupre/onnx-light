// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Float8 bit-level rounding/saturation logic adapted from
// ``include/onnxruntime/core/common/float8.h`` in the Microsoft ONNX
// Runtime project (Copyright (c) Microsoft Corporation, MIT License).

#include "onnx_backend_test/kernels/tensor/cast_float8.h"

#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

inline std::uint32_t BitcastFloatToU32(float v) noexcept {
  std::uint32_t b;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

inline float BitcastU32ToFloat(std::uint32_t b) noexcept {
  float v;
  std::memcpy(&v, &b, sizeof(v));
  return v;
}

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

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
