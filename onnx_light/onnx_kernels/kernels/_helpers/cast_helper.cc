// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr std::int32_t kFloat32ExponentBias = 127;
constexpr std::int32_t kFloat16ExponentBias = 15;

Tensor MakeScalarTensor(std::int32_t data_type, const std::vector<std::uint8_t> &bytes) {
  return Tensor("", data_type, /*shape=*/{}, bytes);
}
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
      // Subnormal half: normalize the mantissa so its implicit leading 1 sits
      // at bit 10, decrementing the (biased-by-one) half exponent for every
      // left shift, then rebias into the float32 exponent field.
      std::uint32_t m = mant;
      std::int32_t e = 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3ffu;
      f = (sign << 31) |
          (static_cast<std::uint32_t>(e - kFloat16ExponentBias + kFloat32ExponentBias) << 23) |
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

std::uint16_t FloatToBfloat16Bits(float f) noexcept {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // NaN: preserve a quiet NaN (non-zero mantissa) in the upper 16 bits.
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
    return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  }
  // Round-to-nearest-even on the lower 16 bits.
  const std::uint32_t rounding_bias = 0x00007fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>((u + rounding_bias) >> 16);
}

float Bfloat16BitsToFloat(std::uint16_t b) noexcept {
  std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::uint8_t FloatToFloat4E2M1Nibble(float v) {
  struct Entry {
    float value;
    std::uint8_t bits;
  };
  static const Entry kTable[] = {
      {0.0f, 0x0},  {0.5f, 0x1},  {1.0f, 0x2},  {1.5f, 0x3},  {2.0f, 0x4},  {3.0f, 0x5},
      {4.0f, 0x6},  {6.0f, 0x7},  {-0.0f, 0x8}, {-0.5f, 0x9}, {-1.0f, 0xA}, {-1.5f, 0xB},
      {-2.0f, 0xC}, {-3.0f, 0xD}, {-4.0f, 0xE}, {-6.0f, 0xF},
  };
  for (const auto &e : kTable) {
    if (e.value == v && std::signbit(e.value) == std::signbit(v)) {
      return e.bits;
    }
  }
  throw std::invalid_argument("FloatToFloat4E2M1Nibble: value not representable in FLOAT4E2M1.");
}

float Float4E2M1NibbleToFloat(std::uint8_t nibble) noexcept {
  // Canonical FLOAT4E2M1 value table indexed by nibble (0x0..0xF).
  static const float kValues[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kValues[nibble & 0x0Fu];
}

std::uint8_t FloatRoundToFloat4E2M1Nibble(float v) noexcept {
  // FLOAT4E2M1 has no NaN or infinity encoding: NaN collapses to zero and
  // infinities saturate to the largest representable magnitude (±6).
  if (std::isnan(v))
    return 0x0u;
  // Saturate to representable range and handle ±0 sign preservation.
  if (v > 6.0f)
    return 0x7u;
  if (v < -6.0f)
    return 0xFu;
  if (v == 0.0f)
    return std::signbit(v) ? 0x8u : 0x0u;

  // Search the same-sign half, including the signed zero so values nearer to
  // zero than to ±0.5 (or exactly halfway, which rounds to even) collapse to
  // zero. Positive: nibbles 0x0..0x7  Negative: nibbles 0x8..0xF.
  static const float kPos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  static const std::uint8_t kPosN[8] = {0x0u, 0x1u, 0x2u, 0x3u, 0x4u, 0x5u, 0x6u, 0x7u};
  static const float kNeg[8] = {-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  static const std::uint8_t kNegN[8] = {0x8u, 0x9u, 0xAu, 0xBu, 0xCu, 0xDu, 0xEu, 0xFu};
  const bool positive = v > 0.0f;
  const float *vals = positive ? kPos : kNeg;
  const std::uint8_t *nibbles = positive ? kPosN : kNegN;

  // Linear scan for the nearest representable value with round-half-to-even
  // tie-breaking. In the FLOAT4E2M1 encoding the mantissa bit is the LSB of
  // the nibble (0 = even, 1 = odd), so we prefer the candidate whose LSB is 0
  // when two candidates are equidistant from ``v``.
  std::uint8_t best = nibbles[0];
  float best_dist = std::abs(v - vals[0]);
  for (int i = 1; i < 8; ++i) {
    const float d = std::abs(v - vals[i]);
    if (d < best_dist || (d == best_dist && (nibbles[i] & 1u) == 0u && (best & 1u) != 0u)) {
      best = nibbles[i];
      best_dist = d;
    }
  }
  return best;
}

std::vector<std::uint8_t> Pack4Bit(const std::vector<std::int8_t> &values) {
  std::vector<std::uint8_t> bytes((values.size() + 1) / 2, 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint8_t nibble = static_cast<std::uint8_t>(values[i]) & 0x0F;
    bytes[i / 2] |= static_cast<std::uint8_t>(nibble << (4 * (i % 2)));
  }
  return bytes;
}

std::vector<std::uint8_t> Pack2Bit(const std::vector<std::int8_t> &values) {
  std::vector<std::uint8_t> bytes((values.size() + 3) / 4, 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint8_t pair = static_cast<std::uint8_t>(values[i]) & 0x03;
    bytes[i / 4] |= static_cast<std::uint8_t>(pair << (2 * (i % 4)));
  }
  return bytes;
}

Tensor MakeFloat16Tensor(const std::string &name, const std::vector<int64_t> &shape,
                         const std::vector<float> &values) {
  std::vector<std::uint16_t> bits(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    bits[i] = FloatToFloat16Bits(values[i]);
  }
  Tensor t = Tensor::FromUint16(name, shape, bits);
  t.data_type = static_cast<std::int32_t>(DataType::FLOAT16);
  return t;
}

Tensor MakeFloat16Scalar(const std::string &name, float value) {
  Tensor t = Tensor::FromUint16(name, {}, {FloatToFloat16Bits(value)});
  t.data_type = static_cast<std::int32_t>(DataType::FLOAT16);
  return t;
}

Tensor MakeBfloat16Tensor(const std::string &name, const std::vector<int64_t> &shape,
                          const std::vector<float> &values) {
  std::vector<std::uint16_t> bits(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    bits[i] = FloatToBfloat16Bits(values[i]);
  }
  Tensor t = Tensor::FromUint16(name, shape, bits);
  t.data_type = static_cast<std::int32_t>(DataType::BFLOAT16);
  return t;
}

Tensor MakeBfloat16Scalar(const std::string &name, float value) {
  Tensor t = Tensor::FromUint16(name, {}, {FloatToBfloat16Bits(value)});
  t.data_type = static_cast<std::int32_t>(DataType::BFLOAT16);
  return t;
}

Tensor FloatToFloat16Tensor(const std::string &name, const Tensor &f) {
  EXT_ENFORCE_INVALID(f.data_type == DataType::FLOAT, "FloatToFloat16Tensor: input must be FLOAT.");
  const int64_t n = f.element_count();
  std::vector<std::uint16_t> bits(static_cast<std::size_t>(n));
  const float *src = f.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    bits[static_cast<std::size_t>(i)] = FloatToFloat16Bits(src[i]);
  }
  Tensor t = Tensor::FromUint16(name, f.shape, bits);
  t.data_type = static_cast<std::int32_t>(DataType::FLOAT16);
  return t;
}

Tensor RoundToFloat16(const Tensor &f) {
  EXT_ENFORCE_INVALID(f.data_type == DataType::FLOAT, "RoundToFloat16: input must be FLOAT.");
  const int64_t n = f.element_count();
  std::vector<float> rounded(static_cast<std::size_t>(n));
  const float *src = f.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    rounded[static_cast<std::size_t>(i)] = Float16BitsToFloat(FloatToFloat16Bits(src[i]));
  }
  return Tensor::FromFloat(f.name, f.shape, rounded);
}

Tensor Uint16ZeroPoint(std::uint16_t value) {
  std::vector<std::uint8_t> bytes(sizeof(std::uint16_t));
  std::memcpy(bytes.data(), &value, sizeof(std::uint16_t));
  return MakeScalarTensor(static_cast<std::int32_t>(DataType::UINT16), bytes);
}

Tensor Int16ZeroPoint(std::int16_t value) {
  std::vector<std::uint8_t> bytes(sizeof(std::int16_t));
  std::memcpy(bytes.data(), &value, sizeof(std::int16_t));
  return MakeScalarTensor(static_cast<std::int32_t>(DataType::INT16), bytes);
}

Tensor MakeFloat8Tensor(DataType dtype, const std::vector<int64_t> &shape,
                        const std::vector<float> &values, std::uint8_t (*encode)(float) noexcept) {
  std::vector<std::uint8_t> bytes(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    bytes[i] = encode(values[i]);
  }
  return Tensor("", static_cast<std::int32_t>(dtype), shape, std::move(bytes));
}

Tensor MakeSubByteTensor(DataType dtype, const std::vector<int64_t> &shape,
                         const std::vector<std::int8_t> &values, int bits) {
  std::vector<std::uint8_t> bytes = (bits == 4) ? Pack4Bit(values) : Pack2Bit(values);
  return Tensor("", static_cast<std::int32_t>(dtype), shape, std::move(bytes));
}

Tensor MakeFloat4E2M1Tensor(const std::vector<int64_t> &shape, const std::vector<float> &values) {
  std::vector<std::uint8_t> bytes((values.size() + 1) / 2, 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint8_t nibble = FloatToFloat4E2M1Nibble(values[i]);
    bytes[i / 2] |= static_cast<std::uint8_t>(nibble << (4 * (i % 2)));
  }
  return Tensor("", static_cast<std::int32_t>(DataType::FLOAT4E2M1), shape, std::move(bytes));
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
