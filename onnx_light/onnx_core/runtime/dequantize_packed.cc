// Copyright (c) ONNX Project Contributors
// Copyright (c) 2023-2026 The ggml authors
//
// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// GGML decoding follows da8e2a54b919557b816b1c2beb432ccb766f7ce0.
// The full GGML MIT license is retained in NOTICE.

#include "onnx_core/runtime/dequantize_packed.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

#include "onnx_core/runtime/dequantize_packed_codebooks.inc"

static_assert(sizeof(k_iq2xxs_grid) == 1025 && sizeof(k_iq2xs_grid) == 2049 &&
              sizeof(k_iq2s_grid) == 4097 && sizeof(k_iq3xxs_grid) == 1025 &&
              sizeof(k_iq3s_grid) == 2049 && sizeof(k_iq1s_grid) == 8193);

constexpr PackedQuantizationProfile kProfiles[] = {
    {"gguf:q4_0", 32, 18, 2},      {"gguf:q4_1", 32, 20, 3},      {"gguf:q5_0", 32, 22, 6},
    {"gguf:q5_1", 32, 24, 7},      {"gguf:q8_0", 32, 34, 8},      {"gguf:q8_1", 32, 36, 9},
    {"gguf:q2_k", 256, 84, 10},    {"gguf:q3_k", 256, 110, 11},   {"gguf:q4_k", 256, 144, 12},
    {"gguf:q5_k", 256, 176, 13},   {"gguf:q6_k", 256, 210, 14},   {"gguf:q8_k", 256, 292, 15},
    {"gguf:iq2_xxs", 256, 66, 16}, {"gguf:iq2_xs", 256, 74, 17},  {"gguf:iq3_xxs", 256, 98, 18},
    {"gguf:iq1_s", 256, 50, 19},   {"gguf:iq4_nl", 32, 18, 20},   {"gguf:iq3_s", 256, 110, 21},
    {"gguf:iq2_s", 256, 82, 22},   {"gguf:iq4_xs", 256, 136, 23}, {"gguf:iq1_m", 256, 56, 29},
    {"gguf:tq1_0", 256, 54, 34},   {"gguf:tq2_0", 256, 66, 35},   {"gguf:mxfp4", 32, 17, 39},
    {"gguf:nvfp4", 64, 36, 40},    {"gguf:q1_0", 128, 18, 41},    {"gguf:q2_0", 64, 18, 42}};

constexpr int kIq4[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int kFp4[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Volatile stores enforce each reference FP32 rounding step, including on FMA-capable targets.
float Mul(float a, float b) {
  volatile float result = a * b;
  return result;
}
float Add(float a, float b) {
  volatile float result = a + b;
  return result;
}
float Sub(float a, float b) {
  volatile float result = a - b;
  return result;
}
uint32_t Uint(const uint8_t *p, size_t count = 2) {
  uint32_t result = 0;
  for (size_t j = 0; j < count; ++j)
    result |= static_cast<uint32_t>(p[j]) << (8 * j);
  return result;
}
float Half(const uint8_t *p) { return Float16BitsToFloat(static_cast<uint16_t>(Uint(p))); }
int SignedByte(uint8_t v) { return v < 128 ? v : static_cast<int>(v) - 256; }

// Extracts grouped bit planes, rather than contiguous bits for adjacent values.
int Unpack(const uint8_t *p, size_t i, size_t bits, size_t group) {
  const size_t planes = 8 / bits;
  return (p[(i / (planes * group)) * group + i % group] >> (bits * ((i / group) % planes))) &
         ((1 << bits) - 1);
}
int Hex(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }
int Grid(const char *text, const int *values, size_t width, size_t index, size_t lane) {
  const size_t element = index * width + lane;
  const size_t bits = width == 4 ? 4 : 2;
  const size_t per_byte = 8 / bits;
  const size_t byte = element / per_byte;
  const int packed = (Hex(text[2 * byte]) << 4) | Hex(text[2 * byte + 1]);
  return values[(packed >> (bits * (element % per_byte))) & ((1 << bits) - 1)];
}
int Sign(uint32_t index, size_t lane, bool parity) {
  if (parity)
    index |= (std::popcount(index) & 1U) << 7;
  return 1 - 2 * static_cast<int>((index >> lane) & 1U);
}

void DecodeIq(const uint8_t *p, int type, float *out) {
  float d = Half(p);
  uint32_t scales[4]{};
  if (type == 29) {
    uint16_t bits = 0;
    for (size_t j = 0; j < 4; ++j) {
      scales[j] = Uint(p + 48 + 2 * j);
      bits |= (scales[j] >> 12) << (4 * j);
    }
    d = Float16BitsToFloat(bits);
  }
  for (size_t group = 0; group < 8; ++group) {
    const uint32_t aux = type == 16   ? Uint(p + 6 + 8 * group, 4)
                         : type == 18 ? Uint(p + 66 + 4 * group, 4)
                                      : 0;
    for (size_t lane = 0; lane < 4; ++lane) {
      const size_t i = 4 * group + lane;
      size_t index = 0;
      uint32_t signs = 0;
      bool parity = false;
      float scale = 0;
      const char *grid = nullptr;
      const int *values = nullptr;
      if (type == 16 || type == 17 || type == 22) {
        if (type == 16) {
          index = p[2 + 8 * group + lane];
          signs = (aux >> (7 * lane)) & 127;
          parity = true;
          scale = Mul(Mul(d, 0.5f + static_cast<float>(aux >> 28)), 0.25f);
          grid = k_iq2xxs_grid;
          values = k_iq2xxs_grid_values;
        } else if (type == 17) {
          const uint32_t word = Uint(p + 2 + 2 * i);
          index = word & 511;
          signs = word >> 9;
          parity = true;
          scale = Mul(Mul(d, 0.5f + ((p[66 + group] >> (4 * (lane / 2))) & 15)), 0.25f);
          grid = k_iq2xs_grid;
          values = k_iq2xs_grid_values;
        } else {
          index = p[2 + i] | (((p[66 + group] >> (2 * lane)) & 3) << 8);
          signs = p[34 + i];
          scale = Mul(Mul(d, 0.5f + ((p[74 + group] >> (4 * (lane / 2))) & 15)), 0.25f);
          grid = k_iq2s_grid;
          values = k_iq2s_grid_values;
        }
        for (size_t j = 0; j < 8; ++j)
          out[8 * i + j] = Mul(Mul(scale, Grid(grid, values, 8, index, j)), Sign(signs, j, parity));
      } else if (type == 18 || type == 21) {
        if (type == 18) {
          signs = (aux >> (7 * lane)) & 127;
          parity = true;
          scale = Mul(Mul(d, 0.5f + static_cast<float>(aux >> 28)), 0.5f);
          grid = k_iq3xxs_grid;
          values = k_iq3xxs_grid_values;
        } else {
          signs = p[74 + i];
          scale = Mul(d, 1 + 2 * ((p[106 + group / 2] >> (4 * (group % 2))) & 15));
          grid = k_iq3s_grid;
          values = k_iq3s_grid_values;
        }
        for (size_t j = 0; j < 8; ++j) {
          index = p[2 + 2 * i + j / 4];
          if (type == 21)
            index |= ((p[66 + group] >> (2 * lane + j / 4)) & 1) << 8;
          out[8 * i + j] =
              Mul(Mul(scale, Grid(grid, values, 4, index, j % 4)), Sign(signs, j, parity));
        }
      } else {
        float delta;
        if (type == 19) {
          const uint32_t high = Uint(p + 34 + 2 * group);
          index = p[2 + i] | (((high >> (3 * lane)) & 7) << 8);
          scale = Mul(d, 2 * ((high >> 12) & 7) + 1);
          delta = high & 0x8000 ? -0.125f : 0.125f;
        } else {
          const uint32_t high = p[32 + i / 2] >> (4 * (i % 2));
          index = p[i] | ((high & 7) << 8);
          scale = Mul(d, 2 * ((scales[group / 2] >> (6 * (group % 2) + 3 * (lane / 2))) & 7) + 1);
          delta = high & 8 ? -0.125f : 0.125f;
        }
        for (size_t j = 0; j < 8; ++j)
          out[8 * i + j] =
              Mul(scale, Add(Grid(k_iq1s_grid, k_iq1s_grid_values, 8, index, j), delta));
      }
    }
  }
}

void Decode(const uint8_t *p, const PackedQuantizationProfile &profile, float *out) {
  const int type = profile.ggml_type;
  if (type == 16 || type == 17 || type == 18 || type == 19 || type == 21 || type == 22 ||
      type == 29) {
    DecodeIq(p, type, out);
    return;
  }
  const float d = type == 15                 ? std::bit_cast<float>(Uint(p, 4))
                  : type == 10               ? Half(p + 80)
                  : type == 11               ? Half(p + 108)
                  : type == 14               ? Half(p + 208)
                  : type == 34               ? Half(p + 52)
                  : type == 35               ? Half(p + 64)
                  : type == 39 || type == 40 ? 0
                                             : Half(p);
  for (size_t i = 0; i < profile.block_size; ++i) {
    float value = 0;
    if (type == 2 || type == 3 || type == 6 || type == 7 || type == 20) {
      const bool affine = type == 3 || type == 7;
      const size_t offset = affine ? 4 : 2;
      const bool five = type == 6 || type == 7;
      int q = Unpack(p + offset + (five ? 4 : 0), i, 4, 16);
      if (five)
        q |= ((Uint(p + offset, 4) >> i) & 1) << 4;
      if (type == 20)
        value = Mul(d, kIq4[q]);
      else {
        if (!affine)
          q -= five ? 16 : 8;
        value = Mul(d, q);
        if (affine)
          value = Add(value, Half(p + 2));
      }
    } else if (type == 8 || type == 9 || type == 15) {
      value = Mul(d, SignedByte(p[(type == 8 ? 2 : 4) + i]));
    } else if (type == 10 || type == 11) {
      int q = Unpack(p + (type == 10 ? 16 : 32), i, 2, 32);
      const size_t g = i / 16;
      if (type == 10) {
        value = Sub(Mul(Mul(d, p[g] & 15), q), Mul(Half(p + 82), p[g] >> 4));
      } else {
        const int scale = (Unpack(p + 96, g, 4, 8) | (Unpack(p + 104, g, 2, 4) << 4)) - 32;
        q -= (1 - Unpack(p, i, 1, 32)) * 4;
        value = Mul(Mul(d, scale), q);
      }
    } else if (type == 12 || type == 13) {
      const size_t g = i / 32;
      const uint8_t *sc = p + 4;
      const int scale = g < 4 ? sc[g] & 63 : (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4);
      const int minimum = g < 4 ? sc[g + 4] & 63 : (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
      int q = Unpack(p + (type == 12 ? 16 : 48), i, 4, 32);
      if (type == 13)
        q |= Unpack(p + 16, i, 1, 32) << 4;
      value = Sub(Mul(Mul(d, scale), q), Mul(Half(p + 2), minimum));
    } else if (type == 14) {
      const int q = (Unpack(p, i, 4, 64) | (Unpack(p + 128, i, 2, 32) << 4)) - 32;
      value = Mul(Mul(d, SignedByte(p[192 + i / 16])), q);
    } else if (type == 23) {
      const size_t g = i / 32;
      const int scale =
          (((p[4 + g / 2] >> (4 * (g % 2))) & 15) | (((Uint(p + 2) >> (2 * g)) & 3) << 4)) - 32;
      value = Mul(Mul(d, scale), kIq4[Unpack(p + 8, i, 4, 16)]);
    } else if (type == 34) {
      const size_t start = i < 160 ? 0 : i < 240 ? 32 : 48;
      const size_t width = i < 160 ? 32 : i < 240 ? 16 : 4;
      const size_t offset = i < 160 ? i : i < 240 ? i - 160 : i - 240;
      constexpr int powers[] = {1, 3, 9, 27, 81};
      const int q = ((p[start + offset % width] * powers[offset / width]) & 255) * 3 / 256 - 1;
      value = Mul(d, q);
    } else if (type == 35) {
      value = Mul(d, Unpack(p, i, 2, 32) - 1);
    } else if (type == 39) {
      const uint32_t e = p[0];
      const uint32_t bits = e <= 1 ? 0x00200000U << e : (e - 1) << 23;
      value = Mul(std::bit_cast<float>(bits), kFp4[Unpack(p + 1, i, 4, 16)]);
    } else if (type == 40) {
      const size_t g = i / 16;
      const int e = p[g], exponent = (e >> 3) & 15, mantissa = e & 7;
      const float scale = e == 0 || e == 127 ? 0
                          : exponent == 0    ? std::ldexp(static_cast<float>(mantissa), -10)
                                             : std::ldexp(1.0f + mantissa / 8.0f, exponent - 8);
      value = Mul(scale, kFp4[Unpack(p + 4 + 8 * g, i % 16, 4, 8)]);
    } else if (type == 41) {
      value = (p[2 + i / 8] >> (i % 8)) & 1 ? d : -d;
    } else if (type == 42) {
      value = Mul(d, Unpack(p + 2, i, 2, 1) - 1);
    }
    out[i] = value;
  }
}

size_t CheckedMultiply(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) / b,
                      "DequantizePacked: size exceeds int64 range.");
  return a * b;
}

} // namespace

std::span<const PackedQuantizationProfile> PackedQuantizationProfiles() { return kProfiles; }

const PackedQuantizationProfile &PackedQuantizationLayout(std::string_view format) {
  for (const auto &profile : kProfiles)
    if (profile.format == format)
      return profile;
  EXT_THROW_INVALID("DequantizePacked: unsupported format '", std::string(format), "'.");
}

size_t ValidatePackedQuantizationShape(const PackedQuantizationProfile &profile, const Shape &shape,
                                       int64_t output_dtype) {
  EXT_ENFORCE_INVALID(output_dtype == TensorProto::FLOAT || output_dtype == TensorProto::FLOAT16 ||
                          output_dtype == TensorProto::BFLOAT16 ||
                          output_dtype == TensorProto::DOUBLE,
                      "DequantizePacked: output_dtype must be FLOAT, FLOAT16, BFLOAT16 or DOUBLE.");
  EXT_ENFORCE_INVALID(!shape.empty(), "DequantizePacked: shape must have rank at least one.");
  EXT_ENFORCE_INVALID(profile.block_size != 0 && profile.block_bytes != 0,
                      "DequantizePacked: block dimensions must be positive.");
  bool empty = false;
  for (const int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "DequantizePacked: dimensions must be nonnegative.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max(),
                        "DequantizePacked: dimension exceeds size_t.");
    empty |= dim == 0;
  }
  EXT_ENFORCE_INVALID(shape[shape.size() - 1] % profile.block_size == 0,
                      "DequantizePacked: last dimension must contain complete blocks.");
  if (empty)
    return 0;
  size_t count = 1;
  for (const int64_t dim : shape)
    count = CheckedMultiply(count, static_cast<size_t>(dim));
  const size_t width = output_dtype == TensorProto::DOUBLE  ? 8
                       : output_dtype == TensorProto::FLOAT ? 4
                                                            : 2;
  CheckedMultiply(count, width);
  return CheckedMultiply(count / profile.block_size, profile.block_bytes);
}

Tensor DequantizePackedTensor(const Tensor &input, std::string_view format, const Shape &shape,
                              int64_t output_dtype, RawBufferAllocator *allocator) {
  const auto &profile = PackedQuantizationLayout(format);
  const size_t expected = ValidatePackedQuantizationShape(profile, shape, output_dtype);
  EXT_ENFORCE_INVALID(input.data_type == TensorProto::UINT8 && input.shape.size() == 1,
                      "DequantizePacked: x must be a one-dimensional UINT8 tensor.");
  EXT_ENFORCE_INVALID(input.shape[0] >= 0 && static_cast<uint64_t>(input.shape[0]) == expected &&
                          input.size_bytes() == expected,
                      "DequantizePacked: packed input length must match exactly.");
  const size_t count = expected / profile.block_bytes * profile.block_size;
  const size_t width = output_dtype == TensorProto::DOUBLE  ? 8
                       : output_dtype == TensorProto::FLOAT ? 4
                                                            : 2;
  Tensor output =
      MakeOutputTensor(static_cast<int32_t>(output_dtype), shape, count * width, allocator);
  std::array<float, 256> block{};
  for (size_t b = 0; b < count / profile.block_size; ++b) {
    Decode(input.bytes() + b * profile.block_bytes, profile, block.data());
    for (size_t j = 0; j < profile.block_size; ++j) {
      const size_t i = b * profile.block_size + j;
      if (output_dtype == TensorProto::FLOAT)
        output.AsFloat()[i] = block[j];
      else if (output_dtype == TensorProto::DOUBLE)
        output.As<double>()[i] = static_cast<double>(block[j]);
      else {
        const uint16_t bits = output_dtype == TensorProto::FLOAT16 ? FloatToFloat16Bits(block[j])
                                                                   : FloatToBfloat16Bits(block[j]);
        std::memcpy(output.mutable_bytes() + 2 * i, &bits, sizeof(bits));
      }
    }
  }
  return output;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
