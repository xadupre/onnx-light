// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

size_t Product(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(b == 0 || a <= std::numeric_limits<size_t>::max() / b,
                      "compressed-tensors: size overflow.");
  return a * b;
}

size_t PackedBytes(size_t count, uint32_t bits, size_t unit) {
  EXT_ENFORCE_INVALID(bits >= 1 && bits <= 8, "compressed-tensors: invalid bit width.");
  const size_t total = Product(count, bits);
  return Product(total / unit + (total % unit != 0), unit / 8);
}

int Code(uint8_t code) { return code < 128 ? code : static_cast<int>(code) - 256; }

void CheckCode(int code, uint32_t bits) {
  const int half = 1 << (bits - 1);
  EXT_ENFORCE_INVALID(code >= -half && code < half, "compressed-tensors: code out of range.");
}

uint32_t ReadWord(std::string_view packed, size_t index) {
  const auto *p = reinterpret_cast<const uint8_t *>(packed.data()) + index * 4;
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

void WriteWord(std::string &packed, size_t index, uint32_t value) {
  for (size_t i = 0; i != 4; ++i)
    packed[index * 4 + i] = static_cast<char>(value >> (8 * i));
}

} // namespace

std::string PackQuantizationCodes(std::string_view codes, uint32_t bits) {
  std::string packed(PackedBytes(codes.size(), bits, 8), '\0');
  for (size_t i = 0; i < codes.size(); ++i) {
    const int code = Code(static_cast<uint8_t>(codes[i]));
    CheckCode(code, bits);
    const size_t bit = i * bits;
    const uint32_t value = (static_cast<uint32_t>(code) & ((1u << bits) - 1)) << (bit % 8);
    packed[bit / 8] |= static_cast<char>(value);
    if (bit % 8 + bits > 8)
      packed[bit / 8 + 1] |= static_cast<char>(value >> 8);
  }
  return packed;
}

std::string UnpackQuantizationCodes(std::string_view packed, size_t count, uint32_t bits) {
  EXT_ENFORCE_INVALID(packed.size() == PackedBytes(count, bits, 8),
                      "compressed-tensors: invalid portable payload extent.");
  const size_t remainder = count * bits % 8;
  EXT_ENFORCE_INVALID(!remainder || (static_cast<uint8_t>(packed.back()) >> remainder) == 0,
                      "compressed-tensors: nonzero portable padding bits.");
  std::string codes(count, '\0');
  for (size_t i = 0; i < count; ++i) {
    const size_t bit = i * bits;
    uint32_t code = static_cast<uint8_t>(packed[bit / 8]) >> (bit % 8);
    if (bit % 8 + bits > 8)
      code |= uint32_t(static_cast<uint8_t>(packed[bit / 8 + 1])) << (8 - bit % 8);
    code &= (1u << bits) - 1;
    codes[i] = static_cast<char>(code < (1u << (bits - 1)) ? code : code - (1u << bits));
  }
  return codes;
}

std::string PackCompressedTensorInt32(std::string_view codes, size_t rows, size_t columns,
                                      uint32_t bits, uint32_t axis) {
  EXT_ENFORCE_INVALID(axis <= 1 && rows > 0 && columns > 0,
                      "compressed-tensors: invalid matrix or packing axis.");
  EXT_ENFORCE_INVALID(codes.size() == Product(rows, columns),
                      "compressed-tensors: invalid code extent.");
  const size_t count = axis == 0 ? rows : columns;
  const size_t lanes = axis == 0 ? columns : rows;
  const size_t words = PackedBytes(count, bits, 32) / 4;
  std::string packed(Product(Product(lanes, words), 4), '\0');
  for (size_t lane = 0; lane < lanes; ++lane) {
    for (size_t i = 0; i < count; ++i) {
      const size_t input = axis == 0 ? i * columns + lane : lane * columns + i;
      const int code = Code(static_cast<uint8_t>(codes[input]));
      CheckCode(code, bits);
      const uint32_t value = code + (1 << (bits - 1));
      const size_t bit = i * bits, word = bit / 32, shift = bit % 32;
      const size_t output = axis == 0 ? word * lanes + lane : lane * words + word;
      WriteWord(packed, output, ReadWord(packed, output) | (value << shift));
      if (shift + bits > 32) {
        const size_t next = axis == 0 ? (word + 1) * lanes + lane : output + 1;
        WriteWord(packed, next, ReadWord(packed, next) | (value >> (32 - shift)));
      }
    }
  }
  return packed;
}

std::string UnpackCompressedTensorInt32(std::string_view packed, size_t rows, size_t columns,
                                        uint32_t bits, uint32_t axis) {
  EXT_ENFORCE_INVALID(axis <= 1 && rows > 0 && columns > 0,
                      "compressed-tensors: invalid matrix or packing axis.");
  const size_t count = axis == 0 ? rows : columns;
  const size_t lanes = axis == 0 ? columns : rows;
  const size_t words = PackedBytes(count, bits, 32) / 4;
  EXT_ENFORCE_INVALID(packed.size() == Product(Product(lanes, words), 4),
                      "compressed-tensors: invalid packed extent.");
  std::string codes(Product(rows, columns), '\0');
  for (size_t lane = 0; lane < lanes; ++lane) {
    const size_t remainder = count * bits % 32;
    const size_t last = axis == 0 ? (words - 1) * lanes + lane : lane * words + words - 1;
    EXT_ENFORCE_INVALID(!remainder || (ReadWord(packed, last) >> remainder) == 0,
                        "compressed-tensors: nonzero int32 padding bits.");
    for (size_t i = 0; i < count; ++i) {
      const size_t bit = i * bits, word = bit / 32, shift = bit % 32;
      const size_t input = axis == 0 ? word * lanes + lane : lane * words + word;
      uint32_t code = ReadWord(packed, input) >> shift;
      if (shift + bits > 32) {
        const size_t next = axis == 0 ? (word + 1) * lanes + lane : input + 1;
        code |= ReadWord(packed, next) << (32 - shift);
      }
      code &= (1u << bits) - 1;
      const size_t output = axis == 0 ? i * columns + lane : lane * columns + i;
      codes[output] = static_cast<char>(static_cast<int>(code) - (1 << (bits - 1)));
    }
  }
  return codes;
}

void ValidateCompressedTensorAffine(std::string_view codes, float scale, int32_t zero_point) {
  EXT_ENFORCE_INVALID(!codes.empty() && std::isfinite(scale) && scale > 0 && zero_point >= -128 &&
                          zero_point <= 127,
                      "compressed-tensors: invalid affine block.");
  int distance = 0;
  for (char code : codes)
    distance = std::max(distance, std::abs(Code(static_cast<uint8_t>(code)) - zero_point));
  volatile float bound = static_cast<float>(distance) * scale;
  EXT_ENFORCE_INVALID(std::isfinite(bound),
                      "compressed-tensors: affine codes/scales overflow FLOAT32 reconstruction.");
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
