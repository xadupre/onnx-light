// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::DataType;
using core::runtime::DefaultOpset;
using core::runtime::KernelContext;
using core::runtime::Tensor;
using onnx_kernels::kernel::ImageDecoder;

namespace Test {
namespace {

struct PngCase {
  const char *mode;
  std::vector<uint8_t> encoded;
  std::vector<uint8_t> rgb;
};

// 3x2 PNGs with Sub-filtered rows, including transparent palette/alpha pixels.
// Expected RGB values are independent of decoding and ignore alpha.
const std::vector<PngCase> &PixelFormatCases() {
  static const std::vector<uint8_t> gray = {0,   0,   0,   51,  51,  51,  102, 102, 102,
                                            153, 153, 153, 204, 204, 204, 255, 255, 255};
  static const std::vector<uint8_t> rgb = {11,  22,  33,  44,  55,  66,  77,  88,  99,
                                           110, 121, 132, 143, 154, 165, 176, 187, 198};
  static const std::vector<PngCase> cases = {
      {"1",
       {137, 80, 78, 71, 13, 10,  26,  10, 0,  0,  0,  13,  73, 72, 68,  82, 0,  0,
        0,   3,  0,  0,  0,  2,   1,   0,  0,  0,  0,  181, 15, 91, 183, 0,  0,  0,
        12,  73, 68, 65, 84, 120, 156, 99, 76, 96, 60, 0,   0,  1,  234, 1,  35, 161,
        28,  32, 75, 0,  0,  0,   0,   73, 69, 78, 68, 174, 66, 96, 130},
       {0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0}},
      {"L",
       {137, 80,  78,  71,  13,  10, 26,  10, 0,  0,   0,   13,  73,  72,  68, 82, 0,   0,  0,
        3,   0,   0,   0,   2,   8,  0,   0,  0,  0,   184, 31,  57,  198, 0,  0,  0,   16, 73,
        68,  65,  84,  120, 156, 99, 100, 48, 54, 102, 156, 105, 108, 12,  0,  4,  169, 1,  104,
        4,   197, 220, 177, 0,   0,  0,   0,  73, 69,  78,  68,  174, 66,  96, 130},
       gray},
      {"LA",
       {137, 80,  78,  71, 13,  10,  26, 10, 0,  0,   0,   13,  73,  72,  68,  82,  0,   0,  0,  3,
        0,   0,   0,   2,  8,   4,   0,  0,  0,  55,  125, 174, 145, 0,   0,   0,   22,  73, 68, 65,
        84,  120, 156, 99, 100, 96,  48, 22, 52, 206, 103, 156, 249, 223, 152, 209, 248, 63, 0,  19,
        6,   3,   231, 12, 30,  131, 8,  0,  0,  0,   0,   73,  69,  78,  68,  174, 66,  96, 130},
       gray},
      {"P",
       {137, 80,  78,  71,  13,  10,  26,  10,  0,   0,   0,   13,  73,  72,  68,  82, 0,
        0,   0,   3,   0,   0,   0,   2,   8,   3,   0,   0,   0,   170, 170, 150, 40, 0,
        0,   0,   18,  80,  76,  84,  69,  11,  22,  33,  44,  55,  66,  77,  88,  99, 110,
        121, 132, 143, 154, 165, 176, 187, 198, 121, 63,  70,  182, 0,   0,   0,   3,  116,
        82,  78,  83,  0,   17,  128, 68,  23,  100, 238, 0,   0,   0,   16,  73,  68, 65,
        84,  120, 156, 99,  100, 96,  100, 100, 100, 102, 100, 4,   0,   0,   43,  0,  10,
        224, 95,  228, 205, 0,   0,   0,   0,   73,  69,  78,  68,  174, 66,  96,  130},
       rgb},
      {"RGB",
       {137, 80, 78, 71,  13,  10,  26,  10, 0,  0,   0,  13,  73,  72,  68, 82,  0,  0,  0,
        3,   0,  0,  0,   2,   8,   2,   0,  0,  0,   18, 22,  241, 77,  0,  0,   0,  19, 73,
        68,  65, 84, 120, 156, 99,  228, 22, 83, 132, 0,  198, 188, 202, 22, 8,   11, 0,  29,
        40,  3,  60, 40,  134, 138, 100, 0,  0,  0,   0,  73,  69,  78,  68, 174, 66, 96, 130},
       rgb},
      {"RGBA",
       {137, 80,  78,  71,  13,  10,  26,  10, 0,   0,  0,   13,  73,  72,  68,  82, 0,   0,
        0,   3,   0,   0,   0,   2,   8,   6,  0,   0,  0,   157, 116, 102, 26,  0,  0,   0,
        30,  73,  68,  65,  84,  120, 156, 99, 228, 22, 83,  100, 80,  84,  84,  20, 4,   226,
        124, 198, 188, 202, 150, 255, 64,  6,  35,  16, 255, 7,   0,   56,  231, 5,  187, 2,
        221, 20,  78,  0,   0,   0,   0,   73, 69,  78, 68,  174, 66,  96,  130},
       rgb},
      {"L2",
       {137, 80, 78,  71, 13, 10,  26,  10, 0,   0,  0,   13,  73,  72, 68,  82, 0, 0,
        0,   3,  0,   0,  0,  2,   2,   0,  0,   0,  0,   242, 175, 33, 103, 0,  0, 0,
        12,  73, 68,  65, 84, 120, 156, 99, 148, 97, 124, 2,   0,   1,  66,  1,  3, 39,
        175, 59, 150, 0,  0,  0,   0,   73, 69,  78, 68,  174, 66,  96, 130},
       {0, 0, 0, 85, 85, 85, 255, 255, 255, 255, 255, 255, 170, 170, 170, 85, 85, 85}},
      {"L4",
       {137, 80, 78,  71,  13,  10,  26,  10, 0,   0,   0,   13,  73,  72,  68,  82, 0,  0,
        0,   3,  0,   0,   0,   2,   4,   0,  0,   0,   0,   125, 239, 212, 199, 0,  0,  0,
        14,  73, 68,  65,  84,  120, 156, 99, 100, 142, 101, 156, 19,  2,   0,   3,  30, 1,
        83,  5,  149, 164, 226, 0,   0,   0,  0,   73,  69,  78,  68,  174, 66,  96, 130},
       gray},
      {"P2",
       {137, 80, 78,  71, 13,  10,  26,  10, 0,   0,  0,   13,  73, 72,  68,  82,  0,   0,
        0,   3,  0,   0,  0,   2,   2,   3,  0,   0,  0,   224, 26, 142, 137, 0,   0,   0,
        9,   80, 76,  84, 69,  11,  22,  33, 44,  55, 66,  77,  88, 99,  94,  193, 181, 217,
        0,   0,  0,   3,  116, 82,  78,  83, 0,   17, 128, 68,  23, 100, 238, 0,   0,   0,
        12,  73, 68,  65, 84,  120, 156, 99, 148, 96, 156, 0,   0,  0,   226, 0,   171, 5,
        191, 28, 138, 0,  0,   0,   0,   73, 69,  78, 68,  174, 66, 96,  130},
       {11, 22, 33, 44, 55, 66, 77, 88, 99, 77, 88, 99, 44, 55, 66, 11, 22, 33}}};
  return cases;
}

} // namespace

TEST(KernelClass, ImageDecoderNormalizesPngSourceModes) {
  const ImageDecoder decoder{KernelContext{DefaultOpset(20)}};
  for (const auto &tc : PixelFormatCases()) {
    SCOPED_TRACE(tc.mode);
    Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(tc.encoded.size())}, tc.encoded);
    for (const std::string format : {"", "RGB", "BGR", "Grayscale"}) {
      SCOPED_TRACE(format);
      std::vector<uint8_t> expected;
      for (size_t i = 0; i < tc.rgb.size(); i += 3) {
        const uint8_t r = tc.rgb[i], g = tc.rgb[i + 1], b = tc.rgb[i + 2];
        if (format == "Grayscale") {
          expected.push_back(static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000));
        } else if (format == "BGR") {
          expected.insert(expected.end(), {b, g, r});
        } else {
          expected.insert(expected.end(), {r, g, b});
        }
      }
      const std::vector<int64_t> shape = {2, 3, format == "Grayscale" ? 1 : 3};
      const Tensor actual = format.empty() ? decoder(encoded) : decoder(encoded, format);
      EXPECT_EQ(actual.data_type, static_cast<int32_t>(DataType::UINT8));
      ASSERT_EQ(actual.shape, shape);
      EXPECT_EQ(actual.data, expected);
      EXPECT_EQ(encoded.data, tc.encoded);
      Tensor output = Tensor::FromUint8("", shape, std::vector<uint8_t>(expected.size()));
      decoder(encoded, format.empty() ? "RGB" : format, output);
      EXPECT_EQ(output.data, expected);
    }
  }
}

TEST(KernelClass, ImageDecoderRejectsMalformedPngSourceModes) {
  const ImageDecoder decoder{KernelContext{DefaultOpset(20)}};
  const auto palette = PixelFormatCases()[3].encoded;
  std::vector<std::vector<uint8_t>> invalid;
  auto missing_palette = palette;
  missing_palette.erase(missing_palette.begin() + 33, missing_palette.begin() + 63);
  invalid.push_back(missing_palette);
  auto invalid_palette_length = palette;
  invalid_palette_length[36] = 17;
  invalid.push_back(invalid_palette_length);
  auto invalid_index = palette;
  invalid_index.erase(invalid_index.begin() + 56, invalid_index.begin() + 59);
  invalid_index[36] = 15; // Last pixel still references removed palette entry 5.
  invalid.push_back(invalid_index);
  auto oversized_palette = PixelFormatCases().back().encoded;
  oversized_palette[24] = 1; // Three entries cannot fit in a one-bit palette.
  invalid.push_back(oversized_palette);
  auto excess_scanlines = palette;
  excess_scanlines[23] = 1;
  invalid.push_back(excess_scanlines);
  auto packed_alpha = PixelFormatCases()[2].encoded;
  packed_alpha[24] = 2; // PNG does not allow packed grayscale-alpha samples.
  invalid.push_back(packed_alpha);
  auto oversized_dimensions = PixelFormatCases()[5].encoded;
  for (size_t i = 16; i < 24; ++i)
    oversized_dimensions[i] = (i == 16 || i == 20) ? 0x7f : 0xff;
  invalid.push_back(oversized_dimensions);
  for (const auto &bytes : invalid) {
    Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bytes.size())}, bytes);
    for (const std::string format : {"RGB", "BGR", "Grayscale"}) {
      const Tensor actual = decoder(encoded, format);
      EXPECT_EQ(actual.shape, (std::vector<int64_t>{0, 0, format == "Grayscale" ? 1 : 3}));
      EXPECT_TRUE(actual.data.empty());
    }
  }
}

TEST(KernelClass, ImageDecoderNormalizesPnmSourceModes) {
  const ImageDecoder decoder{KernelContext{DefaultOpset(20)}};
  struct PnmCase {
    std::string header;
    std::vector<uint8_t> samples;
    std::vector<uint8_t> gray;
  };
  const std::vector<uint8_t> gray = {0, 51, 102, 153, 204, 255};
  const std::vector<PnmCase> cases = {
      {"P1\n3 2\n0 1 0\n1 0 1\n", {}, {255, 0, 255, 0, 255, 0}},
      {"P2\n3 2\n255\n0 51 102\n153 204 255\n", {}, gray},
      {"P3\n3 2\n255\n0 0 0 51 51 51 102 102 102\n153 153 153 204 204 204 255 255 255\n", {}, gray},
      {"P4\n3 2\n", {0x40, 0xa0}, {255, 0, 255, 0, 255, 0}},
      {"P5\n3 2\n255\n", gray, gray},
      {"P6\n3 2\n255\n",
       {0, 0, 0, 51, 51, 51, 102, 102, 102, 153, 153, 153, 204, 204, 204, 255, 255, 255},
       gray}};
  for (const auto &tc : cases) {
    SCOPED_TRACE(tc.header.substr(0, 2));
    std::vector<uint8_t> bytes(tc.header.begin(), tc.header.end());
    bytes.insert(bytes.end(), tc.samples.begin(), tc.samples.end());
    Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bytes.size())}, bytes);
    for (const std::string format : {"", "RGB", "BGR", "Grayscale"}) {
      SCOPED_TRACE(format);
      const int64_t channels = format == "Grayscale" ? 1 : 3;
      std::vector<uint8_t> expected;
      for (uint8_t value : tc.gray)
        expected.insert(expected.end(), static_cast<size_t>(channels), value);
      const Tensor actual = format.empty() ? decoder(encoded) : decoder(encoded, format);
      EXPECT_EQ(actual.data_type, static_cast<int32_t>(DataType::UINT8));
      EXPECT_EQ(actual.shape, (std::vector<int64_t>{2, 3, channels}));
      EXPECT_EQ(actual.data, expected);
      EXPECT_EQ(encoded.data, bytes);
    }
  }
}

} // namespace Test
