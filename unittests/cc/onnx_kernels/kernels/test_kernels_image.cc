// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/image/include_image_kernels.h"
#include "onnx_kernels/kernels/kernel_context.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::DataType;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::ImageDecoder;
using onnx_kernels::kernel::KernelContext;

namespace Test {

TEST(KernelClass, ImageDecoderChannelCountMapsPixelFormat) {
  EXPECT_EQ(ImageDecoder::ChannelCount("RGB"), 3);
  EXPECT_EQ(ImageDecoder::ChannelCount("BGR"), 3);
  EXPECT_EQ(ImageDecoder::ChannelCount("Grayscale"), 1);
  EXPECT_THROW(ImageDecoder::ChannelCount("rgb"), std::invalid_argument);
  EXPECT_THROW(ImageDecoder::ChannelCount(""), std::invalid_argument);
  EXPECT_THROW(ImageDecoder::ChannelCount("RGBA"), std::invalid_argument);
}

TEST(KernelClass, ImageDecoderReturnsEmptyMatrixForRGB) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};

  Tensor encoded = Tensor::FromUint8("", {4}, std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03});
  Tensor out = decoder(encoded, "RGB");

  EXPECT_EQ(out.data_type, static_cast<int32_t>(DataType::UINT8));
  const std::vector<int64_t> expected_shape = {0, 0, 3};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_EQ(out.element_count(), 0);
  EXPECT_TRUE(out.data.empty());
}

TEST(KernelClass, ImageDecoderReturnsEmptyMatrixForGrayscale) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};

  Tensor encoded = Tensor::FromUint8("", {3}, std::vector<uint8_t>{0xff, 0xd8, 0xff});
  Tensor out = decoder(encoded, "Grayscale");

  const std::vector<int64_t> expected_shape = {0, 0, 1};
  EXPECT_EQ(out.shape, expected_shape);
  EXPECT_TRUE(out.data.empty());
}

TEST(KernelClass, ImageDecoderDefaultsToRGBPixelFormat) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {0}, std::vector<uint8_t>{});
  Tensor out = decoder(encoded);
  const std::vector<int64_t> expected_shape = {0, 0, 3};
  EXPECT_EQ(out.shape, expected_shape);
}

TEST(KernelClass, ImageDecoderDecodesBmpRgb) {
  // Minimal 2x2 24-bit uncompressed (BI_RGB) BMP with pixels:
  //   Display row 0: (90,80,70) (120,110,100)
  //   Display row 1: (30,20,10) (60,50,40)
  // Pixels are stored BGR bottom-to-top, so file row 0 = display row 1.
  // clang-format off
  const std::vector<uint8_t> bmp_bytes = {
    // ---- File header (14 bytes) ----
    0x42, 0x4D,              // "BM"
    0x42, 0x00, 0x00, 0x00,  // file size = 66
    0x00, 0x00, 0x00, 0x00,  // reserved
    0x36, 0x00, 0x00, 0x00,  // pixel data offset = 54
    // ---- BITMAPINFOHEADER (40 bytes) ----
    0x28, 0x00, 0x00, 0x00,  // header size = 40
    0x02, 0x00, 0x00, 0x00,  // width = 2
    0x02, 0x00, 0x00, 0x00,  // height = 2 (positive => bottom-up)
    0x01, 0x00,              // planes = 1
    0x18, 0x00,              // bpp = 24
    0x00, 0x00, 0x00, 0x00,  // compression = BI_RGB
    0x0C, 0x00, 0x00, 0x00,  // image size = 12 bytes
    0x00, 0x00, 0x00, 0x00,  // X pixels/meter (ignored)
    0x00, 0x00, 0x00, 0x00,  // Y pixels/meter (ignored)
    0x00, 0x00, 0x00, 0x00,  // color table entries
    0x00, 0x00, 0x00, 0x00,  // important colors
    // ---- Pixel data (bottom-up: file row 0 = display row 1) ----
    // Row stride for width=2, bpp=24: (6+3)&~3 = 8 bytes per row (2 padding bytes).
    10,  20,  30,  40,  50,  60,  0, 0,   // display row 1: (B=10,G=20,R=30), (B=40,G=50,R=60)
    70,  80,  90,  100, 110, 120, 0, 0,   // display row 0: (B=70,G=80,R=90), (B=100,G=110,R=120)
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bmp_bytes.size())}, bmp_bytes);

  Tensor out = decoder(encoded, "RGB");

  EXPECT_EQ(out.data_type, static_cast<int32_t>(DataType::UINT8));
  const std::vector<int64_t> expected_shape = {2, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.element_count(), 12);
  // Display row 0 (top row, file row 1): BGR=(70,80,90),(100,110,120) →
  // RGB=(90,80,70),(120,110,100) Display row 1 (bottom row, file row 0): BGR=(10,20,30),(40,50,60)
  // → RGB=(30,20,10),(60,50,40)
  const std::vector<uint8_t> expected_pixels = {90, 80, 70, 120, 110, 100, 30, 20, 10, 60, 50, 40};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderDecodesBmpBgr) {
  // Same 2x2 BMP as above but pixel_format="BGR" keeps BGR channel order.
  // clang-format off
  const std::vector<uint8_t> bmp_bytes = {
    0x42, 0x4D, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x36, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x02, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    10, 20, 30, 40, 50, 60, 0, 0,    // file row 0 = display row 1 (+ 2-byte padding)
    70, 80, 90, 100, 110, 120, 0, 0, // file row 1 = display row 0 (+ 2-byte padding)
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bmp_bytes.size())}, bmp_bytes);

  Tensor out = decoder(encoded, "BGR");

  const std::vector<int64_t> expected_shape = {2, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  // BGR pixel_format keeps the native BMP channel order.
  // Display row 0: BGR=(70,80,90),(100,110,120)
  // Display row 1: BGR=(10,20,30),(40,50,60)
  const std::vector<uint8_t> expected_pixels = {70, 80, 90, 100, 110, 120, 10, 20, 30, 40, 50, 60};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderDecodesBmpGrayscale) {
  // Same 2x2 BMP decoded to Grayscale using the ITU-R BT.601 luminance formula.
  // clang-format off
  const std::vector<uint8_t> bmp_bytes = {
    0x42, 0x4D, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x36, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x02, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    // pixel: B=255, G=0, R=0 (blue) and B=0, G=255, R=0 (green) — row 0 = display row 1
    255, 0, 0,   0, 255, 0,  0, 0,  // + 2-byte padding
    // pixel: B=0, G=0, R=255 (red) and B=0, G=0, R=0 (black) — row 1 = display row 0
    0, 0, 255,   0, 0, 0,    0, 0,  // + 2-byte padding
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bmp_bytes.size())}, bmp_bytes);

  Tensor out = decoder(encoded, "Grayscale");

  const std::vector<int64_t> expected_shape = {2, 2, 1};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.element_count(), 4);
  // Display row 0, col 0: B=0,G=0,R=255 → grey=(299*255+0+0+500)/1000=76
  // Display row 0, col 1: B=0,G=0,R=0   → grey=0
  // Display row 1, col 0: B=255,G=0,R=0 → grey=(0+0+114*255+500)/1000=29
  // Display row 1, col 1: B=0,G=255,R=0 → grey=(0+587*255+0+500)/1000=150
  const uint8_t r255 = static_cast<uint8_t>((299 * 255 + 500) / 1000);
  const uint8_t b255 = static_cast<uint8_t>((114 * 255 + 500) / 1000);
  const uint8_t g255 = static_cast<uint8_t>((587 * 255 + 500) / 1000);
  const std::vector<uint8_t> expected_pixels = {r255, 0, b255, g255};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderRejectsNonUint8Input) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromInt32("", {4}, {0, 1, 2, 3});
  EXPECT_THROW(decoder(encoded, "RGB"), std::exception);
}

TEST(KernelClass, ImageDecoderRejectsNonOneDimensionalInput) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {2, 2}, std::vector<uint8_t>{0, 0, 0, 0});
  EXPECT_THROW(decoder(encoded, "RGB"), std::exception);
}

TEST(KernelClass, ImageDecoderRejectsUnknownPixelFormat) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {1}, std::vector<uint8_t>{0});
  EXPECT_THROW(decoder(encoded, "rgba"), std::invalid_argument);
}

TEST(KernelClass, ImageDecoderInPlaceOverloadValidatesOutput) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {2}, std::vector<uint8_t>{0xaa, 0xbb});

  // Empty-matrix output with matching channel count succeeds.
  Tensor out_ok = Tensor::FromUint8("", {0, 0, 3}, {});
  EXPECT_NO_THROW(decoder(encoded, "RGB", out_ok));

  // Channel count mismatch with ``pixel_format`` is rejected.
  Tensor out_bad_channels = Tensor::FromUint8("", {0, 0, 1}, {});
  EXPECT_THROW(decoder(encoded, "RGB", out_bad_channels), std::exception);

  // Non-3-D output is rejected.
  Tensor out_bad_rank = Tensor::FromUint8("", {0}, {});
  EXPECT_THROW(decoder(encoded, "RGB", out_bad_rank), std::exception);
}

} // namespace Test
