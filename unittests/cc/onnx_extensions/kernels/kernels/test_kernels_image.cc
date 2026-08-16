// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::DataType;
using core::runtime::RuntimeContext;
using core::runtime::Tensor;
using onnx_kernels::SimpleRawBufferAllocator;
using onnx_kernels::kernel::ImageDecoder;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

// Baseline JFIF JPEG bitstreams (encoded by libjpeg through Pillow) together
// with their libjpeg-decoded reference images. The 16x16 RGB stream uses 4:2:0
// chroma subsampling (luma sampling factors 2x2, chroma 1x1) so the test
// exercises the Huffman decode, dequantisation, IDCT and chroma upsampling
// paths; the 16x16 grayscale stream is a single-component baseline JPEG. Pixel
// values are compared to the libjpeg reference with a tolerance of 2 to absorb
// the rounding difference between the kernel's floating-point IDCT and
// libjpeg's integer IDCT, mirroring the Python regression tests in
// ``unittests/python/bindings/test_reference_evaluator.py``.
// clang-format off
const unsigned char k_jpeg_rgb_in[696] = {
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0,
    2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6,
    5, 6, 6, 6, 7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8, 11, 12, 11,
    10, 12, 9, 10, 10, 10, 255, 219, 0, 67, 1, 2, 2, 2, 2, 2, 2, 5, 3, 3, 5, 10, 7, 6, 7, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 255, 192, 0, 17, 8, 0, 16, 0, 16, 3, 1, 34, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0,
    31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    255, 196, 0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17,
    5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193,
    21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42,
    52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90,
    99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132,
    133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196,
    197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255,
    196, 0, 31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 255, 196, 0, 181, 17, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119, 0, 1, 2, 3,
    17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193,
    9, 35, 51, 82, 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39,
    40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88,
    89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162,
    163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255,
    218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 252, 237, 248, 37, 251, 27, 127, 169, 255, 0,
    137, 87, 167, 240, 87, 216, 63, 3, 255, 0, 99, 156, 121, 32, 233, 92, 113, 252, 21, 244,
    239, 192, 255, 0, 216, 231, 62, 72, 58, 87, 167, 240, 87, 216, 95, 4, 191, 99, 111, 245, 63,
    241, 42, 244, 254, 10, 50, 108, 231, 109, 67, 232, 227, 244, 142, 254, 15, 239, 187, 117,
    63, 255, 217,
};
const unsigned char k_jpeg_rgb_ref[768] = {
    0, 0, 0, 12, 3, 6, 29, 3, 16, 46, 3, 23, 63, 3, 31, 79, 3, 41, 96, 3, 48, 112, 3, 58, 131,
    2, 66, 148, 2, 73, 165, 4, 84, 182, 4, 92, 200, 4, 102, 216, 3, 109, 232, 3, 119, 244, 7,
    125, 5, 14, 11, 17, 17, 17, 34, 17, 25, 51, 17, 33, 69, 16, 42, 85, 15, 49, 102, 17, 59,
    117, 17, 67, 135, 15, 76, 153, 16, 84, 169, 17, 92, 186, 17, 100, 205, 17, 111, 222, 17,
    120, 238, 18, 128, 249, 21, 134, 5, 31, 20, 17, 34, 26, 34, 34, 34, 51, 34, 42, 68, 34, 51,
    86, 33, 59, 101, 34, 69, 119, 34, 76, 136, 34, 84, 153, 34, 92, 170, 35, 102, 186, 34, 109,
    204, 34, 120, 222, 34, 128, 239, 34, 137, 251, 38, 144, 5, 48, 28, 17, 51, 35, 34, 51, 43,
    51, 51, 51, 68, 51, 61, 85, 51, 68, 101, 51, 78, 118, 51, 86, 136, 51, 93, 153, 51, 101,
    169, 51, 111, 185, 50, 117, 203, 51, 128, 220, 50, 136, 239, 51, 145, 250, 54, 152, 5, 65,
    37, 16, 69, 43, 34, 68, 51, 51, 68, 58, 68, 68, 68, 85, 68, 76, 101, 68, 85, 118, 68, 93,
    135, 68, 101, 153, 68, 109, 168, 68, 118, 186, 68, 126, 203, 68, 135, 220, 68, 143, 238, 69,
    152, 251, 73, 161, 6, 82, 44, 16, 86, 52, 33, 86, 60, 51, 85, 68, 68, 85, 77, 85, 85, 85,
    102, 85, 95, 118, 85, 102, 135, 85, 110, 152, 85, 118, 170, 85, 127, 184, 84, 134, 203, 85,
    145, 220, 85, 152, 237, 85, 160, 250, 89, 167, 7, 98, 54, 17, 102, 60, 36, 103, 68, 52, 102,
    75, 69, 102, 85, 85, 102, 92, 102, 102, 102, 119, 102, 110, 136, 102, 118, 154, 102, 125,
    169, 102, 135, 187, 102, 143, 204, 102, 152, 221, 102, 160, 238, 101, 169, 250, 104, 175, 7,
    116, 61, 19, 119, 69, 34, 119, 77, 52, 119, 84, 69, 119, 94, 86, 119, 102, 102, 119, 111,
    119, 119, 119, 137, 117, 126, 153, 119, 135, 169, 119, 144, 186, 119, 152, 204, 119, 161,
    221, 119, 169, 238, 118, 179, 250, 121, 185, 5, 134, 70, 17, 137, 76, 34, 136, 86, 51, 136,
    94, 69, 136, 103, 86, 136, 111, 102, 136, 120, 117, 137, 128, 136, 136, 136, 153, 136, 144,
    169, 136, 153, 186, 136, 161, 203, 136, 171, 221, 136, 178, 236, 136, 186, 248, 139, 194, 4,
    150, 79, 16, 153, 85, 34, 153, 95, 51, 153, 103, 68, 153, 112, 86, 153, 120, 101, 153, 130,
    119, 153, 137, 136, 153, 145, 153, 153, 153, 170, 153, 163, 186, 153, 170, 203, 153, 180,
    219, 152, 187, 238, 153, 195, 248, 157, 201, 5, 166, 88, 18, 170, 95, 35, 170, 103, 52, 170,
    110, 70, 170, 120, 85, 170, 128, 103, 170, 137, 120, 170, 145, 137, 170, 153, 153, 170, 160,
    170, 170, 170, 187, 170, 178, 204, 170, 187, 222, 169, 195, 239, 169, 203, 249, 173, 211, 5,
    183, 95, 17, 186, 103, 34, 186, 111, 51, 186, 119, 69, 187, 129, 87, 187, 137, 102, 187,
    146, 120, 187, 154, 137, 187, 162, 154, 187, 170, 170, 187, 179, 187, 187, 187, 204, 187,
    197, 221, 187, 204, 239, 186, 212, 250, 190, 218, 5, 201, 103, 16, 204, 110, 34, 204, 118,
    51, 203, 126, 69, 204, 137, 86, 204, 144, 102, 204, 154, 120, 205, 163, 137, 204, 169, 154,
    204, 177, 170, 204, 187, 187, 204, 194, 204, 204, 204, 221, 204, 212, 238, 204, 220, 250,
    207, 227, 4, 217, 111, 16, 221, 118, 33, 221, 127, 51, 221, 135, 68, 220, 145, 85, 220, 153,
    102, 221, 163, 119, 221, 171, 136, 221, 179, 154, 221, 186, 169, 222, 196, 187, 221, 204,
    204, 221, 213, 221, 221, 221, 238, 221, 229, 250, 224, 235, 6, 234, 121, 17, 237, 127, 33,
    238, 135, 50, 238, 144, 68, 237, 154, 85, 237, 162, 102, 239, 171, 119, 239, 178, 138, 238,
    188, 153, 238, 196, 170, 240, 206, 186, 239, 213, 204, 238, 222, 221, 238, 230, 238, 238,
    238, 250, 241, 244, 11, 248, 130, 23, 252, 136, 39, 252, 146, 55, 251, 153, 73, 251, 163,
    90, 251, 171, 106, 252, 181, 124, 253, 189, 143, 252, 197, 159, 252, 207, 176, 252, 214,
    192, 252, 224, 209, 252, 232, 226, 252, 239, 243, 252, 249, 255, 255, 255,
};
const unsigned char k_jpeg_gray_in[380] = {
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0,
    2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6,
    5, 6, 6, 6, 7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8, 11, 12, 11,
    10, 12, 9, 10, 10, 10, 255, 192, 0, 11, 8, 0, 16, 0, 16, 1, 1, 17, 0, 255, 196, 0, 31, 0, 0,
    1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196,
    0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33,
    49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209,
    240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54,
    55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100,
    101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134,
    135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166,
    167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198,
    199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229,
    230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 8,
    1, 1, 0, 0, 63, 0, 252, 237, 255, 0, 130, 100, 255, 0, 204, 63, 254, 3, 95, 209, 7, 252, 19,
    39, 254, 97, 255, 0, 240, 26, 254, 119, 255, 0, 224, 153, 63, 243, 15, 255, 0, 128, 215,
    244, 65, 255, 0, 4, 201, 255, 0, 152, 127, 252, 6, 191, 255, 217,
};
const unsigned char k_jpeg_gray_ref[256] = {
    0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68,
    85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136,
    152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204,
    221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0,
    16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85,
    103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152,
    170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221,
    239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16,
    34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85,
    103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152,
    170, 187, 204, 221, 239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221,
    239, 255, 0, 16, 34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255, 0, 16,
    34, 51, 68, 85, 103, 119, 136, 152, 170, 187, 204, 221, 239, 255,
};
// clang-format on

int MaxAbsDiff(const std::vector<uint8_t> &actual, const unsigned char *expected,
               size_t num_elements) {
  int max_diff = 0;
  for (size_t i = 0; i < num_elements; ++i) {
    const int diff = std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

} // namespace

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

// Baseline 8-bit non-interlaced PNG bytestreams together with their expected
// decoded images. The test fixtures were produced ahead of time with Pillow
// and embedded as static byte arrays; at run time the deflate-compressed IDAT
// payload is decoded inline by the kernel itself, which has no dependency on
// libpng/zlib, mirroring the BMP and JPEG regression tests above.

TEST(KernelClass, ImageDecoderDecodesPngRgb) {
  // 2x2 truecolor (color type 2, 8-bit) PNG with display pixels:
  //   row 0: (90,80,70) (120,110,100)
  //   row 1: (30,20,10) (60,50,40)
  // clang-format off
  const std::vector<uint8_t> png_bytes = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2,
    0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 140, 10, 112, 147,
    147, 147, 99, 148, 19, 225, 146, 147, 147, 3, 0, 16, 149, 1, 227, 30, 13, 52, 10, 0, 0, 0,
    0, 73, 69, 78, 68, 174, 66, 96, 130,
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(png_bytes.size())}, png_bytes);

  Tensor out = decoder(encoded, "RGB");

  EXPECT_EQ(out.data_type, static_cast<int32_t>(DataType::UINT8));
  const std::vector<int64_t> expected_shape = {2, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.element_count(), 12);
  const std::vector<uint8_t> expected_pixels = {90, 80, 70, 120, 110, 100, 30, 20, 10, 60, 50, 40};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderDecodesPngBgr) {
  // Same truecolor PNG as above but pixel_format="BGR" swaps the channel order.
  // clang-format off
  const std::vector<uint8_t> png_bytes = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2,
    0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 140, 10, 112, 147,
    147, 147, 99, 148, 19, 225, 146, 147, 147, 3, 0, 16, 149, 1, 227, 30, 13, 52, 10, 0, 0, 0,
    0, 73, 69, 78, 68, 174, 66, 96, 130,
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(png_bytes.size())}, png_bytes);

  Tensor out = decoder(encoded, "BGR");

  const std::vector<int64_t> expected_shape = {2, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<uint8_t> expected_pixels = {70, 80, 90, 100, 110, 120, 10, 20, 30, 40, 50, 60};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderDecodesPngGrayscaleFromRgb) {
  // The same truecolor PNG decoded to Grayscale using the ITU-R BT.601
  // luminance formula grey = (299*R + 587*G + 114*B + 500) / 1000.
  // clang-format off
  const std::vector<uint8_t> png_bytes = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2,
    0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 140, 10, 112, 147,
    147, 147, 99, 148, 19, 225, 146, 147, 147, 3, 0, 16, 149, 1, 227, 30, 13, 52, 10, 0, 0, 0,
    0, 73, 69, 78, 68, 174, 66, 96, 130,
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(png_bytes.size())}, png_bytes);

  Tensor out = decoder(encoded, "Grayscale");

  const std::vector<int64_t> expected_shape = {2, 2, 1};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.element_count(), 4);
  // (90,80,70)->82, (120,110,100)->112, (30,20,10)->22, (60,50,40)->52.
  const std::vector<uint8_t> expected_pixels = {82, 112, 22, 52};
  EXPECT_EQ(out.data, expected_pixels);
}

TEST(KernelClass, ImageDecoderDecodesPngGrayscale) {
  // 2x2 grayscale (color type 0, 8-bit) PNG with display pixels:
  //   row 0: 10 200
  //   row 1: 100 255
  // clang-format off
  const std::vector<uint8_t> png_bytes = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 87, 221, 82, 248, 0, 0, 0, 14, 73, 68, 65, 84, 120, 156, 99, 224, 58, 193, 144,
    242, 31, 0, 5, 31, 2, 54, 131, 148, 64, 10, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(png_bytes.size())}, png_bytes);

  Tensor out = decoder(encoded, "Grayscale");

  const std::vector<int64_t> expected_shape = {2, 2, 1};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.element_count(), 4);
  const std::vector<uint8_t> expected_pixels = {10, 200, 100, 255};
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

TEST(KernelClass, ImageDecoderDecodesJpegRgb) {
  // Baseline JFIF (SOF0) JPEG with 4:2:0 chroma subsampling decoded to RGB.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded =
      Tensor::FromUint8("", {static_cast<int64_t>(sizeof(k_jpeg_rgb_in))},
                        std::vector<uint8_t>(k_jpeg_rgb_in, k_jpeg_rgb_in + sizeof(k_jpeg_rgb_in)));

  Tensor out = decoder(encoded, "RGB");

  const std::vector<int64_t> expected_shape = {16, 16, 3};
  EXPECT_EQ(out.data_type, static_cast<int32_t>(DataType::UINT8));
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.data.size(), sizeof(k_jpeg_rgb_ref));
  EXPECT_LE(MaxAbsDiff(out.data, k_jpeg_rgb_ref, sizeof(k_jpeg_rgb_ref)), 2);
}

TEST(KernelClass, ImageDecoderDecodesJpegBgr) {
  // Same RGB JPEG, but pixel_format="BGR" swaps the channel order per pixel.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded =
      Tensor::FromUint8("", {static_cast<int64_t>(sizeof(k_jpeg_rgb_in))},
                        std::vector<uint8_t>(k_jpeg_rgb_in, k_jpeg_rgb_in + sizeof(k_jpeg_rgb_in)));

  Tensor out = decoder(encoded, "BGR");

  const std::vector<int64_t> expected_shape = {16, 16, 3};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.data.size(), sizeof(k_jpeg_rgb_ref));
  std::vector<uint8_t> bgr_ref(sizeof(k_jpeg_rgb_ref));
  for (size_t pixel = 0; pixel < sizeof(k_jpeg_rgb_ref) / 3; ++pixel) {
    bgr_ref[3 * pixel + 0] = k_jpeg_rgb_ref[3 * pixel + 2];
    bgr_ref[3 * pixel + 1] = k_jpeg_rgb_ref[3 * pixel + 1];
    bgr_ref[3 * pixel + 2] = k_jpeg_rgb_ref[3 * pixel + 0];
  }
  EXPECT_LE(MaxAbsDiff(out.data, bgr_ref.data(), bgr_ref.size()), 2);
}

TEST(KernelClass, ImageDecoderDecodesJpegGrayscale) {
  // Single-component baseline JFIF JPEG decoded to a (H, W, 1) Grayscale image.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8(
      "", {static_cast<int64_t>(sizeof(k_jpeg_gray_in))},
      std::vector<uint8_t>(k_jpeg_gray_in, k_jpeg_gray_in + sizeof(k_jpeg_gray_in)));

  Tensor out = decoder(encoded, "Grayscale");

  const std::vector<int64_t> expected_shape = {16, 16, 1};
  EXPECT_EQ(out.shape, expected_shape);
  ASSERT_EQ(out.data.size(), sizeof(k_jpeg_gray_ref));
  EXPECT_LE(MaxAbsDiff(out.data, k_jpeg_gray_ref, sizeof(k_jpeg_gray_ref)), 2);
}

TEST(KernelClass, ImageDecoderDecodesPnmRgb) {
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  const auto cases =
      core::backend_test::CollectTestCasesByName("^test_cc_image_decoder_decode_pnm_rgb$");
  ASSERT_EQ(cases.size(), 1u);
  ASSERT_EQ(cases[0].data_sets().size(), 1u);
  ASSERT_EQ(cases[0].data_sets()[0].inputs.size(), 1u);
  ASSERT_EQ(cases[0].data_sets()[0].outputs.size(), 1u);

  const Tensor &encoded = cases[0].data_sets()[0].inputs[0];
  const Tensor &expected = cases[0].data_sets()[0].outputs[0];
  Tensor out = decoder(encoded, "RGB");

  EXPECT_EQ(out.data_type, static_cast<int32_t>(DataType::UINT8));
  EXPECT_EQ(out.shape, expected.shape);
  ASSERT_EQ(out.data.size(), expected.data.size());
  EXPECT_EQ(out.data, expected.data);
}

TEST(KernelClass, ImageDecoderDecodesPnmBinaryPpmAndConvertsToBgrAndGrayscale) {
  // Minimal 2x1 binary pixmap (P6): pixels (10, 20, 30) and (40, 50, 60).
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  const std::vector<uint8_t> header = {'P', '6', '\n', '2', ' ', '1', '\n', '2', '5', '5', '\n'};
  std::vector<uint8_t> ppm = header;
  const std::vector<uint8_t> body = {10, 20, 30, 40, 50, 60};
  ppm.insert(ppm.end(), body.begin(), body.end());
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(ppm.size())}, ppm);

  Tensor rgb = decoder(encoded, "RGB");
  const std::vector<int64_t> rgb_shape = {1, 2, 3};
  EXPECT_EQ(rgb.shape, rgb_shape);
  EXPECT_EQ(rgb.data, body);

  Tensor bgr = decoder(encoded, "BGR");
  const std::vector<uint8_t> expected_bgr = {30, 20, 10, 60, 50, 40};
  EXPECT_EQ(bgr.data, expected_bgr);

  Tensor gray = decoder(encoded, "Grayscale");
  const std::vector<int64_t> gray_shape = {1, 2, 1};
  EXPECT_EQ(gray.shape, gray_shape);
  const std::vector<uint8_t> expected_gray = {
      static_cast<uint8_t>((299 * 10 + 587 * 20 + 114 * 30 + 500) / 1000),
      static_cast<uint8_t>((299 * 40 + 587 * 50 + 114 * 60 + 500) / 1000)};
  EXPECT_EQ(gray.data, expected_gray);
}

TEST(KernelClass, ImageDecoderDecodesAsciiPgmGraymap) {
  // 2x2 ASCII graymap (P2) with a comment line and maxval 255.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  const std::string pgm = "P2\n# sample graymap\n2 2\n255\n0 64\n128 255\n";
  std::vector<uint8_t> bytes(pgm.begin(), pgm.end());
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bytes.size())}, bytes);

  Tensor out = decoder(encoded, "Grayscale");
  const std::vector<int64_t> expected_shape = {2, 2, 1};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<uint8_t> expected_pixels = {0, 64, 128, 255};
  EXPECT_EQ(out.data, expected_pixels);
}

// Capacity for the SimpleRawBufferAllocator used in allocator tests below.
// Must be at least the maximum number of concurrent TemporaryTypedBuffer
// allocations made by a single TryDecode* call: PNG uses 1 (rows), PNM uses 1
// (src), JPEG uses up to 4 (one samples buffer per colour component).
constexpr size_t kAllocatorTestCapacity = 8;

TEST(KernelClass, ImageDecoderPngUsesAllocatorForInternalBuffers) {
  // Verifies that TryDecodePng allocates its temporary `rows` buffer through
  // the provided allocator and releases it before returning. The decoded pixels
  // must match the reference values.
  // clang-format off
  const std::vector<uint8_t> png_bytes = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 2,
    0, 0, 0, 253, 212, 154, 115, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 140, 10, 112, 147,
    147, 147, 99, 148, 19, 225, 146, 147, 147, 3, 0, 16, 149, 1, 227, 30, 13, 52, 10, 0, 0, 0,
    0, 73, 69, 78, 68, 174, 66, 96, 130,
  };
  // clang-format on
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(png_bytes.size())}, png_bytes);

  SimpleRawBufferAllocator alloc(kAllocatorTestCapacity);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor out = decoder(encoded, "RGB", &rt);

  // Every temporary buffer is freed; the one remaining allocation is the
  // decoded output tensor, which is routed through the runtime allocator.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_TRUE(out.has_allocation());
  EXPECT_EQ(out.allocation_owner(), &alloc);
  // Output pixels must match the reference values.
  const std::vector<int64_t> expected_shape = {2, 2, 3};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<uint8_t> expected_pixels = {90, 80, 70, 120, 110, 100, 30, 20, 10, 60, 50, 40};
  const std::vector<uint8_t> actual_pixels(out.bytes(), out.bytes() + out.size_bytes());
  EXPECT_EQ(actual_pixels, expected_pixels);
}

TEST(KernelClass, ImageDecoderJpegUsesAllocatorForInternalBuffers) {
  // Verifies that TryDecodeJpeg allocates the per-component `samples`
  // buffer through the provided allocator and releases it before returning.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  Tensor encoded =
      Tensor::FromUint8("", {static_cast<int64_t>(sizeof(k_jpeg_rgb_in))},
                        std::vector<uint8_t>(k_jpeg_rgb_in, k_jpeg_rgb_in + sizeof(k_jpeg_rgb_in)));

  SimpleRawBufferAllocator alloc(kAllocatorTestCapacity);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor out = decoder(encoded, "RGB", &rt);

  // Every temporary buffer is freed; the one remaining allocation is the
  // decoded output tensor, which is routed through the runtime allocator.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_TRUE(out.has_allocation());
  EXPECT_EQ(out.allocation_owner(), &alloc);
  // Output must be a non-empty 16x16x3 tensor.
  ASSERT_EQ(out.shape.size(), 3u);
  EXPECT_EQ(out.shape[0], 16);
  EXPECT_EQ(out.shape[1], 16);
  EXPECT_EQ(out.shape[2], 3);
}

TEST(KernelClass, ImageDecoderPnmUsesAllocatorForInternalBuffers) {
  // Verifies that TryDecodePnm allocates its temporary `src` buffer through
  // the provided allocator and releases it before returning.
  const KernelContext ctx{DefaultOpset(20)};
  const ImageDecoder decoder{ctx};
  const std::string pgm = "P2\n# graymap\n2 2\n255\n0 64\n128 255\n";
  std::vector<uint8_t> bytes(pgm.begin(), pgm.end());
  Tensor encoded = Tensor::FromUint8("", {static_cast<int64_t>(bytes.size())}, bytes);

  SimpleRawBufferAllocator alloc(kAllocatorTestCapacity);
  RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor out = decoder(encoded, "Grayscale", &rt);

  // Every temporary buffer is freed; the one remaining allocation is the
  // decoded output tensor, which is routed through the runtime allocator.
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_TRUE(out.has_allocation());
  EXPECT_EQ(out.allocation_owner(), &alloc);
  // Output pixels must match the reference values.
  const std::vector<int64_t> expected_shape = {2, 2, 1};
  EXPECT_EQ(out.shape, expected_shape);
  const std::vector<uint8_t> expected_pixels = {0, 64, 128, 255};
  const std::vector<uint8_t> actual_pixels(out.bytes(), out.bytes() + out.size_bytes());
  EXPECT_EQ(actual_pixels, expected_pixels);
}

} // namespace Test
