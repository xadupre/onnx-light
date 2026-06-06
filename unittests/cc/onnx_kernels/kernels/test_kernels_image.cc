// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/image/include_image_kernels.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DataType;
using onnx_kernels::DefaultOpset;
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
