// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/image/include_image_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

void CheckImageDecoderInput(const Tensor &encoded_stream) {
  EXT_ENFORCE_INVALID(encoded_stream.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::ImageDecoder only supports UINT8 input tensors.");
  EXT_ENFORCE_INVALID(encoded_stream.shape.size() == 1u,
                      "kernel::ImageDecoder input ``encoded_stream`` must be a 1-D tensor "
                      "carrying the encoded bytestream.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(encoded_stream.size_bytes()) ==
                          encoded_stream.element_count(),
                      "kernel::ImageDecoder input ``encoded_stream`` data size does not "
                      "match its shape.");
}

// ---------------------------------------------------------------------------
// BMP decoder
//
// Decodes an uncompressed 24-bit BMP bytestream (BI_RGB, BITMAPINFOHEADER)
// without any external library dependency. Returns true on success and fills
// ``pixels`` with the channel-last RGB (or BGR / Grayscale) pixel data; on
// failure returns false so the caller can fall back to the empty-matrix path.
// ---------------------------------------------------------------------------

inline uint16_t ReadU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t ReadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline int32_t ReadI32LE(const uint8_t *p) { return static_cast<int32_t>(ReadU32LE(p)); }

bool TryDecodeBmp(const uint8_t *data, size_t size, const std::string &pixel_format,
                  int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels) {
  // Minimum BMP file: 14-byte file header + 40-byte BITMAPINFOHEADER = 54 bytes.
  if (size < 54) {
    return false;
  }
  // Check "BM" magic.
  if (data[0] != 0x42 || data[1] != 0x4D) {
    return false;
  }

  const uint32_t pixel_offset = ReadU32LE(data + 10);
  const uint32_t dib_header_size = ReadU32LE(data + 14);

  // Only BITMAPINFOHEADER (size==40) is supported; BITMAPCOREHEADER (size==12)
  // and extended variants are not encountered in the backend test data.
  if (dib_header_size < 40) {
    return false;
  }

  const int32_t width_signed = ReadI32LE(data + 18);
  const int32_t height_signed = ReadI32LE(data + 22);
  const uint16_t bpp = ReadU16LE(data + 28);
  const uint32_t compression = ReadU32LE(data + 30);

  // Only uncompressed 24-bit BGR is supported.
  if (bpp != 24 || compression != 0) {
    return false;
  }
  if (width_signed <= 0) {
    return false;
  }

  const int32_t width = width_signed;
  // Negative height means top-down (rows are already in display order).
  const bool bottom_up = height_signed > 0;
  const int32_t height = bottom_up ? height_signed : -height_signed;

  if (height <= 0) {
    return false;
  }

  // Row stride is padded to a 4-byte boundary.
  const uint32_t row_stride = (static_cast<uint32_t>(width) * 3u + 3u) & ~3u;
  const uint64_t required_bytes =
      static_cast<uint64_t>(pixel_offset) + static_cast<uint64_t>(row_stride) * height;
  if (required_bytes > size) {
    return false;
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  const size_t total_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
  out_pixels.resize(total_pixels * static_cast<size_t>(channels));

  const uint8_t *pixel_data = data + pixel_offset;
  for (int32_t row = 0; row < height; ++row) {
    // BMP rows are stored bottom-to-top when height is positive.
    const int32_t src_row = bottom_up ? (height - 1 - row) : row;
    const uint8_t *src_row_ptr = pixel_data + static_cast<size_t>(src_row) * row_stride;
    uint8_t *dst_row_ptr =
        out_pixels.data() + static_cast<size_t>(row) * static_cast<size_t>(width) * channels;

    for (int32_t col = 0; col < width; ++col) {
      const uint8_t b = src_row_ptr[col * 3 + 0];
      const uint8_t g = src_row_ptr[col * 3 + 1];
      const uint8_t r = src_row_ptr[col * 3 + 2];
      uint8_t *dst = dst_row_ptr + col * channels;
      if (pixel_format == "RGB") {
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
      } else if (pixel_format == "BGR") {
        dst[0] = b;
        dst[1] = g;
        dst[2] = r;
      } else {
        // Grayscale: standard luminance formula (ITU-R BT.601).
        dst[0] = static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000);
      }
    }
  }

  out_height = static_cast<int64_t>(height);
  out_width = static_cast<int64_t>(width);
  return true;
}

} // namespace

int64_t ImageDecoder::ChannelCount(const std::string &pixel_format) {
  if (pixel_format == "RGB" || pixel_format == "BGR") {
    return 3;
  }
  if (pixel_format == "Grayscale") {
    return 1;
  }
  throw std::invalid_argument("kernel::ImageDecoder: unsupported pixel_format \"" + pixel_format +
                              "\" (expected \"RGB\", \"BGR\" or \"Grayscale\").");
}

Tensor ImageDecoder::operator()(const Tensor &encoded_stream,
                                const std::string &pixel_format) const {
  CheckImageDecoderInput(encoded_stream);
  const int64_t channels = ChannelCount(pixel_format);

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  std::vector<uint8_t> pixels;
  if (TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels)) {
    return Tensor::FromUint8("", {height, width, channels}, std::move(pixels));
  }

  // Per the ONNX schema, fall back to an empty ``(0, 0, C)`` matrix when
  // the bytestream cannot be decoded.
  return Tensor::FromUint8("", {0, 0, channels}, {});
}

void ImageDecoder::operator()(const Tensor &encoded_stream, const std::string &pixel_format,
                              Tensor &output) const {
  CheckImageDecoderInput(encoded_stream);
  const int64_t channels = ChannelCount(pixel_format);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::ImageDecoder preallocated output must be a UINT8 tensor.");
  EXT_ENFORCE_INVALID(output.shape.size() == 3u,
                      "kernel::ImageDecoder preallocated output must be a 3-D tensor "
                      "with channel-last layout (H, W, C).");
  EXT_ENFORCE_INVALID(output.shape[2] == channels,
                      "kernel::ImageDecoder preallocated output channel count does not "
                      "match ``pixel_format``.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.data.size()) == output.element_count(),
                      "kernel::ImageDecoder preallocated output data size does not match "
                      "its shape.");

  const uint8_t *raw = encoded_stream.bytes();
  const size_t raw_size = encoded_stream.size_bytes();

  int64_t height = 0;
  int64_t width = 0;
  std::vector<uint8_t> pixels;
  if (TryDecodeBmp(raw, raw_size, pixel_format, height, width, pixels)) {
    EXT_ENFORCE_INVALID(output.shape[0] == height && output.shape[1] == width,
                        "kernel::ImageDecoder preallocated output shape does not match the decoded "
                        "BMP dimensions.");
    output.data = std::move(pixels);
    return;
  }

  // Fallback empty-matrix path: the caller must provide a ``(0, 0, C)``
  // tensor since the kernel cannot decode the bytestream.
  EXT_ENFORCE_INVALID(output.shape[0] == 0 && output.shape[1] == 0,
                      "kernel::ImageDecoder preallocated output must be (0, 0, C) for the "
                      "empty-matrix fallback used by the lightweight reference kernel.");
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
