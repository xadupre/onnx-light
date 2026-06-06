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
  EXT_ENFORCE_INVALID(static_cast<int64_t>(encoded_stream.data.size()) ==
                          encoded_stream.element_count(),
                      "kernel::ImageDecoder input ``encoded_stream`` data size does not "
                      "match its shape.");
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
  // The reference kernel does not link against any image-decoding
  // library and cannot actually decode ``encoded_stream``. Per the ONNX
  // schema, fall back to returning an empty matrix: a ``(0, 0, C)``
  // ``tensor(uint8)`` whose channel count is derived from
  // ``pixel_format``.
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
  // Fallback empty-matrix path: the caller must provide a ``(0, 0, C)``
  // tensor since the kernel cannot decode the bytestream without an
  // image-decoding dependency.
  EXT_ENFORCE_INVALID(output.shape[0] == 0 && output.shape[1] == 0,
                      "kernel::ImageDecoder preallocated output must be (0, 0, C) for the "
                      "empty-matrix fallback used by the lightweight reference kernel.");
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
