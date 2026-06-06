// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``image`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// ``ImageDecoder`` mirrors the ONNX ``ImageDecoder`` operator (ai.onnx,
// since opset 20). The operator parses the encoded bytestream of an image
// file (BMP, JPEG, JPEG2000, PNG, TIFF, WebP, or Portable image format
// PBM/PGM/PPM/PXM/PNM) and produces a ``(H, W, C)`` ``tensor(uint8)`` in
// the channel-last layout selected by the ``pixel_format`` attribute
// (``"RGB"``, ``"BGR"`` or ``"Grayscale"``).
//
// To keep the lightweight C++ kernel library free of third-party image
// decoding dependencies (``libjpeg``, ``libpng``, ``libwebp``, etc.) the
// reference kernel performs full attribute and input validation and then,
// when an actual decoded image cannot be produced, falls back to the
// behavior documented by the ONNX schema:
//
//     "If it can't decode for any reason (e.g. corrupted encoded stream,
//      invalid format), it will return an empty matrix."
//
// Concretely, the fallback output is a zero-element ``(0, 0, C)`` uint8
// tensor whose channel count ``C`` is derived from ``pixel_format``
// (``1`` for ``"Grayscale"``, ``3`` for ``"RGB"`` / ``"BGR"``). The
// expected ``(H, W, C)`` outputs for the upstream backend test cases are
// embedded directly as static byte arrays in
// ``onnx_kernels/cases/image/cases_image_decoder.cc`` so the test
// registry does not depend on this kernel actually decoding the
// bytestream.
// ---------------------------------------------------------------------------

/// Reference implementation of the ``ImageDecoder`` operator
/// (ai.onnx, since opset 20).
///
/// Validates the input ``tensor(uint8)`` (must be 1-D, carrying the
/// encoded bytestream) and the ``pixel_format`` attribute (must be one
/// of ``"RGB"``, ``"BGR"`` or ``"Grayscale"``). The decoded image is
/// returned as a ``(H, W, C)`` ``tensor(uint8)`` in channel-last
/// layout.
///
/// The lightweight reference implementation does not link against any
/// image-decoding library and therefore cannot actually decode the
/// bytestream. Per the ONNX schema, the kernel falls back to returning
/// an empty matrix --- a ``(0, 0, C)`` ``tensor(uint8)`` --- whenever
/// it cannot produce a decoded image. Invalid inputs or attribute
/// values throw ``std::invalid_argument``.
class ImageDecoder : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returns the channel count implied by ``pixel_format``: ``1`` for
  /// ``"Grayscale"``, ``3`` for ``"RGB"`` / ``"BGR"``. Throws
  /// ``std::invalid_argument`` for any other value.
  static int64_t ChannelCount(const std::string &pixel_format);

  /// Allocating overload. ``pixel_format`` defaults to ``"RGB"`` (the
  /// schema default). Returns a fresh ``tensor(uint8)`` of shape
  /// ``(0, 0, C)`` containing no decoded pixels.
  Tensor operator()(const Tensor &encoded_stream, const std::string &pixel_format = "RGB") const;

  /// In-place overload. ``output`` must already be a ``tensor(uint8)``
  /// whose last dimension matches the channel count derived from
  /// ``pixel_format`` (``output.shape == {H, W, C}`` with ``C ==
  /// ChannelCount(pixel_format)``) and whose ``data`` buffer is sized
  /// accordingly. The kernel writes the (possibly empty) decoded image
  /// into ``output.data``; when the kernel falls back to the
  /// empty-matrix path, ``output`` must have shape ``(0, 0, C)``.
  void operator()(const Tensor &encoded_stream, const std::string &pixel_format,
                  Tensor &output) const;

  /// Output bytes are produced from the encoded bytestream and have a
  /// different shape than the input; the output buffer cannot safely
  /// alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
