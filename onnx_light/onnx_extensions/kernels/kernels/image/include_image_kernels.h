// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"

#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
// Re-exports the runtime types moved to ``onnx_core::runtime`` so
// kernel implementations below can keep referring to them
// unqualified, matching pre-move code.
using namespace ::onnx_light::core::runtime;

namespace kernel {
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::OpsetId;

// ---------------------------------------------------------------------------
// Reference implementations of the ``image`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// ``ImageDecoder`` mirrors the ONNX ``ImageDecoder`` operator (ai.onnx,
// since opset 20). The operator parses the encoded bytestream of an image
// file (BMP, JPEG, PNG, or Portable image format PBM/PGM/PPM/PXM/PNM) and
// produces a ``(H, W, C)`` ``tensor(uint8)`` in the channel-last layout
// selected by the ``pixel_format`` attribute (``"RGB"``, ``"BGR"`` or
// ``"Grayscale"``).
//
// To keep the lightweight C++ kernel library free of build-time third-party
// image decoding dependencies (``libjpeg``, ``libpng``, etc.) the reference
// kernel implements every supported decoder inline:
//
//   * **BMP** — 24-bit uncompressed (BI_RGB, BITMAPINFOHEADER): fully
//     decoded to ``(H, W, C)`` uint8 output in the requested
//     ``pixel_format``.
//   * **JPEG** — baseline JFIF (SOF0, 8-bit precision, 1 or 3
//     components, horizontal/vertical sampling factors in ``{1, 2}``,
//     optional restart intervals): fully decoded to ``(H, W, C)`` uint8
//     output in the requested ``pixel_format`` after applying the
//     standard JFIF YCbCr → RGB conversion (ITU-R BT.601, full range).
//     Non-baseline JPEGs (progressive, arithmetic-coded, 12-bit
//     precision, lossless) fall through to the empty-matrix path.
//   * **PNG** — 8-bit non-interlaced grayscale (color type 0) or
//     truecolor (color type 2) with single or multiple ``IDAT`` chunks:
//     fully decoded to ``(H, W, C)`` uint8 output in the requested
//     ``pixel_format``. DEFLATE (RFC 1951) and zlib (RFC 1950) are
//     implemented inline so no external ``libpng`` / ``zlib`` dependency
//     is required. Palette, alpha (color types 3/4/6), 16-bit depth and
//     interlaced PNGs fall through to the empty-matrix path.
//   * **PNM** — the Netpbm family (``P1``/``P4`` bitmaps, ``P2``/``P5``
//     graymaps, ``P3``/``P6`` pixmaps) with 8-bit samples
//     (``maxval <= 255``): fully decoded inline to ``(H, W, C)`` uint8
//     output in the requested ``pixel_format``. 16-bit (``maxval > 255``)
//     graymaps/pixmaps fall through to the empty-matrix path.
//
// Other image formats defined by the ONNX schema (e.g. TIFF, WebP or
// JPEG2000) are not decoded and fall back to the behavior documented by
// the ONNX schema:
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
/// BMP (24-bit uncompressed, BI_RGB) and baseline-sequential JPEG
/// (JFIF, SOF0, 8-bit precision, 1 or 3 components, sampling factors
/// in ``{1, 2}``, optional restart intervals) images, as well as 8-bit
/// non-interlaced grayscale/truecolor PNG (color types 0 and 2), are
/// decoded natively without any external library dependency. The
/// Netpbm family (``P1``-``P6`` with 8-bit samples) is also decoded
/// natively. Bytestreams that cannot be decoded fall back to returning
/// an empty matrix (``(0, 0, C)`` ``tensor(uint8)``). Invalid inputs
/// or attribute values throw ``std::invalid_argument``.
class ImageDecoder : public KernelBase {
public:
  static constexpr const char *name = "onnx_kernels:CPU:ai.onnx:ImageDecoder";
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Returns the channel count implied by ``pixel_format``: ``1`` for
  /// ``"Grayscale"``, ``3`` for ``"RGB"`` / ``"BGR"``. Throws
  /// ``std::invalid_argument`` for any other value.
  static int64_t ChannelCount(const std::string &pixel_format);

  /// Allocating overload. ``pixel_format`` defaults to ``"RGB"`` (the
  /// schema default). Decodes BMP (24-bit uncompressed) to ``(H, W, C)``
  /// uint8. Falls back to ``(0, 0, C)`` for unsupported/unrecognised formats.
  Tensor operator()(const Tensor &encoded_stream, const std::string &pixel_format = "RGB",
                    RuntimeContext *rt = nullptr) const;

  /// In-place overload. ``output`` must already be a ``tensor(uint8)``
  /// whose last dimension matches the channel count derived from
  /// ``pixel_format`` (``output.shape == {H, W, C}`` with ``C ==
  /// ChannelCount(pixel_format)``). When the bytestream is a supported BMP
  /// the decoded pixels are written into ``output.data`` and the shape must
  /// match the decoded image dimensions; when decoding falls back to the
  /// empty-matrix path, ``output`` must have shape ``(0, 0, C)``.
  void operator()(const Tensor &encoded_stream, const std::string &pixel_format,
                  Tensor &output) const;

  /// Output bytes are produced from the encoded bytestream and have a
  /// different shape than the input; the output buffer cannot safely
  /// alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
