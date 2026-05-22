// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_image_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace image {

std::string MakeImageDecoderDoc(int since_version) {
  if (since_version == 20) {
    return R"DOC(Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
stream, invalid format, it will return an empty matrix).
The following image formats are supported:
* BMP
* JPEG (note: Lossless JPEG support is optional)
* JPEG2000
* TIFF
* PNG
* WebP
* Portable image format (PBM, PGM, PPM, PXM, PNM)
Decoded images follow a channel-last layout: (Height, Width, Channels).
**JPEG chroma upsampling method:**
When upsampling the chroma components by a factor of 2, the pixels are linearly interpolated so that the
centers of the output pixels are 1/4 and 3/4 of the way between input pixel centers.
When rounding, 0.5 is rounded down and up at alternative pixels locations to prevent bias towards
larger values (ordered dither pattern).
Considering adjacent input pixels A, B, and C, B is upsampled to pixels B0 and B1 so that
```
B0 = round_half_down((1/4) * A + (3/4) * B)
B1 = round_half_up((3/4) * B + (1/4) * C)
```
This method,  is the default chroma upsampling method in the well-established libjpeg-turbo library,
also referred as "smooth" or "fancy" upsampling.
)DOC";
  }
  return "";
}

} // namespace image
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
