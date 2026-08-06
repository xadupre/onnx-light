// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/image/shape_image.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::image {

void ComputeShapeImageDecoder(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "ImageDecoder", "ComputeShapeImageDecoder");
  const SymTensor &input = ctx.Get(a);
  const SymShape &in_shape = input.Shape();

  // ImageDecoder requires a 1-D ``tensor(uint8)`` input carrying the
  // encoded bytestream.
  EXT_ENFORCE_INVALID(in_shape.Rank() == 1, "ComputeShapeImageDecoder: input '", a,
                      "' must be 1-dimensional, got rank ", in_shape.Rank(), ".");

  // Channel count is determined by the pixel_format attribute. The
  // schema documents three valid values; default is "RGB".
  const std::string pixel_format =
      GetAttributeOr<std::string>(node, "pixel_format", std::string("RGB"));
  int64_t channels = 0;
  if (pixel_format == "RGB" || pixel_format == "BGR") {
    channels = 3;
  } else if (pixel_format == "Grayscale") {
    channels = 1;
  } else {
    EXT_THROW_INVALID("ComputeShapeImageDecoder: unsupported pixel_format '", pixel_format,
                      "' (expected 'RGB', 'BGR' or 'Grayscale').");
  }

  // The spatial extent of the decoded image is only known at runtime,
  // so the H and W dimensions are kept symbolic and unique per output.
  const std::string output_name = node.output(0);
  SymShape out_shape{
      SymDim("ImageDecoder(" + output_name + ").H"),
      SymDim("ImageDecoder(" + output_name + ").W"),
      SymDim(channels),
  };
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kUint8, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::image
