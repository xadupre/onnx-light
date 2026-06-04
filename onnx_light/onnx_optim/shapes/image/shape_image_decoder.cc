// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/image/shape_image.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace image {

void ComputeShapeImageDecoder(ShapesContext &ctx, const NodeProto &node, const char *a) {
  CheckNodeOpAndOutput(node, "ImageDecoder", "ComputeShapeImageDecoder");
  const OptimTensor &input = ctx.Get(a);
  const OptimShape &in_shape = input.Shape();

  // ImageDecoder requires a 1-D ``tensor(uint8)`` input carrying the
  // encoded bytestream.
  if (in_shape.Rank() != 1) {
    throw std::invalid_argument(std::string("ComputeShapeImageDecoder: input '") + a +
                                "' must be 1-dimensional, got rank " +
                                std::to_string(in_shape.Rank()) + ".");
  }

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
    throw std::invalid_argument("ComputeShapeImageDecoder: unsupported pixel_format '" +
                                pixel_format + "' (expected 'RGB', 'BGR' or 'Grayscale').");
  }

  // The spatial extent of the decoded image is only known at runtime,
  // so the H and W dimensions are kept symbolic and unique per output.
  const std::string output_name = node.output(0).as_string();
  OptimShape out_shape{
      OptimDim("ImageDecoder(" + output_name + ").H"),
      OptimDim("ImageDecoder(" + output_name + ").W"),
      OptimDim(channels),
  };
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kUint8, std::move(out_shape)));
}

} // namespace image
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
