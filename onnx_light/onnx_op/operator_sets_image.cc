// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_image.h"
#include "onnx_op/operator_sets_image_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace image {

namespace {

LightOpSchema MakeImageDecoderSchema(int since_version) {
  return LightOpSchema(
      "ImageDecoder", kOnnxDomain, since_version, MakeImageDecoderDoc(since_version),
      {
          {"encoded_stream", "Encoded stream", "T1"},
      },
      {
          {"image", "Decoded image", "T2"},
      },
      {
          {"T1", {TensorType::kUint8}, "Constrain input types to 8-bit unsigned integer tensor."},
          {"T2", {TensorType::kUint8}, "Constrain output types to 8-bit unsigned integer tensor."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpImageSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      MakeImageDecoderSchema(20),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace image
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
