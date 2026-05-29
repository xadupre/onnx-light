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

std::vector<LightOpSchema> GetAllOnnxOpImageSchemasWithHistory(const std::string &op_type,
                                                               bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"ImageDecoder", [] { return std::vector<LightOpSchema>{MakeImageDecoderSchema(20)}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace image
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
