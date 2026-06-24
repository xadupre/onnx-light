// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_rt.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace rt {

namespace {

std::string MakeDelayedInitializerDoc() {
  return "Defers loading of an initializer tensor until runtime."
         "\n\n"
         "This lightweight runtime-only operator records where the initializer data lives on "
         "disk, which device it should first be loaded on, and which device should receive the "
         "tensor at execution time. Its static output shape comes from the required `shape` "
         "attribute and its element type comes from the required `dtype` attribute.";
}

LightOpSchema MakeDelayedInitializerSchema() {
  return LightOpSchema(
      "DelayedInitializer", kAiRtDomain, 1, MakeDelayedInitializerDoc(), {},
      {
          {"output", "Tensor produced by the delayed initializer.", "T"},
      },
      {
          {"T", AllTensorTypes(), "Constrain output to any tensor type."},
      },
      {
          {"shape", "Shape of the output tensor.", AttributeType::INTS, /*required=*/true,
           std::monostate{}},
          {"dtype", "Element type of the output tensor, encoded as a TensorProto::DataType value.",
           AttributeType::INT, /*required=*/true, std::monostate{}},
          {"load_device", "Device where the initializer is first loaded.", AttributeType::STRING,
           /*required=*/true, std::monostate{}},
          {"runtime_device", "Device where the initializer is moved at runtime.",
           AttributeType::STRING, /*required=*/true, std::monostate{}},
          {"filename", "Filename containing the serialized tensor payload.", AttributeType::STRING,
           /*required=*/true, std::monostate{}},
          {"offset", "Byte offset of the tensor payload within `filename`.", AttributeType::INT,
           /*required=*/true, std::monostate{}},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpRtSchemasWithHistory(const std::string &op_type,
                                                            bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"DelayedInitializer",
       [] { return std::vector<LightOpSchema>{MakeDelayedInitializerSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace rt
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
