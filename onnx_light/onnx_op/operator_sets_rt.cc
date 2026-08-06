// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_rt.h"
#include "onnx_op/operator_sets_rt_doc.h"

#include <algorithm>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt {

namespace {

std::vector<TensorType> DelayedInitializerTensorTypes() {
  std::vector<TensorType> types = AllTensorTypes();
  types.erase(std::remove(types.begin(), types.end(), TensorType::kString), types.end());
  return types;
}

LightOpSchema MakeDelayedInitializerSchema() {
  return LightOpSchema(
      "DelayedInitializer", kAiRtDomain, 1, MakeDelayedInitializerDoc(), {},
      {
          {"output", "Tensor produced by the delayed initializer.", "T"},
      },
      {
          {"T", DelayedInitializerTensorTypes(),
           "Constrain output to tensor types backed by raw byte storage."},
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt
