// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_ort/schema/operator_sets_microsoft.h"
#include "onnx_ort/schema/operator_sets_microsoft_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace microsoft {

namespace {

LightOpSchema MakeBiasGeluSchema() {
  return LightOpSchema(
      "BiasGelu", kMicrosoftDomain, 1, MakeBiasGeluDoc(),
      {
          {"A", "The normal input data.", "T"},
          {"B", "The bias input data that is a 1D tensor.", "T"},
      },
      {
          {"C", "The output.", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      });
}

LightOpSchema MakeBiasGeluGradDxSchema() {
  return LightOpSchema(
      "BiasGeluGrad_dX", kMicrosoftDomain, 1, MakeBiasGeluGradDxDoc(),
      {
          {"dY", "The gradient tensor from output.", "T"},
          {"X", "The input tensor.", "T"},
          {"B", "The bias tensor.", "T"},
      },
      {
          {"dX", "Gradient of the input.", "T"},
      },
      {
          {"T",
           {TensorType::kFloat16, TensorType::kFloat, TensorType::kDouble, TensorType::kBfloat16},
           "Constrain input and output types to float tensors."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpMicrosoftSchemasWithHistory(const std::string &op_type,
                                                                   bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"BiasGelu", [] { return std::vector<LightOpSchema>{MakeBiasGeluSchema()}; }},
      {"BiasGeluGrad_dX", [] { return std::vector<LightOpSchema>{MakeBiasGeluGradDxSchema()}; }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace microsoft
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
