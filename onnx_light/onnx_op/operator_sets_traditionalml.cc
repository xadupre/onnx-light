// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml.h"
#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

namespace {

std::vector<TypeConstraintParam::AllowedType> LabelEncoderTypes() {
  return {
      TensorType::kString, TensorType::kInt64, TensorType::kFloat,
      TensorType::kInt32,  TensorType::kInt16, TensorType::kDouble,
  };
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTraditionalMLSchemasWithHistory() {
  return std::vector<LightOpSchema>{LightOpSchema(
      "LabelEncoder", "ai.onnx.ml", 4, MakeLabelEncoderDoc(),
      {
          {"X", "Input data. It must have the same element type as the keys_* attribute set.",
           "T1"},
      },
      {
          {"Y", "Output data. This tensor's element type is based on the values_* attribute set.",
           "T2"},
      },
      {
          {"T1", LabelEncoderTypes(), "The input type is a tensor of any shape."},
          {"T2", LabelEncoderTypes(),
           "Output type is determined by the specified 'values_*' attribute."},
      })};
}

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
