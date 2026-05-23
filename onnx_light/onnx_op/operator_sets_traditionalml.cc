// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml.h"
#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

namespace {

std::vector<TensorType> LabelEncoderTypes() {
  return {
      TensorType::kString, TensorType::kInt64, TensorType::kFloat,
      TensorType::kInt32,  TensorType::kInt16, TensorType::kDouble,
  };
}

std::vector<TensorType> TreeEnsembleClassicNumericTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
}

std::vector<TensorType> TreeEnsembleClassifierLabelTypes() {
  return {TensorType::kString, TensorType::kInt64};
}

std::vector<TensorType> TreeEnsembleFloatTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16};
}

LightOpSchema MakeTreeEnsembleClassifierSchema(int since_version) {
  return LightOpSchema(
      "TreeEnsembleClassifier", "ai.onnx.ml", since_version,
      MakeTreeEnsembleClassifierDoc(since_version),
      {
          {"X", "Input of shape [N,F]", "T1"},
      },
      {
          {"Y", "N, Top class for each point", "T2"},
          {"Z", "The class score for each class, for each point, a tensor of shape [N,E].",
           "tensor(float)"},
      },
      {
          {"T1", TreeEnsembleClassicNumericTypes(),
           "The input type must be a tensor of a numeric type."},
          {"T2", TreeEnsembleClassifierLabelTypes(),
           "The output type will be a tensor of strings or integers, depending on which of the "
           "classlabels_* attributes is used."},
      });
}

LightOpSchema MakeTreeEnsembleRegressorSchema(int since_version) {
  return LightOpSchema("TreeEnsembleRegressor", "ai.onnx.ml", since_version,
                       MakeTreeEnsembleRegressorDoc(since_version),
                       {
                           {"X", "Input of shape [N,F]", "T"},
                       },
                       {
                           {"Y", "N classes", "tensor(float)"},
                       },
                       {
                           {"T", TreeEnsembleClassicNumericTypes(),
                            "The input type must be a tensor of a numeric type."},
                       });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTraditionalMLSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas{
      LightOpSchema(
          "LabelEncoder", "ai.onnx.ml", 4, MakeLabelEncoderDoc(),
          {
              {"X", "Input data. It must have the same element type as the keys_* attribute set.",
               "T1"},
          },
          {
              {"Y",
               "Output data. This tensor's element type is based on the values_* attribute set.",
               "T2"},
          },
          {
              {"T1", LabelEncoderTypes(), "The input type is a tensor of any shape."},
              {"T2", LabelEncoderTypes(),
               "Output type is determined by the specified 'values_*' attribute."},
          }),
      LightOpSchema(
          "TreeEnsemble", "ai.onnx.ml", 5, MakeTreeEnsembleDoc(),
          {
              {"X", "Input of shape [Batch Size, Number of Features]", "T"},
          },
          {
              {"Y", "Output of shape [Batch Size, Number of targets]", "T"},
          },
          {
              {"T", TreeEnsembleFloatTypes(), "The input type must be a tensor of a numeric type."},
          }),
      MakeTreeEnsembleClassifierSchema(5),
      MakeTreeEnsembleClassifierSchema(3),
      MakeTreeEnsembleClassifierSchema(1),
      MakeTreeEnsembleRegressorSchema(5),
      MakeTreeEnsembleRegressorSchema(3),
      MakeTreeEnsembleRegressorSchema(1),
      LightOpSchema("ZipMap", "ai.onnx.ml", 1, MakeZipMapDoc(),
                    {
                        {"X", "The input values", "tensor(float)"},
                    },
                    {
                        {"Z", "The output map", "T"},
                    },
                    {
                        {"T",
                         {TensorType::kSeqMapStringFloat, TensorType::kSeqMapInt64Float},
                         "The output will be a sequence of string or integer maps to float."},
                    }),
  };
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
