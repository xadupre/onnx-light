// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml.h"
#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

namespace {

std::vector<TensorType> ArrayFeatureExtractorTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32,
          TensorType::kString};
}

std::vector<TensorType> BinarizerTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
}

std::vector<TensorType> ScalerTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
}

std::vector<TensorType> LabelEncoderTypes() {
  return {
      TensorType::kString, TensorType::kInt64, TensorType::kFloat,
      TensorType::kInt32,  TensorType::kInt16, TensorType::kDouble,
  };
}

std::vector<TensorType> OneHotEncoderTypes() {
  return {TensorType::kString, TensorType::kInt64, TensorType::kInt32, TensorType::kFloat,
          TensorType::kDouble};
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

LightOpSchema MakeSVMClassifierSchema() {
  return LightOpSchema(
      "SVMClassifier", "ai.onnx.ml", 1, MakeSVMClassifierDoc(),
      {
          {"X", "Data to be classified.", "T1"},
      },
      {
          {"Y", "Classification outputs (one class per example).", "T2"},
          {"Z",
           "Class scores (one per class per example), if prob_a and prob_b are provided they are "
           "probabilities for each class, otherwise they are raw scores.",
           "tensor(float)"},
      },
      {
          {"T1", TreeEnsembleClassicNumericTypes(),
           "The input must be a tensor of a numeric type, either [C] or [N,C]."},
          {"T2", TreeEnsembleClassifierLabelTypes(),
           "The output type will be a tensor of strings or integers, depending on which of the "
           "classlabels_* attributes is used. Its size will match the batch size of the input."},
      });
}

LightOpSchema MakeSVMRegressorSchema() {
  return LightOpSchema(
      "SVMRegressor", "ai.onnx.ml", 1, MakeSVMRegressorDoc(),
      {
          {"X", "Data to be regressed.", "T"},
      },
      {
          {"Y", "Regression outputs (one score per target per example).", "tensor(float)"},
      },
      {
          {"T", TreeEnsembleClassicNumericTypes(),
           "The input type must be a tensor of a numeric type, either [C] or [N,C]."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpTraditionalMLSchemasWithHistory(const std::string &op_type,
                                                                       bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"ArrayFeatureExtractor",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "ArrayFeatureExtractor", "ai.onnx.ml", 1, MakeArrayFeatureExtractorDoc(),
             {
                 {"X", "Data to be selected", "T"},
                 {"Y", "The indices, based on 0 as the first index of any dimension.",
                  "tensor(int64)"},
             },
             {
                 {"Z", "Selected output data as an array", "T"},
             },
             {
                 {"T", ArrayFeatureExtractorTypes(),
                  "The input must be a tensor of a numeric type or string. The output will be "
                  "of the same tensor type."},
             })};
       }},
      {"Binarizer",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "Binarizer", "ai.onnx.ml", 1, MakeBinarizerDoc(),
             {
                 {"X", "Data to be binarized", "T"},
             },
             {
                 {"Y", "Binarized output data", "T"},
             },
             {
                 {"T", BinarizerTypes(),
                  "The input must be a tensor of a numeric type. The output will be of the "
                  "same tensor type."},
             })};
       }},
      {"LabelEncoder",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "LabelEncoder", "ai.onnx.ml", 4, MakeLabelEncoderDoc(),
             {
                 {"X",
                  "Input data. It must have the same element type as the keys_* attribute set.",
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
             })};
       }},
      {"OneHotEncoder",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "OneHotEncoder", "ai.onnx.ml", 1, MakeOneHotEncoderDoc(),
             {
                 {"X", "Data to be encoded.", "T"},
             },
             {
                 {"Y", "Encoded output data, having one more dimension than X.", "tensor(float)"},
             },
             {
                 {"T", OneHotEncoderTypes(), "The input must be a tensor of a numeric type."},
             })};
       }},
      {"SVMClassifier", [] { return std::vector<LightOpSchema>{MakeSVMClassifierSchema()}; }},
      {"SVMRegressor", [] { return std::vector<LightOpSchema>{MakeSVMRegressorSchema()}; }},
      {"Scaler",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "Scaler", "ai.onnx.ml", 1, MakeScalerDoc(),
             {
                 {"X", "Data to be scaled.", "T"},
             },
             {
                 {"Y", "Scaled output data.", "tensor(float)"},
             },
             {
                 {"T", ScalerTypes(), "The input must be a tensor of a numeric type."},
             })};
       }},
      {"TreeEnsemble",
       [] {
         return std::vector<LightOpSchema>{
             LightOpSchema("TreeEnsemble", "ai.onnx.ml", 5, MakeTreeEnsembleDoc(),
                           {
                               {"X", "Input of shape [Batch Size, Number of Features]", "T"},
                           },
                           {
                               {"Y", "Output of shape [Batch Size, Number of targets]", "T"},
                           },
                           {
                               {"T", TreeEnsembleFloatTypes(),
                                "The input type must be a tensor of a numeric type."},
                           })};
       }},
      {"TreeEnsembleClassifier",
       [] {
         return std::vector<LightOpSchema>{
             MakeTreeEnsembleClassifierSchema(5),
             MakeTreeEnsembleClassifierSchema(3),
             MakeTreeEnsembleClassifierSchema(1),
         };
       }},
      {"TreeEnsembleRegressor",
       [] {
         return std::vector<LightOpSchema>{
             MakeTreeEnsembleRegressorSchema(5),
             MakeTreeEnsembleRegressorSchema(3),
             MakeTreeEnsembleRegressorSchema(1),
         };
       }},
      {"ZipMap",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "ZipMap", "ai.onnx.ml", 1, MakeZipMapDoc(),
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
             })};
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
