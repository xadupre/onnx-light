// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml.h"
#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::traditionalml {

namespace {

std::vector<TensorType> ArrayFeatureExtractorTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32,
          TensorType::kString};
}

std::vector<TensorType> BinarizerTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
}

std::vector<TensorType> CastMapInputTypes() {
  return {TensorType::kMapInt64String, TensorType::kMapInt64Float};
}

std::vector<TensorType> CastMapOutputTypes() {
  return {TensorType::kString, TensorType::kFloat, TensorType::kInt64};
}

std::vector<TensorType> CategoryMapperInputTypes() {
  return {TensorType::kString, TensorType::kInt64};
}

std::vector<TensorType> CategoryMapperOutputTypes() {
  return {TensorType::kString, TensorType::kInt64};
}

std::vector<TensorType> ImputerTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
}

std::vector<TensorType> LabelEncoderTypes() {
  return {
      TensorType::kString, TensorType::kInt64, TensorType::kFloat,
      TensorType::kInt32,  TensorType::kInt16, TensorType::kDouble,
  };
}

std::vector<TensorType> NormalizerTypes() {
  return {TensorType::kFloat, TensorType::kDouble, TensorType::kInt64, TensorType::kInt32};
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

std::vector<TensorType> DictVectorizerInputTypes() {
  return {TensorType::kMapStringInt64, TensorType::kMapInt64String, TensorType::kMapInt64Float,
          TensorType::kMapInt64Double, TensorType::kMapStringFloat, TensorType::kMapStringDouble};
}

std::vector<TensorType> DictVectorizerOutputTypes() {
  return {TensorType::kInt64, TensorType::kFloat, TensorType::kDouble, TensorType::kString};
}

std::vector<TensorType> FeatureVectorizerInputTypes() {
  return {TensorType::kInt32, TensorType::kInt64, TensorType::kFloat, TensorType::kDouble};
}

LightOpSchema MakeTreeEnsembleClassifierSchema(int since_version) {
  std::vector<AttributeParam> attrs = {
      AttributeParam{"nodes_treeids", "Tree id for each node.", AttributeType::INTS, false,
                     std::monostate{}},
      AttributeParam{"nodes_nodeids",
                     "Node id for each node. Ids may restart at zero for each tree.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"nodes_featureids", "Feature id for each node.", AttributeType::INTS, false,
                     std::monostate{}},
      AttributeParam{"nodes_values", "Thresholds to do the splitting on for each node.",
                     AttributeType::FLOATS, false, std::monostate{}},
      AttributeParam{"nodes_hitrates",
                     "Popularity of each node, used for performance and may be omitted.",
                     AttributeType::FLOATS, false, std::monostate{}},
      AttributeParam{"nodes_modes", "The node kind, that is, the comparison to make at the node.",
                     AttributeType::STRINGS, false, std::monostate{}},
      AttributeParam{"nodes_truenodeids", "Child node if expression is true.", AttributeType::INTS,
                     false, std::monostate{}},
      AttributeParam{"nodes_falsenodeids", "Child node if expression is false.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"nodes_missing_value_tracks_true",
                     "For each node, define what to do in the presence of a missing value.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"class_treeids", "The id of the tree that this node is in.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"class_nodeids", "node id that this weight is for.", AttributeType::INTS,
                     false, std::monostate{}},
      AttributeParam{"class_ids", "The index of the class list that each weight is for.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"class_weights", "The weight for the class in class_id.",
                     AttributeType::FLOATS, false, std::monostate{}},
      AttributeParam{"classlabels_strings",
                     "Class labels if using string labels. One and only one of the 'classlabels_*' "
                     "attributes must be defined.",
                     AttributeType::STRINGS, false, std::monostate{}},
      AttributeParam{
          "classlabels_int64s",
          "Class labels if using integer labels. One and only one of the 'classlabels_*' "
          "attributes must be defined.",
          AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"post_transform", "Indicates the transform to apply to the score.",
                     AttributeType::STRING, false, std::string("NONE")},
      AttributeParam{"base_values", "Base values for classification, added to final class score.",
                     AttributeType::FLOATS, false, std::monostate{}},
  };
  if (since_version >= 3) {
    attrs.push_back(AttributeParam{"nodes_values_as_tensor",
                                   "Thresholds to do the splitting on for each node.",
                                   AttributeType::TENSOR, false, std::monostate{}});
    attrs.push_back(AttributeParam{"nodes_hitrates_as_tensor",
                                   "Popularity of each node, used for performance and may be "
                                   "omitted.",
                                   AttributeType::TENSOR, false, std::monostate{}});
    attrs.push_back(AttributeParam{"class_weights_as_tensor",
                                   "The weight for the class in class_id.", AttributeType::TENSOR,
                                   false, std::monostate{}});
    attrs.push_back(AttributeParam{"base_values_as_tensor",
                                   "Base values for classification, added to final class score.",
                                   AttributeType::TENSOR, false, std::monostate{}});
  }
  LightOpSchema schema(
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
      },
      std::move(attrs));
  if (since_version == 5) {
    schema.set_deprecated(true);
  }
  return schema;
}

LightOpSchema MakeTreeEnsembleRegressorSchema(int since_version) {
  std::vector<AttributeParam> attrs = {
      AttributeParam{"nodes_treeids", "Tree id for each node.", AttributeType::INTS, false,
                     std::monostate{}},
      AttributeParam{"nodes_nodeids",
                     "Node id for each node. Node ids must restart at zero for each tree and "
                     "increase sequentially.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"nodes_featureids", "Feature id for each node.", AttributeType::INTS, false,
                     std::monostate{}},
      AttributeParam{"nodes_values", "Thresholds to do the splitting on for each node.",
                     AttributeType::FLOATS, false, std::monostate{}},
      AttributeParam{"nodes_hitrates",
                     "Popularity of each node, used for performance and may be omitted.",
                     AttributeType::FLOATS, false, std::monostate{}},
      AttributeParam{"nodes_modes", "The node kind, that is, the comparison to make at the node.",
                     AttributeType::STRINGS, false, std::monostate{}},
      AttributeParam{"nodes_truenodeids", "Child node if expression is true.", AttributeType::INTS,
                     false, std::monostate{}},
      AttributeParam{"nodes_falsenodeids", "Child node if expression is false.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"nodes_missing_value_tracks_true",
                     "For each node, define what to do in the presence of a NaN.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"target_treeids", "The id of the tree that each node is in.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"target_nodeids", "The node id of each weight.", AttributeType::INTS, false,
                     std::monostate{}},
      AttributeParam{"target_ids", "The index of the target that each weight is for.",
                     AttributeType::INTS, false, std::monostate{}},
      AttributeParam{"target_weights", "The weight for each target.", AttributeType::FLOATS, false,
                     std::monostate{}},
      AttributeParam{"n_targets", "The total number of targets.", AttributeType::INT, false,
                     std::monostate{}},
      AttributeParam{"post_transform", "Indicates the transform to apply to the score.",
                     AttributeType::STRING, false, std::string("NONE")},
      AttributeParam{"aggregate_function", "Defines how to aggregate leaf values within a target.",
                     AttributeType::STRING, false, std::string("SUM")},
      AttributeParam{"base_values",
                     "Base values for regression, added to final prediction after applying "
                     "aggregate_function.",
                     AttributeType::FLOATS, false, std::monostate{}},
  };
  if (since_version >= 3) {
    attrs.push_back(AttributeParam{"nodes_values_as_tensor",
                                   "Thresholds to do the splitting on for each node.",
                                   AttributeType::TENSOR, false, std::monostate{}});
    attrs.push_back(AttributeParam{"nodes_hitrates_as_tensor",
                                   "Popularity of each node, used for performance and may be "
                                   "omitted.",
                                   AttributeType::TENSOR, false, std::monostate{}});
    attrs.push_back(AttributeParam{"target_weights_as_tensor", "The weight for each target.",
                                   AttributeType::TENSOR, false, std::monostate{}});
    attrs.push_back(AttributeParam{"base_values_as_tensor",
                                   "Base values for regression, added to final prediction.",
                                   AttributeType::TENSOR, false, std::monostate{}});
  }
  LightOpSchema schema("TreeEnsembleRegressor", "ai.onnx.ml", since_version,
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
                       },
                       std::move(attrs));
  if (since_version == 5) {
    schema.set_deprecated(true);
  }
  return schema;
}

std::vector<TensorType> LinearClassifierLabelTypes() {
  return {TensorType::kString, TensorType::kInt64};
}

LightOpSchema MakeLinearClassifierSchema() {
  return LightOpSchema(
      "LinearClassifier", "ai.onnx.ml", 1, MakeLinearClassifierDoc(),
      {
          {"X", "Data to be classified.", "T1"},
      },
      {
          {"Y", "Classification outputs (one class per example).", "T2"},
          {"Z", "Classification scores ([N,E] - one score for each class and example",
           "tensor(float)"},
      },
      {
          {"T1", TreeEnsembleClassicNumericTypes(),
           "The input must be a tensor of a numeric type, and of shape [N,C] or [C]. In the "
           "latter case, it will be treated as [1,C]"},
          {"T2", LinearClassifierLabelTypes(),
           "The output will be a tensor of strings or integers."},
      },
      {
          AttributeParam{"coefficients", "A collection of weights of the model(s).",
                         AttributeType::FLOATS, true, std::monostate{}},
          AttributeParam{"intercepts", "A collection of intercepts.", AttributeType::FLOATS, false,
                         std::monostate{}},
          AttributeParam{"multi_class",
                         "Indicates whether to do OvR or multinomial (0=OvR is the default).",
                         AttributeType::INT, false, static_cast<int64_t>(0)},
          AttributeParam{"classlabels_strings",
                         "Class labels when using string labels. One and only one 'classlabels' "
                         "attribute must be defined.",
                         AttributeType::STRINGS, false, std::monostate{}},
          AttributeParam{"classlabels_ints",
                         "Class labels when using integer labels. One and only one 'classlabels' "
                         "attribute must be defined.",
                         AttributeType::INTS, false, std::monostate{}},
          AttributeParam{"post_transform", "Indicates the transform to apply to the scores vector.",
                         AttributeType::STRING, false, std::string("NONE")},
      });
}

LightOpSchema MakeLinearRegressorSchema() {
  return LightOpSchema(
      "LinearRegressor", "ai.onnx.ml", 1, MakeLinearRegressorDoc(),
      {
          {"X", "Data to be regressed.", "T"},
      },
      {
          {"Y", "Regression outputs (one per target, per example).", "tensor(float)"},
      },
      {
          {"T", TreeEnsembleClassicNumericTypes(), "The input must be a tensor of a numeric type."},
      },
      {
          AttributeParam{"post_transform",
                         "Indicates the transform to apply to the regression output vector.",
                         AttributeType::STRING, false, std::string("NONE")},
          AttributeParam{"coefficients", "Weights of the model(s).", AttributeType::FLOATS, false,
                         std::monostate{}},
          AttributeParam{"intercepts", "Weights of the intercepts, if used.", AttributeType::FLOATS,
                         false, std::monostate{}},
          AttributeParam{"targets", "The total number of regression targets, 1 if not defined.",
                         AttributeType::INT, false, static_cast<int64_t>(1)},
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
      },
      {
          AttributeParam{"kernel_type",
                         "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
                         AttributeType::STRING, false, std::string("LINEAR")},
          AttributeParam{"kernel_params",
                         "List of 3 elements containing gamma, coef0, and degree, in that order. "
                         "Zero if unused for the kernel.",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"vectors_per_class", "Number of support vectors per class.",
                         AttributeType::INTS, false, std::monostate{}},
          AttributeParam{"support_vectors", "Chosen support vectors (all classes concatenated).",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"coefficients",
                         "Dual coefficients of the support vector in the decision function.",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"prob_a", "First set of probability coefficients.", AttributeType::FLOATS,
                         false, std::monostate{}},
          AttributeParam{"prob_b",
                         "Second set of probability coefficients. This array must be same size as "
                         "prob_a.",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"rho", "Intercept bias terms for the decision function.",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"post_transform", "Indicates the transform to apply to the score.",
                         AttributeType::STRING, false, std::string("NONE")},
          AttributeParam{"classlabels_strings",
                         "Class labels if using string labels. One and only one of the "
                         "'classlabels_*' attributes must be defined.",
                         AttributeType::STRINGS, false, std::monostate{}},
          AttributeParam{"classlabels_ints",
                         "Class labels if using integer labels. One and only one of the "
                         "'classlabels_*' attributes must be defined.",
                         AttributeType::INTS, false, std::monostate{}},
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
      },
      {
          AttributeParam{"kernel_type",
                         "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
                         AttributeType::STRING, false, std::string("LINEAR")},
          AttributeParam{"kernel_params",
                         "List of 3 elements containing gamma, coef0, and degree, in that order. "
                         "Zero if unused for the kernel.",
                         AttributeType::FLOATS, false, std::monostate{}},
          AttributeParam{"support_vectors", "Chosen support vectors", AttributeType::FLOATS, false,
                         std::monostate{}},
          AttributeParam{"one_class",
                         "Flag indicating whether the regression is a one-class SVM or not.",
                         AttributeType::INT, false, static_cast<int64_t>(0)},
          AttributeParam{"coefficients", "Support vector coefficients.", AttributeType::FLOATS,
                         false, std::monostate{}},
          AttributeParam{"n_supports", "The number of support vectors.", AttributeType::INT, false,
                         static_cast<int64_t>(0)},
          AttributeParam{"post_transform", "Indicates the transform to apply to the score.",
                         AttributeType::STRING, false, std::string("NONE")},
          AttributeParam{"rho", "Intercept bias terms for the decision function.",
                         AttributeType::FLOATS, false, std::monostate{}},
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
             },
             {
                 AttributeParam{"threshold",
                                "Values greater than this are mapped to 1, others to 0.",
                                AttributeType::FLOAT, false, 0.0},
             })};
       }},
      {"CategoryMapper",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "CategoryMapper", "ai.onnx.ml", 1, MakeCategoryMapperDoc(),
             {
                 {"X", "Input data", "T1"},
             },
             {
                 {"Y",
                  "Output data. If strings are input, the output values are integers, and vice "
                  "versa.",
                  "T2"},
             },
             {
                 {"T1", CategoryMapperInputTypes(),
                  "The input must be a tensor of strings or integers, either [N,C] or [C]."},
                 {"T2", CategoryMapperOutputTypes(),
                  "The output is a tensor of strings or integers. Its shape will be the same "
                  "as the input shape."},
             },
             {
                 AttributeParam{"cats_strings",
                                "The strings of the map. This sequence must be the same length as "
                                "the 'cats_int64s' sequence",
                                AttributeType::STRINGS, false, std::monostate{}},
                 AttributeParam{"cats_int64s",
                                "The integers of the map. This sequence must be the same length as "
                                "the 'cats_strings' sequence.",
                                AttributeType::INTS, false, std::monostate{}},
                 AttributeParam{
                     "default_string",
                     "A string to use when an input integer value is not found in the map.",
                     AttributeType::STRING, false, std::string("_Unused")},
                 AttributeParam{
                     "default_int64",
                     "An integer to use when an input string value is not found in the map.",
                     AttributeType::INT, false, static_cast<int64_t>(-1)},
             })};
       }},
      {"CastMap",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "CastMap", "ai.onnx.ml", 1, MakeCastMapDoc(),
             {
                 {"X", "The input map that is to be cast to a tensor", "T1"},
             },
             {
                 {"Y",
                  "A tensor representing the same data as the input map, ordered by their keys",
                  "T2"},
             },
             {
                 {"T1", CastMapInputTypes(),
                  "The input must be an integer map to either string or float."},
                 {"T2", CastMapOutputTypes(),
                  "The output is a 1-D tensor of string, float, or integer."},
             },
             {
                 AttributeParam{
                     "cast_to",
                     "A string indicating the desired element type of the output tensor, "
                     "one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'.",
                     AttributeType::STRING, false, std::string("TO_FLOAT")},
                 AttributeParam{
                     "map_form",
                     "Indicates whether to only output as many values as are in the input "
                     "(dense), or position the input based on using the key of the map as "
                     "the index of the output (sparse).",
                     AttributeType::STRING, false, std::string("DENSE")},
                 AttributeParam{
                     "max_map",
                     "If the value of map_form is 'SPARSE,' this attribute indicates the "
                     "total length of the output tensor.",
                     AttributeType::INT, false, static_cast<int64_t>(1)},
             })};
       }},
      {"DictVectorizer",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "DictVectorizer", "ai.onnx.ml", 1, MakeDictVectorizerDoc(),
             {
                 {"X", "A dictionary.", "T1"},
             },
             {
                 {"Y", "A 1-D tensor holding values from the input dictionary.", "T2"},
             },
             {
                 {"T1", DictVectorizerInputTypes(),
                  "The input must be a map from strings or integers to either strings or a "
                  "numeric type. The key and value types cannot be the same."},
                 {"T2", DictVectorizerOutputTypes(),
                  "The output will be a tensor of the value type of the input map. It's shape "
                  "will be [1,C], where C is the length of the input dictionary."},
             },
             {
                 AttributeParam{"string_vocabulary",
                                "A string vocabulary array. One and only one of the vocabularies "
                                "must be defined.",
                                AttributeType::STRINGS, false, std::monostate{}},
                 AttributeParam{"int64_vocabulary",
                                "An integer vocabulary array. One and only one of the vocabularies "
                                "must be defined.",
                                AttributeType::INTS, false, std::monostate{}},
             })};
       }},
      {"FeatureVectorizer",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "FeatureVectorizer", "ai.onnx.ml", 1, MakeFeatureVectorizerDoc(),
             {
                 {"X", "An ordered collection of tensors, all with the same element type.", "T1"},
             },
             {
                 {"Y", "The output array, elements ordered as the inputs.", "tensor(float)"},
             },
             {
                 {"T1", FeatureVectorizerInputTypes(),
                  "The input type must be a tensor of a numeric type."},
             },
             {
                 AttributeParam{"inputdimensions", "The size of each input in the input list",
                                AttributeType::INTS, false, std::monostate{}},
             })};
       }},
      {"Imputer",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "Imputer", "ai.onnx.ml", 1, MakeImputerDoc(),
             {
                 {"X", "Data to be processed.", "T"},
             },
             {
                 {"Y", "Imputed output data", "T"},
             },
             {
                 {"T", ImputerTypes(),
                  "The input type must be a tensor of a numeric type, either [N,C] or [C]. "
                  "The output type will be of the same tensor type and shape."},
             },
             {
                 AttributeParam{"imputed_value_floats", "Value(s) to change to",
                                AttributeType::FLOATS, false, std::monostate{}},
                 AttributeParam{"replaced_value_float", "A value that needs replacing.",
                                AttributeType::FLOAT, false, 0.0},
                 AttributeParam{"imputed_value_int64s", "Value(s) to change to.",
                                AttributeType::INTS, false, std::monostate{}},
                 AttributeParam{"replaced_value_int64", "A value that needs replacing.",
                                AttributeType::INT, false, static_cast<int64_t>(0)},
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
             },
             {
                 AttributeParam{"keys_tensor",
                                "Keys encoded as a 1D tensor. One and only one of 'keys_*'s should "
                                "be set.",
                                AttributeType::TENSOR, false, std::monostate{}},
                 AttributeParam{"keys_strings", "A list of strings.", AttributeType::STRINGS, false,
                                std::monostate{}},
                 AttributeParam{"keys_int64s", "A list of ints.", AttributeType::INTS, false,
                                std::monostate{}},
                 AttributeParam{"keys_floats", "A list of floats.", AttributeType::FLOATS, false,
                                std::monostate{}},
                 AttributeParam{"values_tensor",
                                "Values encoded as a 1D tensor. One and only one of 'values_*'s "
                                "should be set.",
                                AttributeType::TENSOR, false, std::monostate{}},
                 AttributeParam{"values_strings", "A list of strings.", AttributeType::STRINGS,
                                false, std::monostate{}},
                 AttributeParam{"values_int64s", "A list of ints.", AttributeType::INTS, false,
                                std::monostate{}},
                 AttributeParam{"values_floats", "A list of floats.", AttributeType::FLOATS, false,
                                std::monostate{}},
                 AttributeParam{"default_string", "A string.", AttributeType::STRING, false,
                                std::string("_Unused")},
                 AttributeParam{"default_int64", "An integer.", AttributeType::INT, false,
                                static_cast<int64_t>(-1)},
                 AttributeParam{"default_float", "A float.", AttributeType::FLOAT, false, -0.0},
                 AttributeParam{"default_tensor",
                                "A default tensor. {\"_Unused\"} if values_* has string type, {-1} "
                                "if values_* has integral type, and {-0.f} if values_* has float "
                                "type.",
                                AttributeType::TENSOR, false, std::monostate{}},
             })};
       }},
      {"LinearClassifier", [] { return std::vector<LightOpSchema>{MakeLinearClassifierSchema()}; }},
      {"LinearRegressor", [] { return std::vector<LightOpSchema>{MakeLinearRegressorSchema()}; }},
      {"Normalizer",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "Normalizer", "ai.onnx.ml", 1, MakeNormalizerDoc(),
             {
                 {"X", "Data to be encoded, a tensor of shape [N,C] or [C]", "T"},
             },
             {
                 {"Y", "Encoded output data", "tensor(float)"},
             },
             {
                 {"T", NormalizerTypes(), "The input must be a tensor of a numeric type."},
             },
             {
                 AttributeParam{"norm", "One of 'MAX,' 'L1,' 'L2'", AttributeType::STRING, false,
                                std::string("MAX")},
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
             },
             {
                 AttributeParam{"cats_int64s",
                                "List of categories, ints. One and only one of the 'cats_*' "
                                "attributes must be defined.",
                                AttributeType::INTS, false, std::monostate{}},
                 AttributeParam{"cats_strings",
                                "List of categories, strings. One and only one of the 'cats_*' "
                                "attributes must be defined.",
                                AttributeType::STRINGS, false, std::monostate{}},
                 AttributeParam{"zeros",
                                "If true and category is not present, will return all zeros; if "
                                "false and a category if not found, the operator will fail.",
                                AttributeType::INT, false, static_cast<int64_t>(1)},
             })};
       }},
      {"SVMClassifier", [] { return std::vector<LightOpSchema>{MakeSVMClassifierSchema()}; }},
      {"SVMRegressor", [] { return std::vector<LightOpSchema>{MakeSVMRegressorSchema()}; }},
      {"Scaler",
       [] {
         return std::vector<LightOpSchema>{
             LightOpSchema("Scaler", "ai.onnx.ml", 1, MakeScalerDoc(),
                           {
                               {"X", "Data to be scaled.", "T"},
                           },
                           {
                               {"Y", "Scaled output data.", "tensor(float)"},
                           },
                           {
                               {"T", TreeEnsembleClassicNumericTypes(),
                                "The input must be a tensor of a numeric type."},
                           },
                           {
                               AttributeParam{"offset",
                                              "First, offset by this. Can be length of features in "
                                              "an [N,F] tensor or length 1.",
                                              AttributeType::FLOATS, false, std::monostate{}},
                               AttributeParam{"scale",
                                              "Second, multiply by this. Can be length of features "
                                              "in an [N,F] tensor or length 1.",
                                              AttributeType::FLOATS, false, std::monostate{}},
                           })};
       }},
      {"TreeEnsemble",
       [] {
         return std::vector<LightOpSchema>{LightOpSchema(
             "TreeEnsemble", "ai.onnx.ml", 5, MakeTreeEnsembleDoc(),
             {
                 {"X", "Input of shape [Batch Size, Number of Features]", "T"},
             },
             {
                 {"Y", "Output of shape [Batch Size, Number of targets]", "T"},
             },
             {
                 {"T", TreeEnsembleFloatTypes(),
                  "The input type must be a tensor of a numeric type."},
             },
             {
                 AttributeParam{"nodes_featureids", "Feature id for each node.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"nodes_splits",
                                "Thresholds to do the splitting on for each node "
                                "with mode that is not 'BRANCH_MEMBER'.",
                                AttributeType::TENSOR, true, std::monostate{}},
                 AttributeParam{"nodes_hitrates",
                                "Popularity of each node, used for performance and "
                                "may be omitted.",
                                AttributeType::TENSOR, false, std::monostate{}},
                 AttributeParam{"nodes_modes", "The comparison operation performed by the node.",
                                AttributeType::TENSOR, true, std::monostate{}},
                 AttributeParam{"nodes_truenodeids",
                                "If nodes_trueleafs is false at an entry, this "
                                "represents the position of the true branch node.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"nodes_falsenodeids",
                                "If nodes_falseleafs is false at an entry, this "
                                "represents the position of the false branch node.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"nodes_trueleafs",
                                "1 if true branch is leaf for each node and 0 an "
                                "interior node.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"nodes_falseleafs",
                                "1 if false branch is leaf for each node and 0 if "
                                "an interior node.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"nodes_missing_value_tracks_true",
                                "For each node, define whether to follow the true "
                                "branch in the presence of a NaN input feature.",
                                AttributeType::INTS, false, std::monostate{}},
                 AttributeParam{"tree_roots", "Index into nodes_* for the root of each tree.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"membership_values",
                                "Members to test membership of for each set "
                                "membership node.",
                                AttributeType::TENSOR, false, std::monostate{}},
                 AttributeParam{"leaf_targetids",
                                "The index of the target that this leaf contributes "
                                "to.",
                                AttributeType::INTS, true, std::monostate{}},
                 AttributeParam{"leaf_weights", "The weight for each leaf.", AttributeType::TENSOR,
                                true, std::monostate{}},
                 AttributeParam{"n_targets", "The total number of targets.", AttributeType::INT,
                                false, std::monostate{}},
                 AttributeParam{"post_transform", "Indicates the transform to apply to the score.",
                                AttributeType::INT, false, static_cast<int64_t>(0)},
                 AttributeParam{"aggregate_function",
                                "Defines how to aggregate leaf values within a "
                                "target.",
                                AttributeType::INT, false, static_cast<int64_t>(1)},
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
             },
             {
                 AttributeParam{"classlabels_strings",
                                "The keys when using string keys. One and only one of the "
                                "'classlabels_*' attributes must be defined.",
                                AttributeType::STRINGS, false, std::monostate{}},
                 AttributeParam{"classlabels_int64s",
                                "The keys when using int keys. One and only one of the "
                                "'classlabels_*' attributes must be defined.",
                                AttributeType::INTS, false, std::monostate{}},
             })};
       }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::traditionalml
