// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsArrayFeatureExtractorAndLabelEncoderSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> array_feature_extractor_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("ArrayFeatureExtractor");
  const std::vector<onnx_op::LightOpSchema> binarizer_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("Binarizer");
  const std::vector<onnx_op::LightOpSchema> label_encoder_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("LabelEncoder");

  EXPECT_EQ(schemas.size(), 19u);

  const onnx_op::LightOpSchema *const array_feature_extractor_v1 =
      FindByVersion(array_feature_extractor_schemas, 1);
  ASSERT_NE(nullptr, array_feature_extractor_v1);
  EXPECT_EQ(array_feature_extractor_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(array_feature_extractor_v1->inputs().size(), 2u);
  EXPECT_EQ(array_feature_extractor_v1->outputs().size(), 1u);
  EXPECT_EQ(array_feature_extractor_v1->inputs()[0].name, "X");
  EXPECT_EQ(array_feature_extractor_v1->inputs()[0].type, "T");
  EXPECT_EQ(array_feature_extractor_v1->inputs()[1].name, "Y");
  EXPECT_EQ(array_feature_extractor_v1->inputs()[1].type, "tensor(int64)");
  EXPECT_EQ(array_feature_extractor_v1->outputs()[0].name, "Z");
  EXPECT_EQ(array_feature_extractor_v1->outputs()[0].type, "T");
  EXPECT_EQ(array_feature_extractor_v1->type_constraints().size(), 1u);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs.size(), 5u);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kDouble);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kInt64);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kInt32);
  EXPECT_EQ(array_feature_extractor_v1->type_constraints()[0].allowed_type_strs[4],
            onnx_op::TensorType::kString);

  const onnx_op::LightOpSchema *const binarizer_v1 = FindByVersion(binarizer_schemas, 1);
  ASSERT_NE(nullptr, binarizer_v1);
  EXPECT_EQ(binarizer_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(binarizer_v1->inputs().size(), 1u);
  EXPECT_EQ(binarizer_v1->outputs().size(), 1u);
  EXPECT_EQ(binarizer_v1->inputs()[0].name, "X");
  EXPECT_EQ(binarizer_v1->inputs()[0].type, "T");
  EXPECT_EQ(binarizer_v1->outputs()[0].name, "Y");
  EXPECT_EQ(binarizer_v1->outputs()[0].type, "T");
  EXPECT_EQ(binarizer_v1->type_constraints().size(), 1u);
  EXPECT_EQ(binarizer_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(binarizer_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(binarizer_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(binarizer_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kDouble);
  EXPECT_EQ(binarizer_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kInt64);
  EXPECT_EQ(binarizer_v1->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kInt32);

  const onnx_op::LightOpSchema *const label_encoder_v4 = FindByVersion(label_encoder_schemas, 4);
  ASSERT_NE(nullptr, label_encoder_v4);
  EXPECT_EQ(label_encoder_v4->domain(), "ai.onnx.ml");
  EXPECT_EQ(label_encoder_v4->inputs().size(), 1u);
  EXPECT_EQ(label_encoder_v4->outputs().size(), 1u);
  EXPECT_EQ(label_encoder_v4->type_constraints().size(), 2u);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs,
            label_encoder_v4->type_constraints()[1].allowed_type_strs);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs.size(), 6u);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kString);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kInt64);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kInt32);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[4],
            onnx_op::TensorType::kInt16);
  EXPECT_EQ(label_encoder_v4->type_constraints()[0].allowed_type_strs[5],
            onnx_op::TensorType::kDouble);
  EXPECT_EQ(label_encoder_v4->inputs()[0].name, "X");
  EXPECT_EQ(label_encoder_v4->outputs()[0].name, "Y");
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsZipMapSchemaWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> zip_map_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("ZipMap");

  const onnx_op::LightOpSchema *const zipmap_v1 = FindByVersion(zip_map_schemas, 1);
  ASSERT_NE(nullptr, zipmap_v1);
  EXPECT_EQ(zipmap_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(zipmap_v1->inputs().size(), 1u);
  EXPECT_EQ(zipmap_v1->outputs().size(), 1u);
  EXPECT_EQ(zipmap_v1->type_constraints().size(), 1u);
  EXPECT_EQ(zipmap_v1->inputs()[0].name, "X");
  EXPECT_EQ(zipmap_v1->inputs()[0].type, "tensor(float)");
  EXPECT_EQ(zipmap_v1->outputs()[0].name, "Z");
  EXPECT_EQ(zipmap_v1->outputs()[0].type, "T");
  EXPECT_EQ(zipmap_v1->type_constraints()[0].allowed_type_strs.size(), 2u);
  EXPECT_EQ(zipmap_v1->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kSeqMapStringFloat);
  EXPECT_EQ(zipmap_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kSeqMapInt64Float);
  EXPECT_STREQ(onnx_op::ToTypeString(zipmap_v1->type_constraints()[0].allowed_type_strs[0]),
               "seq(map(string, float))");
  EXPECT_STREQ(onnx_op::ToTypeString(zipmap_v1->type_constraints()[0].allowed_type_strs[1]),
               "seq(map(int64, float))");
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsTreeEnsembleSchema) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> tree_ensemble_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("TreeEnsemble");

  const onnx_op::LightOpSchema *const tree_ensemble_v5 = FindByVersion(tree_ensemble_schemas, 5);
  ASSERT_NE(nullptr, tree_ensemble_v5);
  EXPECT_EQ(tree_ensemble_v5->domain(), "ai.onnx.ml");
  EXPECT_EQ(tree_ensemble_v5->inputs().size(), 1u);
  EXPECT_EQ(tree_ensemble_v5->outputs().size(), 1u);
  EXPECT_EQ(tree_ensemble_v5->type_constraints().size(), 1u);
  EXPECT_EQ(tree_ensemble_v5->inputs()[0].name, "X");
  EXPECT_EQ(tree_ensemble_v5->inputs()[0].type, "T");
  EXPECT_EQ(tree_ensemble_v5->outputs()[0].name, "Y");
  EXPECT_EQ(tree_ensemble_v5->outputs()[0].type, "T");
  EXPECT_EQ(tree_ensemble_v5->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(tree_ensemble_v5->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(tree_ensemble_v5->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kDouble);
  EXPECT_EQ(tree_ensemble_v5->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kFloat16);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsTreeEnsembleClassifierSchemaHistory) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> tree_ensemble_classifier_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("TreeEnsembleClassifier");

  for (int version : {1, 3, 5}) {
    const onnx_op::LightOpSchema *const schema =
        FindByVersion(tree_ensemble_classifier_schemas, version);
    ASSERT_NE(nullptr, schema) << "version=" << version;
    EXPECT_EQ(schema->domain(), "ai.onnx.ml");
    EXPECT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->outputs().size(), 2u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    EXPECT_EQ(schema->outputs()[0].type, "T2");
    EXPECT_EQ(schema->outputs()[1].name, "Z");
    EXPECT_EQ(schema->outputs()[1].type, "tensor(float)");
    EXPECT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs.size(), 4u);
    EXPECT_EQ(schema->type_constraints()[1].type_param_str, "T2");
    EXPECT_EQ(schema->type_constraints()[1].allowed_type_strs.size(), 2u);
    EXPECT_EQ(schema->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kString);
    EXPECT_EQ(schema->type_constraints()[1].allowed_type_strs[1], onnx_op::TensorType::kInt64);
  }
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsTreeEnsembleRegressorSchemaHistory) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> tree_ensemble_regressor_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("TreeEnsembleRegressor");

  for (int version : {1, 3, 5}) {
    const onnx_op::LightOpSchema *const schema =
        FindByVersion(tree_ensemble_regressor_schemas, version);
    ASSERT_NE(nullptr, schema) << "version=" << version;
    EXPECT_EQ(schema->domain(), "ai.onnx.ml");
    EXPECT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[0].type, "T");
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    EXPECT_EQ(schema->outputs()[0].type, "tensor(float)");
    EXPECT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs.size(), 4u);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kDouble);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kInt64);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kInt32);
  }
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsOneHotEncoderSchema) {
  const std::vector<onnx_op::LightOpSchema> one_hot_encoder_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("OneHotEncoder");

  const onnx_op::LightOpSchema *const one_hot_encoder_v1 =
      FindByVersion(one_hot_encoder_schemas, 1);
  ASSERT_NE(nullptr, one_hot_encoder_v1);
  EXPECT_EQ(one_hot_encoder_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(one_hot_encoder_v1->inputs().size(), 1u);
  EXPECT_EQ(one_hot_encoder_v1->outputs().size(), 1u);
  EXPECT_EQ(one_hot_encoder_v1->inputs()[0].name, "X");
  EXPECT_EQ(one_hot_encoder_v1->inputs()[0].type, "T");
  EXPECT_EQ(one_hot_encoder_v1->outputs()[0].name, "Y");
  EXPECT_EQ(one_hot_encoder_v1->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(one_hot_encoder_v1->type_constraints().size(), 1u);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs.size(), 5u);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kString);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kInt64);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kInt32);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(one_hot_encoder_v1->type_constraints()[0].allowed_type_strs[4],
            onnx_op::TensorType::kDouble);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsSVMClassifierSchema) {
  const std::vector<onnx_op::LightOpSchema> svm_classifier_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("SVMClassifier");

  const onnx_op::LightOpSchema *const svm_classifier_v1 = FindByVersion(svm_classifier_schemas, 1);
  ASSERT_NE(nullptr, svm_classifier_v1);
  EXPECT_EQ(svm_classifier_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(svm_classifier_v1->inputs().size(), 1u);
  EXPECT_EQ(svm_classifier_v1->outputs().size(), 2u);
  EXPECT_EQ(svm_classifier_v1->inputs()[0].name, "X");
  EXPECT_EQ(svm_classifier_v1->inputs()[0].type, "T1");
  EXPECT_EQ(svm_classifier_v1->outputs()[0].name, "Y");
  EXPECT_EQ(svm_classifier_v1->outputs()[0].type, "T2");
  EXPECT_EQ(svm_classifier_v1->outputs()[1].name, "Z");
  EXPECT_EQ(svm_classifier_v1->outputs()[1].type, "tensor(float)");
  EXPECT_EQ(svm_classifier_v1->type_constraints().size(), 2u);
  EXPECT_EQ(svm_classifier_v1->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(svm_classifier_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(svm_classifier_v1->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(svm_classifier_v1->type_constraints()[1].allowed_type_strs.size(), 2u);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsSVMRegressorSchema) {
  const std::vector<onnx_op::LightOpSchema> svm_regressor_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("SVMRegressor");

  const onnx_op::LightOpSchema *const svm_regressor_v1 = FindByVersion(svm_regressor_schemas, 1);
  ASSERT_NE(nullptr, svm_regressor_v1);
  EXPECT_EQ(svm_regressor_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(svm_regressor_v1->inputs().size(), 1u);
  EXPECT_EQ(svm_regressor_v1->outputs().size(), 1u);
  EXPECT_EQ(svm_regressor_v1->inputs()[0].name, "X");
  EXPECT_EQ(svm_regressor_v1->inputs()[0].type, "T");
  EXPECT_EQ(svm_regressor_v1->outputs()[0].name, "Y");
  EXPECT_EQ(svm_regressor_v1->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(svm_regressor_v1->type_constraints().size(), 1u);
  EXPECT_EQ(svm_regressor_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(svm_regressor_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsScalerSchema) {
  const std::vector<onnx_op::LightOpSchema> scaler_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("Scaler");

  const onnx_op::LightOpSchema *const scaler_v1 = FindByVersion(scaler_schemas, 1);
  ASSERT_NE(nullptr, scaler_v1);
  EXPECT_EQ(scaler_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(scaler_v1->inputs().size(), 1u);
  EXPECT_EQ(scaler_v1->outputs().size(), 1u);
  EXPECT_EQ(scaler_v1->inputs()[0].name, "X");
  EXPECT_EQ(scaler_v1->inputs()[0].type, "T");
  EXPECT_EQ(scaler_v1->outputs()[0].name, "Y");
  EXPECT_EQ(scaler_v1->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(scaler_v1->type_constraints().size(), 1u);
  EXPECT_EQ(scaler_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(scaler_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(scaler_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(scaler_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kDouble);
  EXPECT_EQ(scaler_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kInt64);
  EXPECT_EQ(scaler_v1->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kInt32);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsLinearClassifierSchema) {
  const std::vector<onnx_op::LightOpSchema> linear_classifier_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("LinearClassifier");

  const onnx_op::LightOpSchema *const linear_classifier_v1 =
      FindByVersion(linear_classifier_schemas, 1);
  ASSERT_NE(nullptr, linear_classifier_v1);
  EXPECT_EQ(linear_classifier_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(linear_classifier_v1->inputs().size(), 1u);
  EXPECT_EQ(linear_classifier_v1->outputs().size(), 2u);
  EXPECT_EQ(linear_classifier_v1->inputs()[0].name, "X");
  EXPECT_EQ(linear_classifier_v1->inputs()[0].type, "T1");
  EXPECT_EQ(linear_classifier_v1->outputs()[0].name, "Y");
  EXPECT_EQ(linear_classifier_v1->outputs()[0].type, "T2");
  EXPECT_EQ(linear_classifier_v1->outputs()[1].name, "Z");
  EXPECT_EQ(linear_classifier_v1->outputs()[1].type, "tensor(float)");
  EXPECT_EQ(linear_classifier_v1->type_constraints().size(), 2u);
  EXPECT_EQ(linear_classifier_v1->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(linear_classifier_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(linear_classifier_v1->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(linear_classifier_v1->type_constraints()[1].allowed_type_strs.size(), 2u);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsLinearRegressorSchema) {
  const std::vector<onnx_op::LightOpSchema> linear_regressor_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("LinearRegressor");

  const onnx_op::LightOpSchema *const linear_regressor_v1 =
      FindByVersion(linear_regressor_schemas, 1);
  ASSERT_NE(nullptr, linear_regressor_v1);
  EXPECT_EQ(linear_regressor_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(linear_regressor_v1->inputs().size(), 1u);
  EXPECT_EQ(linear_regressor_v1->outputs().size(), 1u);
  EXPECT_EQ(linear_regressor_v1->inputs()[0].name, "X");
  EXPECT_EQ(linear_regressor_v1->inputs()[0].type, "T");
  EXPECT_EQ(linear_regressor_v1->outputs()[0].name, "Y");
  EXPECT_EQ(linear_regressor_v1->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(linear_regressor_v1->type_constraints().size(), 1u);
  EXPECT_EQ(linear_regressor_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(linear_regressor_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsImputerSchema) {
  const std::vector<onnx_op::LightOpSchema> imputer_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("Imputer");

  const onnx_op::LightOpSchema *const imputer_v1 = FindByVersion(imputer_schemas, 1);
  ASSERT_NE(nullptr, imputer_v1);
  EXPECT_EQ(imputer_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(imputer_v1->inputs().size(), 1u);
  EXPECT_EQ(imputer_v1->outputs().size(), 1u);
  EXPECT_EQ(imputer_v1->inputs()[0].name, "X");
  EXPECT_EQ(imputer_v1->inputs()[0].type, "T");
  EXPECT_EQ(imputer_v1->outputs()[0].name, "Y");
  EXPECT_EQ(imputer_v1->outputs()[0].type, "T");
  EXPECT_EQ(imputer_v1->type_constraints().size(), 1u);
  EXPECT_EQ(imputer_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(imputer_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(imputer_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(imputer_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kDouble);
  EXPECT_EQ(imputer_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kInt64);
  EXPECT_EQ(imputer_v1->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kInt32);
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsNormalizerSchema) {
  const std::vector<onnx_op::LightOpSchema> normalizer_schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory("Normalizer");

  const onnx_op::LightOpSchema *const normalizer_v1 = FindByVersion(normalizer_schemas, 1);
  ASSERT_NE(nullptr, normalizer_v1);
  EXPECT_EQ(normalizer_v1->domain(), "ai.onnx.ml");
  EXPECT_EQ(normalizer_v1->inputs().size(), 1u);
  EXPECT_EQ(normalizer_v1->outputs().size(), 1u);
  EXPECT_EQ(normalizer_v1->inputs()[0].name, "X");
  EXPECT_EQ(normalizer_v1->inputs()[0].type, "T");
  EXPECT_EQ(normalizer_v1->outputs()[0].name, "Y");
  EXPECT_EQ(normalizer_v1->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(normalizer_v1->type_constraints().size(), 1u);
  EXPECT_EQ(normalizer_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(normalizer_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(normalizer_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(normalizer_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kDouble);
  EXPECT_EQ(normalizer_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kInt64);
  EXPECT_EQ(normalizer_v1->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kInt32);
}

} // namespace Test
