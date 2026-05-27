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

const onnx_op::LightOpSchema *
FindTraditionalMLSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                        const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTraditionalMLRegistrationTest, ReturnsLabelEncoderSchemaWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::traditionalml::GetAllOnnxOpTraditionalMLSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 10u);

  const onnx_op::LightOpSchema *const binarizer_v1 =
      FindTraditionalMLSchema(schemas, "Binarizer", 1);
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

  const onnx_op::LightOpSchema *const label_encoder_v4 =
      FindTraditionalMLSchema(schemas, "LabelEncoder", 4);
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

  const onnx_op::LightOpSchema *const zipmap_v1 = FindTraditionalMLSchema(schemas, "ZipMap", 1);
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

  const onnx_op::LightOpSchema *const tree_ensemble_v5 =
      FindTraditionalMLSchema(schemas, "TreeEnsemble", 5);
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

  for (int version : {1, 3, 5}) {
    const onnx_op::LightOpSchema *const schema =
        FindTraditionalMLSchema(schemas, "TreeEnsembleClassifier", version);
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

  for (int version : {1, 3, 5}) {
    const onnx_op::LightOpSchema *const schema =
        FindTraditionalMLSchema(schemas, "TreeEnsembleRegressor", version);
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

} // namespace Test
