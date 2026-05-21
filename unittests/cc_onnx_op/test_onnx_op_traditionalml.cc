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

  EXPECT_EQ(schemas.size(), 2u);

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

} // namespace Test
