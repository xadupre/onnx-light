// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_preview.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const core::schema::LightOpSchema *
FindByVersion(const std::vector<core::schema::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpPreviewRegistrationTest, ReturnsFlexAttentionSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::preview::GetAllOnnxOpPreviewSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> flex_attention_schemas =
      onnx_op::preview::GetAllOnnxOpPreviewSchemasWithHistory("FlexAttention");

  EXPECT_EQ(schemas.size(), 1u);

  const core::schema::LightOpSchema *const flex_attention_v1 =
      FindByVersion(flex_attention_schemas, 1);
  ASSERT_NE(nullptr, flex_attention_v1);
  EXPECT_EQ(flex_attention_v1->domain(), "ai.onnx.preview");
  EXPECT_EQ(flex_attention_v1->domain(), onnx_op::preview::kOnnxPreviewDomain);
  EXPECT_TRUE(flex_attention_v1->has_function_implementation());

  ASSERT_EQ(flex_attention_v1->inputs().size(), 3u);
  EXPECT_EQ(flex_attention_v1->inputs()[0].name, "Q");
  EXPECT_EQ(flex_attention_v1->inputs()[1].name, "K");
  EXPECT_EQ(flex_attention_v1->inputs()[2].name, "V");
  EXPECT_EQ(flex_attention_v1->inputs()[0].type, "T1");
  EXPECT_EQ(flex_attention_v1->inputs()[1].type, "T1");
  EXPECT_EQ(flex_attention_v1->inputs()[2].type, "T1");

  ASSERT_EQ(flex_attention_v1->outputs().size(), 1u);
  EXPECT_EQ(flex_attention_v1->outputs()[0].name, "Y");
  EXPECT_EQ(flex_attention_v1->outputs()[0].type, "T1");

  ASSERT_EQ(flex_attention_v1->type_constraints().size(), 1u);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].description,
            "Constrain Q, K, V to float tensors.");
  ASSERT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[0],
            core::schema::TensorType::kBfloat16);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[1],
            core::schema::TensorType::kFloat16);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[2],
            core::schema::TensorType::kFloat);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[3],
            core::schema::TensorType::kDouble);

  EXPECT_NE(flex_attention_v1->doc().find("scaled dot-product attention"), std::string::npos);
  EXPECT_NE(flex_attention_v1->doc().find("score_mod"), std::string::npos);
  EXPECT_NE(flex_attention_v1->doc().find("prob_mod"), std::string::npos);
}

} // namespace Test
