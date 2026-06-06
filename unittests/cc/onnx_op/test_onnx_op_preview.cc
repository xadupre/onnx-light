// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_preview.h"

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

TEST(OnnxOpPreviewRegistrationTest, ReturnsFlexAttentionSchemaWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::preview::GetAllOnnxOpPreviewSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> flex_attention_schemas =
      onnx_op::preview::GetAllOnnxOpPreviewSchemasWithHistory("FlexAttention");

  EXPECT_EQ(schemas.size(), 1u);

  const onnx_op::LightOpSchema *const flex_attention_v1 = FindByVersion(flex_attention_schemas, 1);
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
            onnx_op::TensorType::kBfloat16);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kFloat16);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(flex_attention_v1->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kDouble);

  EXPECT_NE(flex_attention_v1->doc().find("scaled dot-product attention"), std::string::npos);
  EXPECT_NE(flex_attention_v1->doc().find("score_mod"), std::string::npos);
  EXPECT_NE(flex_attention_v1->doc().find("prob_mod"), std::string::npos);
}

} // namespace Test
