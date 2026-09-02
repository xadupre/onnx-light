// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_optional.h"

#include <gtest/gtest.h>

#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedOptionalSchemaCount = 8;

static const core::schema::LightOpSchema *
FindByVersion(const std::vector<core::schema::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpOptionalRegistrationTest, ReturnsOptionalSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> optional_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("Optional");
  const std::vector<core::schema::LightOpSchema> optional_has_element_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("OptionalHasElement");
  const std::vector<core::schema::LightOpSchema> optional_get_element_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("OptionalGetElement");

  EXPECT_EQ(schemas.size(), kExpectedOptionalSchemaCount);

  const core::schema::LightOpSchema *const optional_v15 = FindByVersion(optional_schemas, 15);
  const core::schema::LightOpSchema *const optional_v28 = FindByVersion(optional_schemas, 28);
  const core::schema::LightOpSchema *const has_v18 =
      FindByVersion(optional_has_element_schemas, 18);
  const core::schema::LightOpSchema *const has_v28 =
      FindByVersion(optional_has_element_schemas, 28);
  const core::schema::LightOpSchema *const has_v15 =
      FindByVersion(optional_has_element_schemas, 15);
  const core::schema::LightOpSchema *const get_v18 =
      FindByVersion(optional_get_element_schemas, 18);
  const core::schema::LightOpSchema *const get_v28 =
      FindByVersion(optional_get_element_schemas, 28);
  const core::schema::LightOpSchema *const get_v15 =
      FindByVersion(optional_get_element_schemas, 15);

  ASSERT_NE(nullptr, optional_v15);
  ASSERT_NE(nullptr, optional_v28);
  ASSERT_NE(nullptr, has_v18);
  ASSERT_NE(nullptr, has_v28);
  ASSERT_NE(nullptr, has_v15);
  ASSERT_NE(nullptr, get_v18);
  ASSERT_NE(nullptr, get_v28);
  ASSERT_NE(nullptr, get_v15);

  // Optional v15
  EXPECT_EQ(optional_v15->domain(), "ai.onnx");
  EXPECT_EQ(optional_v15->inputs().size(), 1u);
  EXPECT_EQ(optional_v15->inputs()[0].name, "input");
  EXPECT_EQ(optional_v15->inputs()[0].type, "V");
  EXPECT_EQ(optional_v15->outputs().size(), 1u);
  EXPECT_EQ(optional_v15->outputs()[0].name, "output");
  EXPECT_EQ(optional_v15->outputs()[0].type, "O");
  ASSERT_EQ(optional_v15->type_constraints().size(), 2u);
  EXPECT_EQ(optional_v15->type_constraints()[0].type_param_str, "V");
  // 15 tensor + 15 sequence tensor types
  EXPECT_EQ(optional_v15->type_constraints()[0].allowed_type_strs.size(), 30u);
  EXPECT_EQ(optional_v15->type_constraints()[1].type_param_str, "O");
  EXPECT_EQ(optional_v15->type_constraints()[1].allowed_type_strs.size(), 30u);
  EXPECT_STREQ(core::schema::ToTypeString(optional_v15->type_constraints()[1].allowed_type_strs[0]),
               "optional(seq(tensor(uint8)))");
  EXPECT_STREQ(
      core::schema::ToTypeString(optional_v15->type_constraints()[1].allowed_type_strs.back()),
      "optional(tensor(complex128))");

  // OptionalHasElement v18 has expanded input domain (optional + tensor + seq)
  ASSERT_EQ(has_v18->type_constraints().size(), 2u);
  EXPECT_EQ(has_v18->type_constraints()[0].type_param_str, "O");
  EXPECT_EQ(has_v18->type_constraints()[0].allowed_type_strs.size(), 60u);
  EXPECT_EQ(has_v18->type_constraints()[1].type_param_str, "B");
  ASSERT_EQ(has_v18->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(has_v18->type_constraints()[1].allowed_type_strs[0], core::schema::TensorType::kBool);

  // OptionalHasElement v15 input domain is only optional types
  EXPECT_EQ(has_v15->type_constraints()[0].allowed_type_strs.size(), 30u);

  // OptionalGetElement v18 expanded input domain
  ASSERT_EQ(get_v18->type_constraints().size(), 2u);
  EXPECT_EQ(get_v18->type_constraints()[0].allowed_type_strs.size(), 60u);
  EXPECT_EQ(get_v18->type_constraints()[1].type_param_str, "V");
  EXPECT_EQ(get_v18->type_constraints()[1].allowed_type_strs.size(), 30u);

  // OptionalGetElement v15 input domain is only optional types
  EXPECT_EQ(get_v15->type_constraints()[0].allowed_type_strs.size(), 30u);

  // Opset 28 extends all three operators to every IR14 tensor and sequence type.
  ASSERT_EQ(optional_v28->type_constraints().size(), 2u);
  EXPECT_EQ(optional_v28->type_constraints()[0].allowed_type_strs.size(), 56u);
  EXPECT_EQ(optional_v28->type_constraints()[1].allowed_type_strs.size(), 56u);
  EXPECT_STREQ(
      core::schema::ToTypeString(optional_v28->type_constraints()[0].allowed_type_strs.back()),
      "seq(tensor(float6e3m2))");
  EXPECT_STREQ(
      core::schema::ToTypeString(optional_v28->type_constraints()[1].allowed_type_strs.back()),
      "optional(tensor(float6e3m2))");

  EXPECT_EQ(has_v28->type_constraints()[0].allowed_type_strs.size(), 112u);
  EXPECT_EQ(get_v28->type_constraints()[0].allowed_type_strs.size(), 112u);
  EXPECT_EQ(get_v28->type_constraints()[1].allowed_type_strs.size(), 56u);
  EXPECT_EQ(optional_v28->doc(), optional_v15->doc());
  EXPECT_EQ(has_v28->doc(), has_v18->doc());
  EXPECT_EQ(get_v28->doc(), get_v18->doc());
}

} // namespace Test
