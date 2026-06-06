// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_optional.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedOptionalSchemaCount = 5;

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpOptionalRegistrationTest, ReturnsOptionalSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> optional_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("Optional");
  const std::vector<onnx_op::LightOpSchema> optional_has_element_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("OptionalHasElement");
  const std::vector<onnx_op::LightOpSchema> optional_get_element_schemas =
      onnx_op::optional::GetAllOnnxOpOptionalSchemasWithHistory("OptionalGetElement");

  EXPECT_EQ(schemas.size(), kExpectedOptionalSchemaCount);

  const onnx_op::LightOpSchema *const optional_v15 = FindByVersion(optional_schemas, 15);
  const onnx_op::LightOpSchema *const has_v18 = FindByVersion(optional_has_element_schemas, 18);
  const onnx_op::LightOpSchema *const has_v15 = FindByVersion(optional_has_element_schemas, 15);
  const onnx_op::LightOpSchema *const get_v18 = FindByVersion(optional_get_element_schemas, 18);
  const onnx_op::LightOpSchema *const get_v15 = FindByVersion(optional_get_element_schemas, 15);

  ASSERT_NE(nullptr, optional_v15);
  ASSERT_NE(nullptr, has_v18);
  ASSERT_NE(nullptr, has_v15);
  ASSERT_NE(nullptr, get_v18);
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
  EXPECT_STREQ(onnx_op::ToTypeString(optional_v15->type_constraints()[1].allowed_type_strs[0]),
               "optional(seq(tensor(uint8)))");
  EXPECT_STREQ(onnx_op::ToTypeString(optional_v15->type_constraints()[1].allowed_type_strs.back()),
               "optional(tensor(complex128))");

  // OptionalHasElement v18 has expanded input domain (optional + tensor + seq)
  ASSERT_EQ(has_v18->type_constraints().size(), 2u);
  EXPECT_EQ(has_v18->type_constraints()[0].type_param_str, "O");
  EXPECT_EQ(has_v18->type_constraints()[0].allowed_type_strs.size(), 60u);
  EXPECT_EQ(has_v18->type_constraints()[1].type_param_str, "B");
  ASSERT_EQ(has_v18->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(has_v18->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kBool);

  // OptionalHasElement v15 input domain is only optional types
  EXPECT_EQ(has_v15->type_constraints()[0].allowed_type_strs.size(), 30u);

  // OptionalGetElement v18 expanded input domain
  ASSERT_EQ(get_v18->type_constraints().size(), 2u);
  EXPECT_EQ(get_v18->type_constraints()[0].allowed_type_strs.size(), 60u);
  EXPECT_EQ(get_v18->type_constraints()[1].type_param_str, "V");
  EXPECT_EQ(get_v18->type_constraints()[1].allowed_type_strs.size(), 30u);

  // OptionalGetElement v15 input domain is only optional types
  EXPECT_EQ(get_v15->type_constraints()[0].allowed_type_strs.size(), 30u);
}

} // namespace Test
