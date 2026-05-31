// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"

#include <gtest/gtest.h>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

const onnx_op::logical::LightOpSchema *
FindByVersion(const std::vector<onnx_op::logical::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpLogicalRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::logical::LightOpSchema> schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory();
  const std::vector<onnx_op::logical::LightOpSchema> and_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("And");
  const std::vector<onnx_op::logical::LightOpSchema> or_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Or");
  const std::vector<onnx_op::logical::LightOpSchema> xor_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Xor");
  const std::vector<onnx_op::logical::LightOpSchema> greater_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Greater");
  const std::vector<onnx_op::logical::LightOpSchema> greater_or_equal_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("GreaterOrEqual");
  const std::vector<onnx_op::logical::LightOpSchema> less_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Less");
  const std::vector<onnx_op::logical::LightOpSchema> equal_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Equal");
  const std::vector<onnx_op::logical::LightOpSchema> not_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Not");
  const std::vector<onnx_op::logical::LightOpSchema> where_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("Where");

  EXPECT_EQ(schemas.size(), 28u);

  const onnx_op::logical::LightOpSchema *const and_v7 = FindByVersion(and_schemas, 7);
  const onnx_op::logical::LightOpSchema *const and_v1 = FindByVersion(and_schemas, 1);
  const onnx_op::logical::LightOpSchema *const or_v7 = FindByVersion(or_schemas, 7);
  const onnx_op::logical::LightOpSchema *const or_v1 = FindByVersion(or_schemas, 1);
  const onnx_op::logical::LightOpSchema *const xor_v7 = FindByVersion(xor_schemas, 7);
  const onnx_op::logical::LightOpSchema *const xor_v1 = FindByVersion(xor_schemas, 1);
  const onnx_op::logical::LightOpSchema *const greater_v13 = FindByVersion(greater_schemas, 13);
  const onnx_op::logical::LightOpSchema *const greater_v9 = FindByVersion(greater_schemas, 9);
  const onnx_op::logical::LightOpSchema *const greater_v7 = FindByVersion(greater_schemas, 7);
  const onnx_op::logical::LightOpSchema *const greater_v1 = FindByVersion(greater_schemas, 1);
  const onnx_op::logical::LightOpSchema *const greater_or_equal_v16 =
      FindByVersion(greater_or_equal_schemas, 16);
  const onnx_op::logical::LightOpSchema *const greater_or_equal_v12 =
      FindByVersion(greater_or_equal_schemas, 12);
  const onnx_op::logical::LightOpSchema *const less_v13 = FindByVersion(less_schemas, 13);
  const onnx_op::logical::LightOpSchema *const less_v9 = FindByVersion(less_schemas, 9);
  const onnx_op::logical::LightOpSchema *const less_v7 = FindByVersion(less_schemas, 7);
  const onnx_op::logical::LightOpSchema *const less_v1 = FindByVersion(less_schemas, 1);
  const onnx_op::logical::LightOpSchema *const equal_v19 = FindByVersion(equal_schemas, 19);
  const onnx_op::logical::LightOpSchema *const equal_v13 = FindByVersion(equal_schemas, 13);
  const onnx_op::logical::LightOpSchema *const equal_v11 = FindByVersion(equal_schemas, 11);
  const onnx_op::logical::LightOpSchema *const equal_v7 = FindByVersion(equal_schemas, 7);
  const onnx_op::logical::LightOpSchema *const equal_v1 = FindByVersion(equal_schemas, 1);
  const onnx_op::logical::LightOpSchema *const not_v1 = FindByVersion(not_schemas, 1);
  const onnx_op::logical::LightOpSchema *const where_v16 = FindByVersion(where_schemas, 16);
  const onnx_op::logical::LightOpSchema *const where_v9 = FindByVersion(where_schemas, 9);
  ASSERT_NE(nullptr, and_v7);
  ASSERT_NE(nullptr, and_v1);
  ASSERT_NE(nullptr, or_v7);
  ASSERT_NE(nullptr, or_v1);
  ASSERT_NE(nullptr, xor_v7);
  ASSERT_NE(nullptr, xor_v1);
  ASSERT_NE(nullptr, greater_v13);
  ASSERT_NE(nullptr, greater_v9);
  ASSERT_NE(nullptr, greater_v7);
  ASSERT_NE(nullptr, greater_v1);
  ASSERT_NE(nullptr, greater_or_equal_v16);
  ASSERT_NE(nullptr, greater_or_equal_v12);
  ASSERT_NE(nullptr, less_v13);
  ASSERT_NE(nullptr, less_v9);
  ASSERT_NE(nullptr, less_v7);
  ASSERT_NE(nullptr, less_v1);
  ASSERT_NE(nullptr, equal_v19);
  ASSERT_NE(nullptr, equal_v13);
  ASSERT_NE(nullptr, equal_v11);
  ASSERT_NE(nullptr, equal_v7);
  ASSERT_NE(nullptr, equal_v1);
  ASSERT_NE(nullptr, not_v1);
  ASSERT_NE(nullptr, where_v16);
  ASSERT_NE(nullptr, where_v9);
  EXPECT_EQ(and_v7->inputs().size(), 2u);
  EXPECT_EQ(and_v7->outputs().size(), 1u);
  EXPECT_EQ(and_v7->type_constraints().size(), 2u);
  EXPECT_EQ(and_v7->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(and_v7->type_constraints()[1].type_param_str, "T1");
  EXPECT_EQ(and_v7->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v7->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(and_v7->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v7->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(and_v1->since_version(), 1);
  EXPECT_EQ(and_v1->type_constraints().size(), 2u);
  EXPECT_EQ(and_v1->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_NE(and_v1->doc(), and_v7->doc());
  EXPECT_NE(and_v1->inputs()[0].description, and_v7->inputs()[0].description);
  EXPECT_EQ(or_v7->inputs().size(), 2u);
  EXPECT_EQ(or_v7->type_constraints().size(), 2u);
  EXPECT_EQ(or_v1->inputs()[0].description, and_v1->inputs()[0].description);
  EXPECT_EQ(or_v7->inputs()[0].description, and_v7->inputs()[0].description);
  EXPECT_EQ(xor_v7->type_constraints().size(), 2u);
  EXPECT_EQ(xor_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(greater_v7->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(greater_v7->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);
  EXPECT_EQ(greater_v9->type_constraints()[0].allowed_type_strs.size(), 11u);
  EXPECT_EQ(greater_v13->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(greater_v1->inputs()[0].description, and_v1->inputs()[0].description);
  EXPECT_EQ(less_v13->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(greater_or_equal_v12->type_constraints()[0].allowed_type_strs.size(), 11u);
  EXPECT_EQ(greater_or_equal_v16->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(greater_or_equal_v16->inputs().size(), 2u);
  EXPECT_EQ(greater_or_equal_v16->outputs().size(), 1u);
  EXPECT_EQ(greater_or_equal_v16->inputs()[0].name, "A");
  EXPECT_EQ(greater_or_equal_v16->inputs()[1].name, "B");
  EXPECT_EQ(greater_or_equal_v16->outputs()[0].name, "C");
  EXPECT_EQ(greater_or_equal_v16->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(greater_or_equal_v16->type_constraints()[1].allowed_type_strs[0],
            onnx_op::TensorType::kBool);
  EXPECT_EQ(equal_v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(equal_v11->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(equal_v13->type_constraints()[0].allowed_type_strs.size(), 13u);
  EXPECT_EQ(equal_v19->type_constraints()[0].allowed_type_strs.size(), 14u);
  EXPECT_EQ(equal_v19->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kString);
  EXPECT_EQ(not_v1->inputs().size(), 1u);
  EXPECT_EQ(not_v1->outputs().size(), 1u);
  EXPECT_EQ(not_v1->inputs()[0].name, "X");
  EXPECT_EQ(not_v1->outputs()[0].name, "Y");
  EXPECT_EQ(not_v1->type_constraints().size(), 1u);
  EXPECT_EQ(not_v1->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(not_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(where_v16->inputs().size(), 3u);
  EXPECT_EQ(where_v16->outputs().size(), 1u);
  EXPECT_EQ(where_v16->inputs()[0].name, "condition");
  EXPECT_EQ(where_v16->inputs()[1].name, "X");
  EXPECT_EQ(where_v16->inputs()[2].name, "Y");
  EXPECT_EQ(where_v16->outputs()[0].name, "output");
  EXPECT_EQ(where_v16->type_constraints().size(), 2u);
  EXPECT_EQ(where_v16->type_constraints()[0].type_param_str, "B");
  EXPECT_EQ(where_v16->type_constraints()[1].type_param_str, "T");
  EXPECT_EQ(where_v9->type_constraints()[1].allowed_type_strs.size(), 15u);
  EXPECT_EQ(where_v16->type_constraints()[1].allowed_type_strs.size(), 16u);
  EXPECT_EQ(where_v16->type_constraints()[1].allowed_type_strs[8], onnx_op::TensorType::kBfloat16);
}

TEST(OnnxOpLogicalRegistrationTest, BitwiseSchemasArePresent) {
  const std::vector<onnx_op::logical::LightOpSchema> bw_and_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("BitwiseAnd");
  const std::vector<onnx_op::logical::LightOpSchema> bw_or_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("BitwiseOr");
  const std::vector<onnx_op::logical::LightOpSchema> bw_xor_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("BitwiseXor");
  const std::vector<onnx_op::logical::LightOpSchema> bw_not_schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory("BitwiseNot");

  const onnx_op::logical::LightOpSchema *const bw_and = FindByVersion(bw_and_schemas, 18);
  const onnx_op::logical::LightOpSchema *const bw_or = FindByVersion(bw_or_schemas, 18);
  const onnx_op::logical::LightOpSchema *const bw_xor = FindByVersion(bw_xor_schemas, 18);
  const onnx_op::logical::LightOpSchema *const bw_not = FindByVersion(bw_not_schemas, 18);
  ASSERT_NE(nullptr, bw_and);
  ASSERT_NE(nullptr, bw_or);
  ASSERT_NE(nullptr, bw_xor);
  ASSERT_NE(nullptr, bw_not);

  // Binary bitwise operators: two T inputs / one T output / single
  // ``T`` type constraint covering the 8 integer dtypes.
  for (const auto *schema : {bw_and, bw_or, bw_xor}) {
    EXPECT_EQ(schema->inputs().size(), 2u);
    EXPECT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "A");
    EXPECT_EQ(schema->inputs()[1].name, "B");
    EXPECT_EQ(schema->outputs()[0].name, "C");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs.size(), 8u);
    EXPECT_EQ(schema->type_constraints()[0].description, "Constrain input to integer tensors.");
  }

  // Unary BitwiseNot: single T input/output, single ``T`` type constraint.
  EXPECT_EQ(bw_not->inputs().size(), 1u);
  EXPECT_EQ(bw_not->outputs().size(), 1u);
  EXPECT_EQ(bw_not->inputs()[0].name, "X");
  EXPECT_EQ(bw_not->outputs()[0].name, "Y");
  ASSERT_EQ(bw_not->type_constraints().size(), 1u);
  EXPECT_EQ(bw_not->type_constraints()[0].allowed_type_strs.size(), 8u);
  EXPECT_EQ(bw_not->type_constraints()[0].description,
            "Constrain input/output to integer tensors.");
}

} // namespace Test
