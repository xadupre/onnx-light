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
FindLogicalSchema(const std::vector<onnx_op::logical::LightOpSchema> &schemas,
                  const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpLogicalRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::logical::LightOpSchema> schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 24u);

  const onnx_op::logical::LightOpSchema *const and_v7 = FindLogicalSchema(schemas, "And", 7);
  const onnx_op::logical::LightOpSchema *const and_v1 = FindLogicalSchema(schemas, "And", 1);
  const onnx_op::logical::LightOpSchema *const or_v7 = FindLogicalSchema(schemas, "Or", 7);
  const onnx_op::logical::LightOpSchema *const or_v1 = FindLogicalSchema(schemas, "Or", 1);
  const onnx_op::logical::LightOpSchema *const xor_v7 = FindLogicalSchema(schemas, "Xor", 7);
  const onnx_op::logical::LightOpSchema *const xor_v1 = FindLogicalSchema(schemas, "Xor", 1);
  const onnx_op::logical::LightOpSchema *const greater_v13 =
      FindLogicalSchema(schemas, "Greater", 13);
  const onnx_op::logical::LightOpSchema *const greater_v9 =
      FindLogicalSchema(schemas, "Greater", 9);
  const onnx_op::logical::LightOpSchema *const greater_v7 =
      FindLogicalSchema(schemas, "Greater", 7);
  const onnx_op::logical::LightOpSchema *const greater_v1 =
      FindLogicalSchema(schemas, "Greater", 1);
  const onnx_op::logical::LightOpSchema *const less_v13 = FindLogicalSchema(schemas, "Less", 13);
  const onnx_op::logical::LightOpSchema *const less_v9 = FindLogicalSchema(schemas, "Less", 9);
  const onnx_op::logical::LightOpSchema *const less_v7 = FindLogicalSchema(schemas, "Less", 7);
  const onnx_op::logical::LightOpSchema *const less_v1 = FindLogicalSchema(schemas, "Less", 1);
  const onnx_op::logical::LightOpSchema *const equal_v19 = FindLogicalSchema(schemas, "Equal", 19);
  const onnx_op::logical::LightOpSchema *const equal_v13 = FindLogicalSchema(schemas, "Equal", 13);
  const onnx_op::logical::LightOpSchema *const equal_v11 = FindLogicalSchema(schemas, "Equal", 11);
  const onnx_op::logical::LightOpSchema *const equal_v7 = FindLogicalSchema(schemas, "Equal", 7);
  const onnx_op::logical::LightOpSchema *const equal_v1 = FindLogicalSchema(schemas, "Equal", 1);
  const onnx_op::logical::LightOpSchema *const not_v1 = FindLogicalSchema(schemas, "Not", 1);
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
}

TEST(OnnxOpLogicalRegistrationTest, BitwiseSchemasArePresent) {
  const std::vector<onnx_op::logical::LightOpSchema> schemas =
      onnx_op::logical::GetAllOnnxOpLogicalSchemasWithHistory();

  const onnx_op::logical::LightOpSchema *const bw_and =
      FindLogicalSchema(schemas, "BitwiseAnd", 18);
  const onnx_op::logical::LightOpSchema *const bw_or = FindLogicalSchema(schemas, "BitwiseOr", 18);
  const onnx_op::logical::LightOpSchema *const bw_xor =
      FindLogicalSchema(schemas, "BitwiseXor", 18);
  const onnx_op::logical::LightOpSchema *const bw_not =
      FindLogicalSchema(schemas, "BitwiseNot", 18);
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
