// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include "onnx_lib/defs/operator_sets.h"
#include "onnx_lib/defs/schema.h"
#include <gtest/gtest.h>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
const onnx_op::math::LightOpSchema *
FindSchema(const std::vector<onnx_op::math::LightOpSchema> &schemas, const std::string &op_type,
           int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpMathRegistrationTest, ReturnsSchemasWithoutShapeInference) {
  const std::vector<onnx_op::math::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 22u);

  const onnx_op::math::LightOpSchema *const add = FindSchema(schemas, "Add", 14);
  const onnx_op::math::LightOpSchema *const add_v1 = FindSchema(schemas, "Add", 1);
  const onnx_op::math::LightOpSchema *const mul_v13 = FindSchema(schemas, "Mul", 13);
  const onnx_op::math::LightOpSchema *const div_v7 = FindSchema(schemas, "Div", 7);
  const onnx_op::math::LightOpSchema *const sub_v6 = FindSchema(schemas, "Sub", 6);
  const onnx_op::math::LightOpSchema *const and_v7 = FindSchema(schemas, "And", 7);
  const onnx_op::math::LightOpSchema *const and_v1 = FindSchema(schemas, "And", 1);
  ASSERT_NE(nullptr, add);
  ASSERT_NE(nullptr, add_v1);
  ASSERT_NE(nullptr, mul_v13);
  ASSERT_NE(nullptr, div_v7);
  ASSERT_NE(nullptr, sub_v6);
  ASSERT_NE(nullptr, and_v7);
  ASSERT_NE(nullptr, and_v1);
  EXPECT_EQ(add->domain(), "ai.onnx");
  EXPECT_EQ(add->since_version(), 14);
  EXPECT_FALSE(add->has_function_implementation());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);
  EXPECT_EQ(add->type_constraints().size(), 1u);
  EXPECT_EQ(and_v7->inputs().size(), 2u);
  EXPECT_EQ(and_v7->outputs().size(), 1u);
  EXPECT_EQ(and_v7->type_constraints().size(), 2u);
  EXPECT_EQ(and_v7->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(and_v7->type_constraints()[1].type_param_str, "T1");
  EXPECT_EQ(and_v7->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v7->type_constraints()[0].allowed_type_strs[0], "tensor(bool)");
  EXPECT_EQ(and_v7->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v7->type_constraints()[1].allowed_type_strs[0], "tensor(bool)");
  EXPECT_EQ(and_v1->since_version(), 1);
  EXPECT_EQ(and_v1->type_constraints().size(), 2u);
  EXPECT_EQ(and_v1->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(and_v1->type_constraints()[0].allowed_type_strs[0], "tensor(bool)");
}

TEST(OnnxOpMathRegistrationTest, MatchesOnnxLibDefinitionsForAllOnnxOpSchemas) {
  onnx_light::RegisterOnnxOperatorSetSchema(0, false);
  const std::vector<onnx_op::math::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();

  for (const onnx_op::math::LightOpSchema &schema : schemas) {
    SCOPED_TRACE(schema.name() + "@" + std::to_string(schema.since_version()));
    const std::string domain =
        onnx_light::IsOnnxDomain(schema.domain()) ? onnx_light::ONNX_DOMAIN : schema.domain();
    const onnx_light::OpSchema *const onnx_lib_schema =
        onnx_light::OpSchemaRegistry::Schema(schema.name(), schema.since_version(), domain);
    ASSERT_NE(onnx_lib_schema, nullptr);
    EXPECT_EQ(onnx_lib_schema->Name(), schema.name());
    EXPECT_EQ(onnx_lib_schema->SinceVersion(), schema.since_version());
  }
}

} // namespace Test
