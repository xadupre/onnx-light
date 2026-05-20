// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

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

  EXPECT_EQ(schemas.size(), 28u);

  const onnx_op::math::LightOpSchema *const add = FindSchema(schemas, "Add", 14);
  const onnx_op::math::LightOpSchema *const add_v1 = FindSchema(schemas, "Add", 1);
  const onnx_op::math::LightOpSchema *const mul_v13 = FindSchema(schemas, "Mul", 13);
  const onnx_op::math::LightOpSchema *const div_v7 = FindSchema(schemas, "Div", 7);
  const onnx_op::math::LightOpSchema *const sub_v6 = FindSchema(schemas, "Sub", 6);
  const onnx_op::math::LightOpSchema *const sin_v22 = FindSchema(schemas, "Sin", 22);
  const onnx_op::math::LightOpSchema *const sin_v7 = FindSchema(schemas, "Sin", 7);
  const onnx_op::math::LightOpSchema *const cos_v22 = FindSchema(schemas, "Cos", 22);
  const onnx_op::math::LightOpSchema *const cos_v7 = FindSchema(schemas, "Cos", 7);
  const onnx_op::math::LightOpSchema *const sinh_v22 = FindSchema(schemas, "Sinh", 22);
  const onnx_op::math::LightOpSchema *const sinh_v9 = FindSchema(schemas, "Sinh", 9);
  const onnx_op::math::LightOpSchema *const cosh_v22 = FindSchema(schemas, "Cosh", 22);
  const onnx_op::math::LightOpSchema *const cosh_v9 = FindSchema(schemas, "Cosh", 9);
  ASSERT_NE(nullptr, add);
  ASSERT_NE(nullptr, add_v1);
  ASSERT_NE(nullptr, mul_v13);
  ASSERT_NE(nullptr, div_v7);
  ASSERT_NE(nullptr, sub_v6);
  ASSERT_NE(nullptr, sin_v22);
  ASSERT_NE(nullptr, sin_v7);
  ASSERT_NE(nullptr, cos_v22);
  ASSERT_NE(nullptr, cos_v7);
  ASSERT_NE(nullptr, sinh_v22);
  ASSERT_NE(nullptr, sinh_v9);
  ASSERT_NE(nullptr, cosh_v22);
  ASSERT_NE(nullptr, cosh_v9);
  EXPECT_EQ(add->domain(), "ai.onnx");
  EXPECT_EQ(add->since_version(), 14);
  EXPECT_FALSE(add->has_function_implementation());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);
  EXPECT_EQ(add->type_constraints().size(), 1u);
  EXPECT_NE(add_v1->inputs()[0].description, add->inputs()[0].description);
  EXPECT_NE(add_v1->type_constraints()[0].allowed_type_strs,
            add->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sin_v22->inputs().size(), 1u);
  EXPECT_EQ(sin_v22->outputs().size(), 1u);
  EXPECT_EQ(sin_v22->type_constraints().size(), 1u);
  EXPECT_EQ(sin_v7->type_constraints().size(), 1u);
  EXPECT_NE(sin_v7->type_constraints()[0].allowed_type_strs,
            sin_v22->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(sin_v22->type_constraints()[0].allowed_type_strs.front(), "tensor(bfloat16)");
  EXPECT_EQ(cosh_v22->type_constraints()[0].allowed_type_strs.front(), "tensor(bfloat16)");
}

} // namespace Test
