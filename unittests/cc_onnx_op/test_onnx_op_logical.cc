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

  EXPECT_EQ(schemas.size(), 15u);

  const onnx_op::logical::LightOpSchema *const and_v7 = FindLogicalSchema(schemas, "And", 7);
  const onnx_op::logical::LightOpSchema *const and_v1 = FindLogicalSchema(schemas, "And", 1);
  const onnx_op::logical::LightOpSchema *const greater_v13 =
      FindLogicalSchema(schemas, "Greater", 13);
  const onnx_op::logical::LightOpSchema *const less_v9 = FindLogicalSchema(schemas, "Less", 9);
  const onnx_op::logical::LightOpSchema *const equal_v19 = FindLogicalSchema(schemas, "Equal", 19);
  ASSERT_NE(nullptr, and_v7);
  ASSERT_NE(nullptr, and_v1);
  ASSERT_NE(nullptr, greater_v13);
  ASSERT_NE(nullptr, less_v9);
  ASSERT_NE(nullptr, equal_v19);
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
  EXPECT_EQ(greater_v13->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(less_v9->type_constraints()[0].allowed_type_strs.size(), 11u);
  EXPECT_EQ(equal_v19->type_constraints()[0].allowed_type_strs.back(), "tensor(string)");
  EXPECT_NE(and_v1->doc(), and_v7->doc());
  EXPECT_NE(and_v1->inputs()[0].description, and_v7->inputs()[0].description);
}

} // namespace Test
