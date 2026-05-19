// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

const onnx_op::math::OpSchema *FindSchema(const std::vector<onnx_op::math::OpSchema> &schemas,
                                          const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpMathRegistrationTest, RegistersSchemasManuallyWithoutShapeInference) {
  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(false);

  const std::vector<onnx_op::math::OpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory();

  const onnx_op::math::OpSchema *const add = FindSchema(schemas, "Add", 14);
  ASSERT_NE(nullptr, add);
  EXPECT_EQ(add->domain(), "ai.onnx");
  EXPECT_EQ(add->since_version(), 14);
  EXPECT_FALSE(add->has_function_implementation());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);
  EXPECT_EQ(add->type_constraints().size(), 1u);

  EXPECT_NO_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(false));
}

TEST(OnnxOpMathRegistrationTest, DuplicateRegistrationCanFailWhenRequested) {
  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(false);
  EXPECT_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(true),
               onnx_op::math::SchemaError);
}

} // namespace Test
