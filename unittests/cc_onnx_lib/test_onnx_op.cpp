// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/math/operator_sets.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxOpMathRegistrationTest, RegistersSchemasManuallyWithoutShapeInference) {
  const std::string &domain = onnx_op::math::OnnxOpMathDomain();
  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();

  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add", 14, domain));

  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema();

  const OpSchema *const add = OpSchemaRegistry::Schema("Add", 14, domain);
  ASSERT_NE(nullptr, add);
  EXPECT_FALSE(add->has_type_and_shape_inference_function());
  EXPECT_FALSE(add->has_data_propagation_function());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);

  EXPECT_NO_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(0, false));

  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add", 14, domain));
}

TEST(OnnxOpMathRegistrationTest, DuplicateRegistrationCanFailWhenRequested) {
  const std::string &domain = onnx_op::math::OnnxOpMathDomain();
  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema();

  EXPECT_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(0, true), SchemaError);

  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add", 14, domain));
}

} // namespace Test
