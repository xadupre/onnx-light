// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxOpMathRegistrationTest, RegistersSchemasManuallyWithoutShapeInference) {
  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();

  EXPECT_EQ(nullptr, onnx_op::math::GetOnnxOpMathSchema("Add", 14));

  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema();

  const onnx_op::math::MathOpSchema *const add = onnx_op::math::GetOnnxOpMathSchema("Add", 14);
  ASSERT_NE(nullptr, add);
  EXPECT_EQ(add->domain(), onnx_op::math::OnnxOpMathDomain());
  EXPECT_EQ(add->since_version(), 14);
  EXPECT_FALSE(add->has_type_and_shape_inference_function());
  EXPECT_FALSE(add->has_data_propagation_function());
  EXPECT_EQ(add->inputs().size(), 2u);
  EXPECT_EQ(add->outputs().size(), 1u);
  EXPECT_EQ(add->type_constraints().size(), 1u);

  EXPECT_NO_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(0, false));

  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  EXPECT_EQ(nullptr, onnx_op::math::GetOnnxOpMathSchema("Add", 14));
}

TEST(OnnxOpMathRegistrationTest, DuplicateRegistrationCanFailWhenRequested) {
  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  onnx_op::math::RegisterOnnxOpMathOperatorSetSchema();

  EXPECT_THROW(onnx_op::math::RegisterOnnxOpMathOperatorSetSchema(0, true),
               onnx_op::math::SchemaError);

  onnx_op::math::DeregisterOnnxOpMathOperatorSetSchema();
  EXPECT_EQ(nullptr, onnx_op::math::GetOnnxOpMathSchema("Add", 14));
}

} // namespace Test
