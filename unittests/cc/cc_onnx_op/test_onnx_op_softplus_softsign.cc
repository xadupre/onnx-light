// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

const onnx_op::LightOpSchema *FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas,
                                            int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

} // namespace

TEST(OnnxOpMathSoftplus, HasSchemasForV1AndV22) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Softplus");
  ASSERT_EQ(schemas.size(), 2u);
  const onnx_op::LightOpSchema *const softplus_v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const softplus_v1 = FindByVersion(schemas, 1);
  ASSERT_NE(nullptr, softplus_v22);
  ASSERT_NE(nullptr, softplus_v1);
  EXPECT_EQ(softplus_v22->domain(), "ai.onnx");
  EXPECT_EQ(softplus_v22->name(), "Softplus");
  EXPECT_TRUE(softplus_v22->has_function_implementation());
  ASSERT_EQ(softplus_v22->inputs().size(), 1u);
  EXPECT_EQ(softplus_v22->inputs()[0].name, "X");
  ASSERT_EQ(softplus_v22->outputs().size(), 1u);
  EXPECT_EQ(softplus_v22->outputs()[0].name, "Y");
  EXPECT_TRUE(softplus_v22->attributes().empty());
  EXPECT_TRUE(softplus_v1->has_function_implementation());
}

TEST(OnnxOpMathSoftsign, HasSchemasForV1AndV22) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Softsign");
  ASSERT_EQ(schemas.size(), 2u);
  const onnx_op::LightOpSchema *const softsign_v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const softsign_v1 = FindByVersion(schemas, 1);
  ASSERT_NE(nullptr, softsign_v22);
  ASSERT_NE(nullptr, softsign_v1);
  EXPECT_EQ(softsign_v22->domain(), "ai.onnx");
  EXPECT_EQ(softsign_v22->name(), "Softsign");
  EXPECT_TRUE(softsign_v22->has_function_implementation());
  ASSERT_EQ(softsign_v22->inputs().size(), 1u);
  EXPECT_EQ(softsign_v22->inputs()[0].name, "input");
  ASSERT_EQ(softsign_v22->outputs().size(), 1u);
  EXPECT_EQ(softsign_v22->outputs()[0].name, "output");
  EXPECT_TRUE(softsign_v22->attributes().empty());
  EXPECT_TRUE(softsign_v1->has_function_implementation());
}

} // namespace Test
