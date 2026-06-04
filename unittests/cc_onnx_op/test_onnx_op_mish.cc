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

TEST(OnnxOpMathMish, HasSchemasForV18AndV22) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Mish");
  ASSERT_EQ(schemas.size(), 2u);
  const onnx_op::LightOpSchema *const mish_v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const mish_v18 = FindByVersion(schemas, 18);
  ASSERT_NE(nullptr, mish_v22);
  ASSERT_NE(nullptr, mish_v18);
  EXPECT_EQ(mish_v22->domain(), "ai.onnx");
  EXPECT_EQ(mish_v22->name(), "Mish");
  EXPECT_TRUE(mish_v22->has_function_implementation());
  ASSERT_EQ(mish_v22->inputs().size(), 1u);
  EXPECT_EQ(mish_v22->inputs()[0].name, "X");
  ASSERT_EQ(mish_v22->outputs().size(), 1u);
  EXPECT_EQ(mish_v22->outputs()[0].name, "Y");
  EXPECT_TRUE(mish_v22->attributes().empty());
  EXPECT_TRUE(mish_v18->has_function_implementation());
  EXPECT_TRUE(mish_v18->attributes().empty());
}

} // namespace Test
