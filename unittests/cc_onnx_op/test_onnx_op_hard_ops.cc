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

const onnx_op::AttributeParam *FindAttr(const onnx_op::LightOpSchema &schema, const char *name) {
  for (const auto &attr : schema.attributes()) {
    if (attr.name == name) {
      return &attr;
    }
  }
  return nullptr;
}

} // namespace

TEST(OnnxOpMathHardSigmoid, HasSchemasForV1V6V22) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("HardSigmoid");
  ASSERT_EQ(schemas.size(), 3u);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v6 = FindByVersion(schemas, 6);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v6);
  ASSERT_NE(nullptr, v1);
  EXPECT_EQ(v22->domain(), "ai.onnx");
  EXPECT_EQ(v22->name(), "HardSigmoid");
  EXPECT_TRUE(v22->has_function_implementation());
  ASSERT_EQ(v22->inputs().size(), 1u);
  EXPECT_EQ(v22->inputs()[0].name, "X");
  ASSERT_EQ(v22->outputs().size(), 1u);
  EXPECT_EQ(v22->outputs()[0].name, "Y");
  ASSERT_NE(FindAttr(*v22, "alpha"), nullptr);
  ASSERT_NE(FindAttr(*v22, "beta"), nullptr);
  EXPECT_TRUE(v6->has_function_implementation());
}

TEST(OnnxOpMathHardSwish, HasSchemasForV14AndV22) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("HardSwish");
  ASSERT_EQ(schemas.size(), 2u);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v14 = FindByVersion(schemas, 14);
  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v14);
  EXPECT_EQ(v22->name(), "HardSwish");
  EXPECT_TRUE(v22->has_function_implementation());
  EXPECT_TRUE(v14->has_function_implementation());
  ASSERT_EQ(v22->inputs().size(), 1u);
  EXPECT_EQ(v22->inputs()[0].name, "X");
  ASSERT_EQ(v22->outputs().size(), 1u);
  EXPECT_EQ(v22->outputs()[0].name, "Y");
  EXPECT_TRUE(v22->attributes().empty());
}

TEST(OnnxOpMathHardmax, HasSchemasForV1V11V13) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("Hardmax");
  ASSERT_EQ(schemas.size(), 3u);
  const onnx_op::LightOpSchema *const v13 = FindByVersion(schemas, 13);
  const onnx_op::LightOpSchema *const v11 = FindByVersion(schemas, 11);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v11);
  ASSERT_NE(nullptr, v1);
  EXPECT_EQ(v13->name(), "Hardmax");
  ASSERT_EQ(v13->inputs().size(), 1u);
  EXPECT_EQ(v13->inputs()[0].name, "input");
  ASSERT_EQ(v13->outputs().size(), 1u);
  EXPECT_EQ(v13->outputs()[0].name, "output");
  const onnx_op::AttributeParam *axis_v13 = FindAttr(*v13, "axis");
  ASSERT_NE(axis_v13, nullptr);
  EXPECT_FALSE(axis_v13->required);
  const onnx_op::AttributeParam *axis_v11 = FindAttr(*v11, "axis");
  ASSERT_NE(axis_v11, nullptr);
  EXPECT_FALSE(axis_v11->required);
}

} // namespace Test
