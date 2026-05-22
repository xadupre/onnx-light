// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_nn.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedAveragePoolSchemaCount = 6;

const onnx_op::LightOpSchema *FindNnSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                                           const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpNnRegistrationTest, ReturnsAveragePoolSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();

  EXPECT_EQ(schemas.size(), kExpectedAveragePoolSchemaCount);

  const onnx_op::LightOpSchema *const ap_v22 = FindNnSchema(schemas, "AveragePool", 22);
  const onnx_op::LightOpSchema *const ap_v19 = FindNnSchema(schemas, "AveragePool", 19);
  const onnx_op::LightOpSchema *const ap_v11 = FindNnSchema(schemas, "AveragePool", 11);
  const onnx_op::LightOpSchema *const ap_v10 = FindNnSchema(schemas, "AveragePool", 10);
  const onnx_op::LightOpSchema *const ap_v7 = FindNnSchema(schemas, "AveragePool", 7);
  const onnx_op::LightOpSchema *const ap_v1 = FindNnSchema(schemas, "AveragePool", 1);

  ASSERT_NE(nullptr, ap_v22);
  ASSERT_NE(nullptr, ap_v19);
  ASSERT_NE(nullptr, ap_v11);
  ASSERT_NE(nullptr, ap_v10);
  ASSERT_NE(nullptr, ap_v7);
  ASSERT_NE(nullptr, ap_v1);

  EXPECT_EQ(ap_v22->domain(), "ai.onnx");
  EXPECT_EQ(ap_v22->inputs().size(), 1u);
  EXPECT_EQ(ap_v22->outputs().size(), 1u);
  EXPECT_EQ(ap_v22->type_constraints().size(), 1u);
  EXPECT_EQ(ap_v22->inputs()[0].name, "X");
  EXPECT_EQ(ap_v22->inputs()[0].type, "T");
  EXPECT_EQ(ap_v22->outputs()[0].name, "Y");
  EXPECT_EQ(ap_v22->outputs()[0].type, "T");
  EXPECT_EQ(ap_v22->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(ap_v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(ap_v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);

  EXPECT_EQ(ap_v19->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v11->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v10->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v7->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kFloat);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kDouble);

  EXPECT_FALSE(ap_v1->doc().empty());
  EXPECT_FALSE(ap_v22->doc().empty());
}

} // namespace Test
