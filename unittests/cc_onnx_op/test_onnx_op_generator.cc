// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_generator.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

const onnx_op::LightOpSchema *
FindGeneratorSchema(const std::vector<onnx_op::LightOpSchema> &schemas, const std::string &op_type,
                    int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsConstantSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 10u);

  const onnx_op::LightOpSchema *const constant_v25 = FindGeneratorSchema(schemas, "Constant", 25);
  const onnx_op::LightOpSchema *const constant_v24 = FindGeneratorSchema(schemas, "Constant", 24);
  const onnx_op::LightOpSchema *const constant_v23 = FindGeneratorSchema(schemas, "Constant", 23);
  const onnx_op::LightOpSchema *const constant_v21 = FindGeneratorSchema(schemas, "Constant", 21);
  const onnx_op::LightOpSchema *const constant_v19 = FindGeneratorSchema(schemas, "Constant", 19);
  const onnx_op::LightOpSchema *const constant_v13 = FindGeneratorSchema(schemas, "Constant", 13);
  const onnx_op::LightOpSchema *const constant_v12 = FindGeneratorSchema(schemas, "Constant", 12);
  const onnx_op::LightOpSchema *const constant_v11 = FindGeneratorSchema(schemas, "Constant", 11);
  const onnx_op::LightOpSchema *const constant_v9 = FindGeneratorSchema(schemas, "Constant", 9);
  const onnx_op::LightOpSchema *const constant_v1 = FindGeneratorSchema(schemas, "Constant", 1);

  ASSERT_NE(nullptr, constant_v25);
  ASSERT_NE(nullptr, constant_v24);
  ASSERT_NE(nullptr, constant_v23);
  ASSERT_NE(nullptr, constant_v21);
  ASSERT_NE(nullptr, constant_v19);
  ASSERT_NE(nullptr, constant_v13);
  ASSERT_NE(nullptr, constant_v12);
  ASSERT_NE(nullptr, constant_v11);
  ASSERT_NE(nullptr, constant_v9);
  ASSERT_NE(nullptr, constant_v1);

  EXPECT_EQ(constant_v25->domain(), "ai.onnx");
  EXPECT_EQ(constant_v25->inputs().size(), 0u);
  EXPECT_EQ(constant_v25->outputs().size(), 1u);
  EXPECT_EQ(constant_v25->type_constraints().size(), 1u);
  EXPECT_EQ(constant_v25->outputs()[0].name, "output");
  EXPECT_EQ(constant_v25->outputs()[0].description,
            "Output tensor containing the same value of the provided tensor.");
  EXPECT_EQ(constant_v1->type_constraints()[0].allowed_type_strs, onnx_op::FloatTypes());
  EXPECT_EQ(constant_v9->type_constraints()[0].allowed_type_strs.size(), 15u);
  EXPECT_EQ(constant_v13->type_constraints()[0].allowed_type_strs.size(), 16u);
  EXPECT_EQ(constant_v19->type_constraints()[0].allowed_type_strs.size(), 20u);
  EXPECT_EQ(constant_v21->type_constraints()[0].allowed_type_strs.size(), 22u);
  EXPECT_EQ(constant_v23->type_constraints()[0].allowed_type_strs.size(), 23u);
  EXPECT_EQ(constant_v24->type_constraints()[0].allowed_type_strs.size(), 24u);
  EXPECT_EQ(constant_v25->type_constraints()[0].allowed_type_strs.size(), 26u);
  EXPECT_EQ(constant_v25->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kInt2);
  EXPECT_EQ(constant_v24->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kFloat8e8m0);
  EXPECT_EQ(constant_v23->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kFloat4e2m1);
  EXPECT_EQ(constant_v21->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kInt4);
  EXPECT_EQ(constant_v19->type_constraints()[0].allowed_type_strs.back(),
            onnx_op::TensorType::kFloat8e5m2fnuz);
  EXPECT_EQ(constant_v11->doc(),
            R"DOC(
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
)DOC");
  EXPECT_EQ(constant_v9->doc(), "A constant tensor.");
  EXPECT_EQ(constant_v25->doc(),
            R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC");
}

} // namespace Test
