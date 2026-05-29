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

constexpr size_t kExpectedConstantSchemaCount = 10;
constexpr size_t kExpectedConstantOfShapeSchemaCount = 6;

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsConstantSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> constant_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("Constant");

  EXPECT_EQ(schemas.size(), kExpectedConstantSchemaCount + kExpectedConstantOfShapeSchemaCount);

  const onnx_op::LightOpSchema *const constant_v25 = FindByVersion(constant_schemas, 25);
  const onnx_op::LightOpSchema *const constant_v24 = FindByVersion(constant_schemas, 24);
  const onnx_op::LightOpSchema *const constant_v23 = FindByVersion(constant_schemas, 23);
  const onnx_op::LightOpSchema *const constant_v21 = FindByVersion(constant_schemas, 21);
  const onnx_op::LightOpSchema *const constant_v19 = FindByVersion(constant_schemas, 19);
  const onnx_op::LightOpSchema *const constant_v13 = FindByVersion(constant_schemas, 13);
  const onnx_op::LightOpSchema *const constant_v12 = FindByVersion(constant_schemas, 12);
  const onnx_op::LightOpSchema *const constant_v11 = FindByVersion(constant_schemas, 11);
  const onnx_op::LightOpSchema *const constant_v9 = FindByVersion(constant_schemas, 9);
  const onnx_op::LightOpSchema *const constant_v1 = FindByVersion(constant_schemas, 1);

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

TEST(OnnxOpGeneratorRegistrationTest, ReturnsConstantOfShapeSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> constant_of_shape_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("ConstantOfShape");

  const onnx_op::LightOpSchema *const cos_v25 = FindByVersion(constant_of_shape_schemas, 25);
  const onnx_op::LightOpSchema *const cos_v24 = FindByVersion(constant_of_shape_schemas, 24);
  const onnx_op::LightOpSchema *const cos_v23 = FindByVersion(constant_of_shape_schemas, 23);
  const onnx_op::LightOpSchema *const cos_v21 = FindByVersion(constant_of_shape_schemas, 21);
  const onnx_op::LightOpSchema *const cos_v20 = FindByVersion(constant_of_shape_schemas, 20);
  const onnx_op::LightOpSchema *const cos_v9 = FindByVersion(constant_of_shape_schemas, 9);

  ASSERT_NE(nullptr, cos_v25);
  ASSERT_NE(nullptr, cos_v24);
  ASSERT_NE(nullptr, cos_v23);
  ASSERT_NE(nullptr, cos_v21);
  ASSERT_NE(nullptr, cos_v20);
  ASSERT_NE(nullptr, cos_v9);

  EXPECT_EQ(cos_v25->domain(), "ai.onnx");
  ASSERT_EQ(cos_v25->inputs().size(), 1u);
  EXPECT_EQ(cos_v25->inputs()[0].name, "input");
  EXPECT_EQ(cos_v25->inputs()[0].type, "T1");
  ASSERT_EQ(cos_v25->outputs().size(), 1u);
  EXPECT_EQ(cos_v25->outputs()[0].name, "output");
  EXPECT_EQ(cos_v25->outputs()[0].type, "T2");

  ASSERT_EQ(cos_v25->type_constraints().size(), 2u);
  EXPECT_EQ(cos_v25->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(cos_v25->type_constraints()[0].allowed_type_strs,
            (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt64}));
  EXPECT_EQ(cos_v25->type_constraints()[1].type_param_str, "T2");

  // Type-count progression (matches upstream ONNX schema history).
  EXPECT_EQ(cos_v9->type_constraints()[1].allowed_type_strs.size(), 12u);
  EXPECT_EQ(cos_v20->type_constraints()[1].allowed_type_strs.size(), 17u);
  EXPECT_EQ(cos_v21->type_constraints()[1].allowed_type_strs.size(), 19u);
  EXPECT_EQ(cos_v23->type_constraints()[1].allowed_type_strs.size(), 20u);
  EXPECT_EQ(cos_v24->type_constraints()[1].allowed_type_strs.size(), 21u);
  EXPECT_EQ(cos_v25->type_constraints()[1].allowed_type_strs.size(), 23u);

  // T2 description switched to "numerics or boolean" at v21.
  EXPECT_EQ(cos_v20->type_constraints()[1].description, "Constrain output types to be numerics.");
  EXPECT_EQ(cos_v21->type_constraints()[1].description,
            "Constrain output types to be numerics or boolean.");

  // ``value`` attribute metadata.
  ASSERT_EQ(cos_v25->attributes().size(), 1u);
  EXPECT_EQ(cos_v25->attributes()[0].name, "value");
  EXPECT_EQ(cos_v25->attributes()[0].type, onnx_op::AttributeType::TENSOR);
  EXPECT_FALSE(cos_v25->attributes()[0].required);

  EXPECT_EQ(cos_v9->doc(),
            R"DOC(
Generate a tensor with given value and shape.
)DOC");
}

} // namespace Test
