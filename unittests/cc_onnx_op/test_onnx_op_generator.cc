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

constexpr size_t kExpectedBernoulliSchemaCount = 2;
constexpr size_t kExpectedConstantSchemaCount = 10;
constexpr size_t kExpectedConstantOfShapeSchemaCount = 6;
constexpr size_t kExpectedEyeLikeSchemaCount = 2;
constexpr size_t kExpectedRandomNormalSchemaCount = 2;
constexpr size_t kExpectedRandomNormalLikeSchemaCount = 2;
constexpr size_t kExpectedRandomUniformSchemaCount = 2;
constexpr size_t kExpectedRandomUniformLikeSchemaCount = 2;
constexpr size_t kExpectedRangeSchemaCount = 1;
constexpr size_t kExpectedMultinomialSchemaCount = 2;

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

  EXPECT_EQ(schemas.size(),
            kExpectedConstantSchemaCount + kExpectedConstantOfShapeSchemaCount +
                kExpectedEyeLikeSchemaCount + kExpectedBernoulliSchemaCount +
                kExpectedRandomNormalSchemaCount + kExpectedRandomNormalLikeSchemaCount +
                kExpectedRandomUniformSchemaCount + kExpectedRandomUniformLikeSchemaCount +
                kExpectedRangeSchemaCount + kExpectedMultinomialSchemaCount);

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

TEST(OnnxOpGeneratorRegistrationTest, ReturnsBernoulliSchemas) {
  const std::vector<onnx_op::LightOpSchema> bernoulli_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("Bernoulli");

  ASSERT_EQ(bernoulli_schemas.size(), kExpectedBernoulliSchemaCount);

  const onnx_op::LightOpSchema *const bernoulli_v22 = FindByVersion(bernoulli_schemas, 22);
  const onnx_op::LightOpSchema *const bernoulli_v15 = FindByVersion(bernoulli_schemas, 15);
  ASSERT_NE(nullptr, bernoulli_v22);
  ASSERT_NE(nullptr, bernoulli_v15);

  for (const onnx_op::LightOpSchema *schema : {bernoulli_v22, bernoulli_v15}) {
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    EXPECT_EQ(schema->outputs()[0].type, "T2");
    ASSERT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(schema->type_constraints()[1].type_param_str, "T2");
    EXPECT_EQ(schema->type_constraints()[0].description, "Constrain input types to float tensors.");
    EXPECT_EQ(schema->type_constraints()[1].description,
              "Constrain output types to all numeric tensors and bool tensors.");

    ASSERT_EQ(schema->attributes().size(), 2u);
    EXPECT_EQ(schema->attributes()[0].name, "seed");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::FLOAT);
    EXPECT_FALSE(schema->attributes()[0].required);
    EXPECT_EQ(schema->attributes()[1].name, "dtype");
    EXPECT_EQ(schema->attributes()[1].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[1].required);
  }

  // v22 added bfloat16 to the T1 constraint.
  EXPECT_EQ(bernoulli_v15->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(bernoulli_v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(bernoulli_v22->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kBfloat16);

  // T2 size is 13 in both versions.
  EXPECT_EQ(bernoulli_v15->type_constraints()[1].allowed_type_strs.size(), 13u);
  EXPECT_EQ(bernoulli_v22->type_constraints()[1].allowed_type_strs.size(), 13u);
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsEyeLikeSchemas) {
  const std::vector<onnx_op::LightOpSchema> eye_like_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("EyeLike");
  ASSERT_EQ(eye_like_schemas.size(), kExpectedEyeLikeSchemaCount);

  const onnx_op::LightOpSchema *const eye_like_v22 = FindByVersion(eye_like_schemas, 22);
  const onnx_op::LightOpSchema *const eye_like_v9 = FindByVersion(eye_like_schemas, 9);
  ASSERT_NE(nullptr, eye_like_v22);
  ASSERT_NE(nullptr, eye_like_v9);

  for (const onnx_op::LightOpSchema *schema : {eye_like_v22, eye_like_v9}) {
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    EXPECT_EQ(schema->outputs()[0].type, "T2");
    ASSERT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].description,
              "Constrain input types. Strings and complex are not supported.");
    EXPECT_EQ(schema->type_constraints()[1].description,
              "Constrain output types. Strings and complex are not supported.");
    ASSERT_EQ(schema->attributes().size(), 2u);
    EXPECT_EQ(schema->attributes()[0].name, "k");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::INT);
    EXPECT_EQ(schema->attributes()[1].name, "dtype");
    EXPECT_EQ(schema->attributes()[1].type, onnx_op::AttributeType::INT);
  }

  EXPECT_EQ(eye_like_v9->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(eye_like_v22->type_constraints()[0].allowed_type_strs.size(), 13u);
  EXPECT_EQ(eye_like_v22->type_constraints()[0].allowed_type_strs[8],
            onnx_op::TensorType::kBfloat16);
}

namespace {

// Common per-version invariants for the four Random* schemas.
void CheckRandomSchemaInvariants(const onnx_op::LightOpSchema *schema, bool is_like,
                                 size_t expected_attr_count) {
  ASSERT_NE(nullptr, schema);
  EXPECT_EQ(schema->domain(), "ai.onnx");
  ASSERT_EQ(schema->outputs().size(), 1u);
  EXPECT_EQ(schema->outputs()[0].name, "output");
  if (is_like) {
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    EXPECT_EQ(schema->outputs()[0].type, "T2");
    ASSERT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(schema->type_constraints()[1].type_param_str, "T2");
  } else {
    EXPECT_EQ(schema->inputs().size(), 0u);
    EXPECT_EQ(schema->outputs()[0].type, "T");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
  }
  ASSERT_EQ(schema->attributes().size(), expected_attr_count);
}

} // namespace

TEST(OnnxOpGeneratorRegistrationTest, ReturnsRandomNormalSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("RandomNormal");
  ASSERT_EQ(schemas.size(), kExpectedRandomNormalSchemaCount);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  CheckRandomSchemaInvariants(v22, /*is_like=*/false, /*expected_attr_count=*/5);
  CheckRandomSchemaInvariants(v1, /*is_like=*/false, /*expected_attr_count=*/5);

  // v22 adds bfloat16 to T.
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);

  // Attribute order: mean, scale, seed, dtype, shape.
  EXPECT_EQ(v22->attributes()[0].name, "mean");
  EXPECT_EQ(v22->attributes()[0].type, onnx_op::AttributeType::FLOAT);
  EXPECT_EQ(v22->attributes()[1].name, "scale");
  EXPECT_EQ(v22->attributes()[2].name, "seed");
  EXPECT_EQ(v22->attributes()[3].name, "dtype");
  EXPECT_EQ(v22->attributes()[4].name, "shape");
  EXPECT_EQ(v22->attributes()[4].type, onnx_op::AttributeType::INTS);
  EXPECT_TRUE(v22->attributes()[4].required);
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsRandomUniformSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("RandomUniform");
  ASSERT_EQ(schemas.size(), kExpectedRandomUniformSchemaCount);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  CheckRandomSchemaInvariants(v22, /*is_like=*/false, /*expected_attr_count=*/5);
  CheckRandomSchemaInvariants(v1, /*is_like=*/false, /*expected_attr_count=*/5);

  EXPECT_EQ(v22->attributes()[0].name, "low");
  EXPECT_EQ(v22->attributes()[1].name, "high");
  EXPECT_EQ(v22->attributes()[2].name, "seed");
  EXPECT_EQ(v22->attributes()[3].name, "dtype");
  EXPECT_EQ(v22->attributes()[4].name, "shape");
  EXPECT_TRUE(v22->attributes()[4].required);
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsRandomNormalLikeSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("RandomNormalLike");
  ASSERT_EQ(schemas.size(), kExpectedRandomNormalLikeSchemaCount);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  CheckRandomSchemaInvariants(v22, /*is_like=*/true, /*expected_attr_count=*/4);
  CheckRandomSchemaInvariants(v1, /*is_like=*/true, /*expected_attr_count=*/4);

  // v22 adds bfloat16 to T1 and T2.
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 15u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 16u);
  EXPECT_EQ(v1->type_constraints()[1].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v22->type_constraints()[1].allowed_type_strs.size(), 4u);

  EXPECT_EQ(v22->attributes()[0].name, "mean");
  EXPECT_EQ(v22->attributes()[1].name, "scale");
  EXPECT_EQ(v22->attributes()[2].name, "seed");
  EXPECT_EQ(v22->attributes()[3].name, "dtype");
  EXPECT_FALSE(v22->attributes()[3].required);
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsRandomUniformLikeSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("RandomUniformLike");
  ASSERT_EQ(schemas.size(), kExpectedRandomUniformLikeSchemaCount);
  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  CheckRandomSchemaInvariants(v22, /*is_like=*/true, /*expected_attr_count=*/4);
  CheckRandomSchemaInvariants(v1, /*is_like=*/true, /*expected_attr_count=*/4);

  EXPECT_EQ(v22->attributes()[0].name, "low");
  EXPECT_EQ(v22->attributes()[1].name, "high");
  EXPECT_EQ(v22->attributes()[2].name, "seed");
  EXPECT_EQ(v22->attributes()[3].name, "dtype");
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsRangeSchemas) {
  const std::vector<onnx_op::LightOpSchema> range_schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("Range");
  ASSERT_EQ(range_schemas.size(), kExpectedRangeSchemaCount);

  const onnx_op::LightOpSchema *const range_v11 = FindByVersion(range_schemas, 11);
  ASSERT_NE(nullptr, range_v11);

  EXPECT_EQ(range_v11->domain(), "ai.onnx");
  ASSERT_EQ(range_v11->inputs().size(), 3u);
  EXPECT_EQ(range_v11->inputs()[0].name, "start");
  EXPECT_EQ(range_v11->inputs()[0].type, "T");
  EXPECT_EQ(range_v11->inputs()[1].name, "limit");
  EXPECT_EQ(range_v11->inputs()[1].type, "T");
  EXPECT_EQ(range_v11->inputs()[2].name, "delta");
  EXPECT_EQ(range_v11->inputs()[2].type, "T");

  ASSERT_EQ(range_v11->outputs().size(), 1u);
  EXPECT_EQ(range_v11->outputs()[0].name, "output");
  EXPECT_EQ(range_v11->outputs()[0].type, "T");

  ASSERT_EQ(range_v11->type_constraints().size(), 1u);
  EXPECT_EQ(range_v11->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(
      range_v11->type_constraints()[0].allowed_type_strs,
      (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kFloat, onnx_op::TensorType::kDouble,
                                        onnx_op::TensorType::kInt16, onnx_op::TensorType::kInt32,
                                        onnx_op::TensorType::kInt64}));
  EXPECT_EQ(range_v11->type_constraints()[0].description,
            "Constrain input types to common numeric type tensors.");

  EXPECT_EQ(range_v11->attributes().size(), 0u);
}

TEST(OnnxOpGeneratorRegistrationTest, ReturnsMultinomialSchemas) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::generator::GetAllOnnxOpGeneratorSchemasWithHistory("Multinomial");
  ASSERT_EQ(schemas.size(), kExpectedMultinomialSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v7 = FindByVersion(schemas, 7);
  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v7);

  for (const onnx_op::LightOpSchema *schema : {v22, v7}) {
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    EXPECT_EQ(schema->outputs()[0].type, "T2");

    ASSERT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(schema->type_constraints()[1].type_param_str, "T2");
    EXPECT_EQ(schema->type_constraints()[0].description, "Constrain input types to float tensors.");
    EXPECT_EQ(schema->type_constraints()[1].description,
              "Constrain output types to integral tensors.");
    EXPECT_EQ(schema->type_constraints()[1].allowed_type_strs,
              (std::vector<onnx_op::TensorType>{onnx_op::TensorType::kInt32,
                                                onnx_op::TensorType::kInt64}));

    ASSERT_EQ(schema->attributes().size(), 3u);
    EXPECT_EQ(schema->attributes()[0].name, "sample_size");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[0].required);
    EXPECT_EQ(schema->attributes()[1].name, "seed");
    EXPECT_EQ(schema->attributes()[1].type, onnx_op::AttributeType::FLOAT);
    EXPECT_FALSE(schema->attributes()[1].required);
    EXPECT_EQ(schema->attributes()[2].name, "dtype");
    EXPECT_EQ(schema->attributes()[2].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[2].required);
  }

  // v22 adds bfloat16 to the T1 constraint.
  EXPECT_EQ(v7->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
}

} // namespace Test
