// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_quantization.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedQuantizeLinearSchemaCount = 7;
constexpr size_t kExpectedDequantizeLinearSchemaCount = 7;
constexpr size_t kExpectedQLinearConvSchemaCount = 1;
constexpr size_t kExpectedQLinearMatMulSchemaCount = 2;
constexpr size_t kExpectedQuantizationSchemaCount =
    kExpectedQuantizeLinearSchemaCount + kExpectedDequantizeLinearSchemaCount +
    kExpectedQLinearConvSchemaCount + kExpectedQLinearMatMulSchemaCount;

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpQuantizationRegistrationTest, ReturnsQuantizeLinearSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::quantization::GetAllOnnxOpQuantizationSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> quantize_linear_schemas =
      onnx_op::quantization::GetAllOnnxOpQuantizationSchemasWithHistory("QuantizeLinear");

  EXPECT_EQ(schemas.size(), kExpectedQuantizationSchemaCount);

  const onnx_op::LightOpSchema *const v25 = FindByVersion(quantize_linear_schemas, 25);
  const onnx_op::LightOpSchema *const v24 = FindByVersion(quantize_linear_schemas, 24);
  const onnx_op::LightOpSchema *const v23 = FindByVersion(quantize_linear_schemas, 23);
  const onnx_op::LightOpSchema *const v21 = FindByVersion(quantize_linear_schemas, 21);
  const onnx_op::LightOpSchema *const v19 = FindByVersion(quantize_linear_schemas, 19);
  const onnx_op::LightOpSchema *const v13 = FindByVersion(quantize_linear_schemas, 13);
  const onnx_op::LightOpSchema *const v10 = FindByVersion(quantize_linear_schemas, 10);

  ASSERT_NE(nullptr, v25);
  ASSERT_NE(nullptr, v24);
  ASSERT_NE(nullptr, v23);
  ASSERT_NE(nullptr, v21);
  ASSERT_NE(nullptr, v19);
  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v10);

  EXPECT_EQ(v25->domain(), "ai.onnx");
  EXPECT_FALSE(v25->has_function_implementation());

  // QuantizeLinear has always had 3 inputs (x, y_scale, y_zero_point) and 1 output (y).
  ASSERT_EQ(v25->inputs().size(), 3u);
  EXPECT_EQ(v25->inputs()[0].name, "x");
  EXPECT_EQ(v25->inputs()[1].name, "y_scale");
  EXPECT_EQ(v25->inputs()[2].name, "y_zero_point");
  ASSERT_EQ(v25->outputs().size(), 1u);
  EXPECT_EQ(v25->outputs()[0].name, "y");

  ASSERT_EQ(v10->inputs().size(), 3u);
  EXPECT_EQ(v10->inputs()[0].name, "x");
  EXPECT_EQ(v10->inputs()[1].name, "y_scale");
  EXPECT_EQ(v10->inputs()[2].name, "y_zero_point");
  ASSERT_EQ(v10->outputs().size(), 1u);
  EXPECT_EQ(v10->outputs()[0].name, "y");

  // v25 introduced the T3 constraint with int2/uint2 in addition to v24.
  ASSERT_EQ(v25->type_constraints().size(), 3u);
  EXPECT_EQ(v25->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(v25->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(v25->type_constraints()[2].type_param_str, "T3");
  EXPECT_EQ(v25->type_constraints()[2].allowed_type_strs.size(), 13u);
  EXPECT_EQ(v25->type_constraints()[2].allowed_type_strs.back(), onnx_op::TensorType::kInt2);

  // v24 added float8e8m0 to T2 and the float4e2m1 entry to T3.
  ASSERT_EQ(v24->type_constraints().size(), 3u);
  EXPECT_EQ(v24->type_constraints()[1].allowed_type_strs.size(), 5u);
  EXPECT_EQ(v24->type_constraints()[1].allowed_type_strs.back(), onnx_op::TensorType::kFloat8e8m0);
  EXPECT_EQ(v24->type_constraints()[2].allowed_type_strs.size(), 11u);
  EXPECT_EQ(v24->type_constraints()[2].allowed_type_strs.back(), onnx_op::TensorType::kFloat4e2m1);

  // v23 has T2 without float8e8m0 but T3 with float4e2m1.
  ASSERT_EQ(v23->type_constraints().size(), 3u);
  EXPECT_EQ(v23->type_constraints()[1].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v23->type_constraints()[2].allowed_type_strs.size(), 11u);

  // v21 still used T1 for y_scale (only two type constraints).
  ASSERT_EQ(v21->type_constraints().size(), 2u);
  EXPECT_EQ(v21->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(v21->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(v21->inputs()[1].type, "T1");
  EXPECT_EQ(v21->inputs()[2].type, "T2");

  // v19: y_scale uses T1; T2 has float8 variants but no int16/uint16/4bit types.
  ASSERT_EQ(v19->type_constraints().size(), 2u);
  EXPECT_EQ(v19->type_constraints()[1].allowed_type_strs.size(), 6u);

  // v13: y_scale type is the literal "tensor(float)", T1/T2 restricted to legacy types.
  ASSERT_EQ(v13->type_constraints().size(), 2u);
  EXPECT_EQ(v13->inputs()[1].type, "tensor(float)");
  EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs.size(), 2u);
  EXPECT_EQ(v13->type_constraints()[1].allowed_type_strs.size(), 2u);

  // v10: legacy per-tensor only schema with T1/T2 of two entries each.
  ASSERT_EQ(v10->type_constraints().size(), 2u);
  EXPECT_EQ(v10->inputs()[1].type, "tensor(float)");
  EXPECT_EQ(v10->type_constraints()[0].allowed_type_strs.size(), 2u);
  EXPECT_EQ(v10->type_constraints()[1].allowed_type_strs.size(), 2u);

  EXPECT_NE(v25->doc().find("linear quantization operator"), std::string::npos);
  EXPECT_NE(v25->doc().find("int2"), std::string::npos);
  EXPECT_EQ(v24->doc().find("int2"), std::string::npos);
  EXPECT_NE(v10->doc().find("per-tensor/layer quantization"), std::string::npos);
}

TEST(OnnxOpQuantizationRegistrationTest, ReturnsDequantizeLinearSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> dequantize_linear_schemas =
      onnx_op::quantization::GetAllOnnxOpQuantizationSchemasWithHistory("DequantizeLinear");

  EXPECT_EQ(dequantize_linear_schemas.size(), kExpectedDequantizeLinearSchemaCount);

  const onnx_op::LightOpSchema *const v25 = FindByVersion(dequantize_linear_schemas, 25);
  const onnx_op::LightOpSchema *const v24 = FindByVersion(dequantize_linear_schemas, 24);
  const onnx_op::LightOpSchema *const v23 = FindByVersion(dequantize_linear_schemas, 23);
  const onnx_op::LightOpSchema *const v21 = FindByVersion(dequantize_linear_schemas, 21);
  const onnx_op::LightOpSchema *const v19 = FindByVersion(dequantize_linear_schemas, 19);
  const onnx_op::LightOpSchema *const v13 = FindByVersion(dequantize_linear_schemas, 13);
  const onnx_op::LightOpSchema *const v10 = FindByVersion(dequantize_linear_schemas, 10);

  ASSERT_NE(nullptr, v25);
  ASSERT_NE(nullptr, v24);
  ASSERT_NE(nullptr, v23);
  ASSERT_NE(nullptr, v21);
  ASSERT_NE(nullptr, v19);
  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v10);

  EXPECT_EQ(v25->domain(), "ai.onnx");
  EXPECT_FALSE(v25->has_function_implementation());

  // DequantizeLinear has always had 3 inputs (x, x_scale, x_zero_point) and 1 output (y).
  ASSERT_EQ(v25->inputs().size(), 3u);
  EXPECT_EQ(v25->inputs()[0].name, "x");
  EXPECT_EQ(v25->inputs()[1].name, "x_scale");
  EXPECT_EQ(v25->inputs()[2].name, "x_zero_point");
  ASSERT_EQ(v25->outputs().size(), 1u);
  EXPECT_EQ(v25->outputs()[0].name, "y");

  ASSERT_EQ(v10->inputs().size(), 3u);
  EXPECT_EQ(v10->inputs()[0].name, "x");
  EXPECT_EQ(v10->inputs()[1].name, "x_scale");
  EXPECT_EQ(v10->inputs()[2].name, "x_zero_point");
  ASSERT_EQ(v10->outputs().size(), 1u);
  EXPECT_EQ(v10->outputs()[0].name, "y");

  // v25 introduces int2/uint2 into T1 (14 entries) on top of v24 (12 entries).
  ASSERT_EQ(v25->type_constraints().size(), 3u);
  EXPECT_EQ(v25->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(v25->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(v25->type_constraints()[2].type_param_str, "T3");
  EXPECT_EQ(v25->type_constraints()[0].allowed_type_strs.size(), 14u);
  EXPECT_EQ(v25->type_constraints()[0].allowed_type_strs.back(), onnx_op::TensorType::kInt2);

  // v24 introduced T3 + the float8e8m0 entry on T2 vs. v23.
  ASSERT_EQ(v24->type_constraints().size(), 3u);
  EXPECT_EQ(v24->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(v24->type_constraints()[1].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v24->type_constraints()[1].allowed_type_strs.back(), onnx_op::TensorType::kFloat8e8m0);

  // v23 has T2 without float8e8m0; T1 still has 12 entries.
  ASSERT_EQ(v23->type_constraints().size(), 3u);
  EXPECT_EQ(v23->type_constraints()[0].allowed_type_strs.size(), 12u);
  EXPECT_EQ(v23->type_constraints()[1].allowed_type_strs.size(), 3u);

  // v21: output type is T2 (no T3), T1 lacks float4e2m1.
  ASSERT_EQ(v21->type_constraints().size(), 2u);
  EXPECT_EQ(v21->type_constraints()[0].allowed_type_strs.size(), 11u);
  EXPECT_EQ(v21->outputs()[0].type, "T2");

  // v19: T1 has only legacy 8-bit/int32/float8 variants (7 entries).
  ASSERT_EQ(v19->type_constraints().size(), 2u);
  EXPECT_EQ(v19->type_constraints()[0].allowed_type_strs.size(), 7u);

  // v13: single type constraint T with legacy 8-bit/int32 types; x_scale and y
  // use the literal tensor(float) type.
  ASSERT_EQ(v13->type_constraints().size(), 1u);
  EXPECT_EQ(v13->inputs()[1].type, "tensor(float)");
  EXPECT_EQ(v13->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs.size(), 3u);

  // v10: legacy per-tensor only schema, single T constraint with 3 entries.
  ASSERT_EQ(v10->type_constraints().size(), 1u);
  EXPECT_EQ(v10->inputs()[1].type, "tensor(float)");
  EXPECT_EQ(v10->outputs()[0].type, "tensor(float)");
  EXPECT_EQ(v10->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_NE(v25->doc().find("linear dequantization operator"), std::string::npos);
  EXPECT_NE(v25->doc().find("output_dtype"), std::string::npos);
  EXPECT_NE(v10->doc().find("linear dequantization operator"), std::string::npos);
}

} // namespace Test
