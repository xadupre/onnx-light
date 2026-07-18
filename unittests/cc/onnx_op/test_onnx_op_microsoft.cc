// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_ort/operator_sets_microsoft.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
// onnx_lib headers define ONNX_LIGHT_NAMESPACE as a macro alias (onnx_light),
// while onnx_op headers in this target use the literal ONNX_LIGHT_NAMESPACE namespace.
// Undefining keeps this test bound to onnx_op symbols while still using onnx_lib APIs explicitly.
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpMicrosoftRegistrationTest, ReturnsBiasGeluSchemaWithoutFunctionImplementation) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::microsoft::GetAllOnnxOpMicrosoftSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> bias_gelu_schemas =
      onnx_op::microsoft::GetAllOnnxOpMicrosoftSchemasWithHistory("BiasGelu");

  EXPECT_EQ(schemas.size(), 2u);

  const onnx_op::LightOpSchema *const bias_gelu_v1 = FindByVersion(bias_gelu_schemas, 1);
  ASSERT_NE(nullptr, bias_gelu_v1);
  EXPECT_EQ(bias_gelu_v1->domain(), "com.microsoft");
  EXPECT_EQ(bias_gelu_v1->domain(), onnx_op::microsoft::kMicrosoftDomain);
  EXPECT_FALSE(bias_gelu_v1->has_function_implementation());

  ASSERT_EQ(bias_gelu_v1->inputs().size(), 2u);
  EXPECT_EQ(bias_gelu_v1->inputs()[0].name, "A");
  EXPECT_EQ(bias_gelu_v1->inputs()[1].name, "B");
  EXPECT_EQ(bias_gelu_v1->inputs()[0].type, "T");
  EXPECT_EQ(bias_gelu_v1->inputs()[1].type, "T");

  ASSERT_EQ(bias_gelu_v1->outputs().size(), 1u);
  EXPECT_EQ(bias_gelu_v1->outputs()[0].name, "C");
  EXPECT_EQ(bias_gelu_v1->outputs()[0].type, "T");

  ASSERT_EQ(bias_gelu_v1->type_constraints().size(), 1u);
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].description,
            "Constrain input and output types to float tensors.");
  ASSERT_EQ(bias_gelu_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kFloat16);
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kFloat);
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kDouble);
  EXPECT_EQ(bias_gelu_v1->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kBfloat16);

  EXPECT_NE(bias_gelu_v1->doc().find("Bias Gelu"), std::string::npos);
}

TEST(OnnxOpMicrosoftRegistrationTest, ReturnsBiasGeluGradDxSchemaWithoutFunctionImplementation) {
  const std::vector<onnx_op::LightOpSchema> bias_gelu_grad_dx_schemas =
      onnx_op::microsoft::GetAllOnnxOpMicrosoftSchemasWithHistory("BiasGeluGrad_dX");

  const onnx_op::LightOpSchema *const bias_gelu_grad_dx_v1 =
      FindByVersion(bias_gelu_grad_dx_schemas, 1);
  ASSERT_NE(nullptr, bias_gelu_grad_dx_v1);
  EXPECT_EQ(bias_gelu_grad_dx_v1->domain(), onnx_op::microsoft::kMicrosoftDomain);
  EXPECT_FALSE(bias_gelu_grad_dx_v1->has_function_implementation());

  ASSERT_EQ(bias_gelu_grad_dx_v1->inputs().size(), 3u);
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[0].name, "dY");
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[1].name, "X");
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[2].name, "B");
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[0].type, "T");
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[1].type, "T");
  EXPECT_EQ(bias_gelu_grad_dx_v1->inputs()[2].type, "T");

  ASSERT_EQ(bias_gelu_grad_dx_v1->outputs().size(), 1u);
  EXPECT_EQ(bias_gelu_grad_dx_v1->outputs()[0].name, "dX");
  EXPECT_EQ(bias_gelu_grad_dx_v1->outputs()[0].type, "T");

  ASSERT_EQ(bias_gelu_grad_dx_v1->type_constraints().size(), 1u);
  EXPECT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].type_param_str, "T");
  ASSERT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].allowed_type_strs[0],
            onnx_op::TensorType::kFloat16);
  EXPECT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].allowed_type_strs[1],
            onnx_op::TensorType::kFloat);
  EXPECT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].allowed_type_strs[2],
            onnx_op::TensorType::kDouble);
  EXPECT_EQ(bias_gelu_grad_dx_v1->type_constraints()[0].allowed_type_strs[3],
            onnx_op::TensorType::kBfloat16);

  EXPECT_NE(bias_gelu_grad_dx_v1->doc().find("Computes ``dX`` for BiasGelu"), std::string::npos);
}

} // namespace Test
