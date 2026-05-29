// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_training.h"

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

TEST(OnnxOpTrainingRegistrationTest, ReturnsGradientSchemaWithoutFunctionImplementation) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::training::GetAllOnnxOpTrainingSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> gradient_schemas =
      onnx_op::training::GetAllOnnxOpTrainingSchemasWithHistory("Gradient");

  EXPECT_EQ(schemas.size(), 2u);

  const onnx_op::LightOpSchema *const gradient_v1 = FindByVersion(gradient_schemas, 1);
  ASSERT_NE(nullptr, gradient_v1);
  EXPECT_EQ(gradient_v1->domain(), "ai.onnx.preview.training");
  EXPECT_EQ(gradient_v1->domain(), onnx_op::training::kOnnxPreviewTrainingDomain);
  EXPECT_FALSE(gradient_v1->has_function_implementation());

  ASSERT_EQ(gradient_v1->inputs().size(), 1u);
  EXPECT_EQ(gradient_v1->inputs()[0].name, "Inputs");
  EXPECT_EQ(gradient_v1->inputs()[0].type, "T1");

  ASSERT_EQ(gradient_v1->outputs().size(), 1u);
  EXPECT_EQ(gradient_v1->outputs()[0].name, "Outputs");
  EXPECT_EQ(gradient_v1->outputs()[0].type, "T2");

  ASSERT_EQ(gradient_v1->type_constraints().size(), 2u);
  EXPECT_EQ(gradient_v1->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(gradient_v1->type_constraints()[0].description,
            "Allow outputs to be any kind of tensor.");
  EXPECT_EQ(gradient_v1->type_constraints()[0].allowed_type_strs.size(),
            onnx_op::AllTensorTypes().size());

  EXPECT_EQ(gradient_v1->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(gradient_v1->type_constraints()[1].description,
            "Allow inputs to be any kind of floating-point tensor.");
  ASSERT_EQ(gradient_v1->type_constraints()[1].allowed_type_strs.size(), 3u);
  EXPECT_EQ(gradient_v1->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kFloat16);
  EXPECT_EQ(gradient_v1->type_constraints()[1].allowed_type_strs[1], onnx_op::TensorType::kFloat);
  EXPECT_EQ(gradient_v1->type_constraints()[1].allowed_type_strs[2], onnx_op::TensorType::kDouble);

  EXPECT_NE(gradient_v1->doc().find("Gradient operator computes the partial derivatives"),
            std::string::npos);
}

TEST(OnnxOpTrainingRegistrationTest, ReturnsAdamSchemaWithoutFunctionImplementation) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::training::GetAllOnnxOpTrainingSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> adam_schemas =
      onnx_op::training::GetAllOnnxOpTrainingSchemasWithHistory("Adam");

  const onnx_op::LightOpSchema *const adam_v1 = FindByVersion(adam_schemas, 1);
  ASSERT_NE(nullptr, adam_v1);
  EXPECT_EQ(adam_v1->domain(), "ai.onnx.preview.training");
  EXPECT_EQ(adam_v1->domain(), onnx_op::training::kOnnxPreviewTrainingDomain);
  EXPECT_FALSE(adam_v1->has_function_implementation());

  ASSERT_EQ(adam_v1->inputs().size(), 3u);
  EXPECT_EQ(adam_v1->inputs()[0].name, "R");
  EXPECT_EQ(adam_v1->inputs()[0].type, "T1");
  EXPECT_EQ(adam_v1->inputs()[1].name, "T");
  EXPECT_EQ(adam_v1->inputs()[1].type, "T2");
  EXPECT_EQ(adam_v1->inputs()[2].name, "inputs");
  EXPECT_EQ(adam_v1->inputs()[2].type, "T3");

  ASSERT_EQ(adam_v1->outputs().size(), 1u);
  EXPECT_EQ(adam_v1->outputs()[0].name, "outputs");
  EXPECT_EQ(adam_v1->outputs()[0].type, "T3");

  ASSERT_EQ(adam_v1->type_constraints().size(), 3u);
  EXPECT_EQ(adam_v1->type_constraints()[0].type_param_str, "T1");
  ASSERT_EQ(adam_v1->type_constraints()[0].allowed_type_strs.size(), 2u);
  EXPECT_EQ(adam_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(adam_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kDouble);

  EXPECT_EQ(adam_v1->type_constraints()[1].type_param_str, "T2");
  ASSERT_EQ(adam_v1->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(adam_v1->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kInt64);

  EXPECT_EQ(adam_v1->type_constraints()[2].type_param_str, "T3");
  ASSERT_EQ(adam_v1->type_constraints()[2].allowed_type_strs.size(), 2u);
  EXPECT_EQ(adam_v1->type_constraints()[2].allowed_type_strs[0], onnx_op::TensorType::kFloat);
  EXPECT_EQ(adam_v1->type_constraints()[2].allowed_type_strs[1], onnx_op::TensorType::kDouble);

  EXPECT_NE(adam_v1->doc().find("Compute one iteration of Adam"), std::string::npos);
}

} // namespace Test
