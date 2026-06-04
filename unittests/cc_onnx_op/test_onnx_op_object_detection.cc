// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_object_detection.h"

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

TEST(OnnxOpObjectDetectionRegistrationTest, ReturnsRoiAlignSchemaHistory) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::object_detection::GetAllOnnxOpObjectDetectionSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> roi_align_schemas =
      onnx_op::object_detection::GetAllOnnxOpObjectDetectionSchemasWithHistory("RoiAlign");

  EXPECT_EQ(schemas.size(), 5u);

  const std::vector<int> expected_versions = {22, 16, 10};
  for (int version : expected_versions) {
    SCOPED_TRACE("RoiAlign@" + std::to_string(version));
    const onnx_op::LightOpSchema *const schema = FindByVersion(roi_align_schemas, version);
    ASSERT_NE(nullptr, schema);
    EXPECT_EQ(schema->name(), "RoiAlign");
    EXPECT_EQ(schema->domain(), onnx_op::kOnnxDomain);
    EXPECT_EQ(schema->since_version(), version);

    ASSERT_EQ(schema->inputs().size(), 3u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[0].type, "T1");
    EXPECT_EQ(schema->inputs()[1].name, "rois");
    EXPECT_EQ(schema->inputs()[1].type, "T1");
    EXPECT_EQ(schema->inputs()[2].name, "batch_indices");
    EXPECT_EQ(schema->inputs()[2].type, "T2");

    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    EXPECT_EQ(schema->outputs()[0].type, "T1");

    ASSERT_EQ(schema->type_constraints().size(), 2u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs.size(), 4u);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kFloat16);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kFloat);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs[3], onnx_op::TensorType::kDouble);
    EXPECT_EQ(schema->type_constraints()[1].type_param_str, "T2");
    ASSERT_EQ(schema->type_constraints()[1].allowed_type_strs.size(), 1u);
    EXPECT_EQ(schema->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kInt64);
    EXPECT_STREQ(onnx_op::ToTypeString(schema->type_constraints()[1].allowed_type_strs[0]),
                 "tensor(int64)");

    EXPECT_FALSE(schema->doc().empty());
  }
}

TEST(OnnxOpObjectDetectionRegistrationTest, ReturnsNonMaxSuppressionSchemaHistory) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::object_detection::GetAllOnnxOpObjectDetectionSchemasWithHistory("NonMaxSuppression");
  ASSERT_EQ(schemas.size(), 2u);

  const std::vector<int> expected_versions = {11, 10};
  for (int version : expected_versions) {
    SCOPED_TRACE("NonMaxSuppression@" + std::to_string(version));
    const onnx_op::LightOpSchema *const schema = FindByVersion(schemas, version);
    ASSERT_NE(nullptr, schema);
    EXPECT_EQ(schema->name(), "NonMaxSuppression");
    EXPECT_EQ(schema->domain(), onnx_op::kOnnxDomain);
    EXPECT_EQ(schema->since_version(), version);

    ASSERT_EQ(schema->inputs().size(), 5u);
    EXPECT_EQ(schema->inputs()[0].name, "boxes");
    EXPECT_EQ(schema->inputs()[1].name, "scores");
    EXPECT_EQ(schema->inputs()[2].name, "max_output_boxes_per_class");
    EXPECT_EQ(schema->inputs()[3].name, "iou_threshold");
    EXPECT_EQ(schema->inputs()[4].name, "score_threshold");

    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "selected_indices");
    EXPECT_EQ(schema->outputs()[0].type, "tensor(int64)");

    // NonMaxSuppression uses literal type strings on its inputs/outputs and
    // declares no named type-parameter constraints.
    EXPECT_EQ(schema->type_constraints().size(), 0u);

    ASSERT_EQ(schema->attributes().size(), 1u);
    EXPECT_EQ(schema->attributes()[0].name, "center_point_box");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[0].required);

    EXPECT_FALSE(schema->doc().empty());
  }
}

} // namespace Test
