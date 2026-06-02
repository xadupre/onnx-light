// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_controlflow_doc.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedIfSchemaCount = 3;
constexpr size_t kExpectedLoopSchemaCount = 3;
constexpr size_t kExpectedScanSchemaCount = 3;

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpControlflowRegistrationTest, ReturnsIfSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> if_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory("If");

  EXPECT_EQ(schemas.size(),
            kExpectedIfSchemaCount + kExpectedLoopSchemaCount + kExpectedScanSchemaCount);

  const onnx_op::LightOpSchema *const if_v13 = FindByVersion(if_schemas, 13);
  const onnx_op::LightOpSchema *const if_v11 = FindByVersion(if_schemas, 11);
  const onnx_op::LightOpSchema *const if_v1 = FindByVersion(if_schemas, 1);

  ASSERT_NE(nullptr, if_v13);
  ASSERT_NE(nullptr, if_v11);
  ASSERT_NE(nullptr, if_v1);

  EXPECT_EQ(if_v13->domain(), "ai.onnx");
  EXPECT_EQ(if_v13->inputs().size(), 1u);
  EXPECT_EQ(if_v13->outputs().size(), 1u);
  EXPECT_EQ(if_v13->inputs()[0].name, "cond");
  EXPECT_EQ(if_v13->outputs()[0].name, "outputs");
  EXPECT_EQ(if_v13->type_constraints().size(), 2u);

  EXPECT_EQ(if_v1->type_constraints()[0].allowed_type_strs.size(), 15u);
  EXPECT_EQ(if_v13->type_constraints()[0].allowed_type_strs.size(), 30u);
  EXPECT_STREQ(onnx_op::ToTypeString(if_v1->type_constraints()[0].allowed_type_strs.front()),
               "tensor(uint8)");
  EXPECT_STREQ(onnx_op::ToTypeString(if_v1->type_constraints()[0].allowed_type_strs.back()),
               "tensor(complex128)");
  EXPECT_STREQ(onnx_op::ToTypeString(if_v13->type_constraints()[0].allowed_type_strs[15]),
               "seq(tensor(uint8))");
  EXPECT_EQ(if_v13->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(if_v13->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kBool);
  EXPECT_EQ(if_v13->type_constraints()[0].description, "All Tensor and Sequence types");
  EXPECT_EQ(if_v11->type_constraints()[0].description, "All Tensor types");
  EXPECT_EQ(if_v13->doc(), "If conditional");
  EXPECT_EQ(if_v11->outputs()[0].description,
            "Values that are live-out to the enclosing scope. The return values in the "
            "`then_branch` and `else_branch` must be of the same data type. "
            "The `then_branch` and `else_branch` may produce tensors with the same "
            "element type and different shapes. "
            "If corresponding outputs from the then-branch and the else-branch have "
            "static shapes S1 and S2, then the shape of the corresponding output "
            "variable of the if-node (if present) must be compatible with both S1 "
            "and S2 as it represents the union of both possible shapes."
            "For example, if in a model file, the first "
            "output of `then_branch` is typed float tensor with shape [2] and the "
            "first output of `else_branch` is another float tensor with shape [3], "
            "If's first output should have (a) no shape set, or (b) "
            "a shape of rank 1 with neither `dim_value` nor `dim_param` set, or (c) "
            "a shape of rank 1 with a unique `dim_param`. "
            "In contrast, the first output cannot have the shape [2] since [2] and "
            "[3] are not compatible.");
  EXPECT_EQ(if_v1->outputs()[0].description,
            "Values that are live-out to the enclosing scope. The return values in the "
            "`then_branch` and `else_branch` must be of the same shape and same data type.");
}

TEST(OnnxOpControlflowRegistrationTest, ReturnsIfSchemasWithExpectedBranchAttributes) {
  const std::vector<onnx_op::LightOpSchema> if_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory("If");
  EXPECT_EQ(if_schemas.size(), kExpectedIfSchemaCount);

  const onnx_op::LightOpSchema *const if_v13 = FindByVersion(if_schemas, 13);
  const onnx_op::LightOpSchema *const if_v11 = FindByVersion(if_schemas, 11);
  const onnx_op::LightOpSchema *const if_v1 = FindByVersion(if_schemas, 1);

  ASSERT_NE(nullptr, if_v13);
  ASSERT_NE(nullptr, if_v11);
  ASSERT_NE(nullptr, if_v1);

  // Required GRAPH attributes "then_branch" and "else_branch" are present on
  // all opset versions and use the same descriptions across versions.
  for (const auto *s : {if_v1, if_v11, if_v13}) {
    ASSERT_EQ(s->attributes().size(), 2u);
    EXPECT_EQ(s->attributes()[0].name, "then_branch");
    EXPECT_EQ(s->attributes()[0].type, onnx_op::AttributeType::GRAPH);
    EXPECT_TRUE(s->attributes()[0].required);
    EXPECT_EQ(s->attributes()[0].description,
              onnx_op::controlflow::MakeIfThenBranchAttributeDescription());
    EXPECT_EQ(s->attributes()[1].name, "else_branch");
    EXPECT_EQ(s->attributes()[1].type, onnx_op::AttributeType::GRAPH);
    EXPECT_TRUE(s->attributes()[1].required);
    EXPECT_EQ(s->attributes()[1].description,
              onnx_op::controlflow::MakeIfElseBranchAttributeDescription());
  }
}

TEST(OnnxOpControlflowRegistrationTest, ReturnsLoopSchemasWithExpectedTypeHistory) {
  const std::vector<onnx_op::LightOpSchema> loop_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory("Loop");
  EXPECT_EQ(loop_schemas.size(), kExpectedLoopSchemaCount);

  const onnx_op::LightOpSchema *const loop_v13 = FindByVersion(loop_schemas, 13);
  const onnx_op::LightOpSchema *const loop_v11 = FindByVersion(loop_schemas, 11);
  const onnx_op::LightOpSchema *const loop_v1 = FindByVersion(loop_schemas, 1);

  ASSERT_NE(nullptr, loop_v13);
  ASSERT_NE(nullptr, loop_v11);
  ASSERT_NE(nullptr, loop_v1);

  EXPECT_EQ(loop_v13->domain(), "ai.onnx");
  EXPECT_EQ(loop_v13->inputs().size(), 3u);
  EXPECT_EQ(loop_v13->outputs().size(), 1u);
  EXPECT_EQ(loop_v13->inputs()[0].name, "M");
  EXPECT_EQ(loop_v13->inputs()[1].name, "cond");
  EXPECT_EQ(loop_v13->inputs()[2].name, "v_initial");
  EXPECT_EQ(loop_v13->outputs()[0].name, "v_final_and_scan_outputs");
  EXPECT_EQ(loop_v13->type_constraints().size(), 3u);
  EXPECT_EQ(loop_v13->type_constraints()[0].type_param_str, "V");
  EXPECT_EQ(loop_v13->type_constraints()[0].allowed_type_strs.size(), 30u);
  EXPECT_EQ(loop_v13->type_constraints()[0].description, "All Tensor and Sequence types");
  EXPECT_EQ(loop_v1->type_constraints()[0].allowed_type_strs.size(), 15u);
  EXPECT_EQ(loop_v1->type_constraints()[0].description, "All Tensor types");
  EXPECT_EQ(loop_v11->type_constraints()[0].allowed_type_strs.size(), 15u);

  // I and B type constraints are stable across versions.
  EXPECT_EQ(loop_v13->type_constraints()[1].type_param_str, "I");
  EXPECT_EQ(loop_v13->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(loop_v13->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kInt64);
  EXPECT_EQ(loop_v13->type_constraints()[2].type_param_str, "B");
  EXPECT_EQ(loop_v13->type_constraints()[2].allowed_type_strs[0], onnx_op::TensorType::kBool);

  // Required GRAPH attribute "body" is present on all opset versions.
  for (const auto *s : {loop_v1, loop_v11, loop_v13}) {
    ASSERT_EQ(s->attributes().size(), 1u);
    EXPECT_EQ(s->attributes()[0].name, "body");
    EXPECT_EQ(s->attributes()[0].type, onnx_op::AttributeType::GRAPH);
    EXPECT_TRUE(s->attributes()[0].required);
  }

  // v13 adds the "Scan outputs must be Tensors." sentence to the output desc.
  EXPECT_EQ(loop_v1->outputs()[0].description,
            "Final N loop carried dependency values then K scan_outputs");
  EXPECT_EQ(loop_v11->outputs()[0].description,
            "Final N loop carried dependency values then K scan_outputs");
  EXPECT_EQ(loop_v13->outputs()[0].description,
            "Final N loop carried dependency values then K scan_outputs. "
            "Scan outputs must be Tensors.");
}

TEST(OnnxOpControlflowRegistrationTest, ReturnsScanSchemasWithExpectedHistory) {
  const std::vector<onnx_op::LightOpSchema> scan_schemas =
      onnx_op::controlflow::GetAllOnnxOpControlflowSchemasWithHistory("Scan");
  EXPECT_EQ(scan_schemas.size(), kExpectedScanSchemaCount);

  const onnx_op::LightOpSchema *const scan_v11 = FindByVersion(scan_schemas, 11);
  const onnx_op::LightOpSchema *const scan_v9 = FindByVersion(scan_schemas, 9);
  const onnx_op::LightOpSchema *const scan_v8 = FindByVersion(scan_schemas, 8);

  ASSERT_NE(nullptr, scan_v11);
  ASSERT_NE(nullptr, scan_v9);
  ASSERT_NE(nullptr, scan_v8);

  EXPECT_EQ(scan_v11->domain(), "ai.onnx");

  // Opset 8 has the optional ``sequence_lens`` input prepended.
  ASSERT_EQ(scan_v8->inputs().size(), 2u);
  EXPECT_EQ(scan_v8->inputs()[0].name, "sequence_lens");
  EXPECT_EQ(scan_v8->inputs()[0].type, "I");
  EXPECT_EQ(scan_v8->inputs()[1].name, "initial_state_and_scan_inputs");
  EXPECT_EQ(scan_v8->inputs()[1].type, "V");
  // Opsets 9 and 11 dropped the ``sequence_lens`` input.
  ASSERT_EQ(scan_v9->inputs().size(), 1u);
  EXPECT_EQ(scan_v9->inputs()[0].name, "initial_state_and_scan_inputs");
  ASSERT_EQ(scan_v11->inputs().size(), 1u);
  EXPECT_EQ(scan_v11->inputs()[0].name, "initial_state_and_scan_inputs");

  for (const auto *s : {scan_v8, scan_v9, scan_v11}) {
    ASSERT_EQ(s->outputs().size(), 1u);
    EXPECT_EQ(s->outputs()[0].name, "final_state_and_scan_outputs");
    EXPECT_EQ(s->outputs()[0].type, "V");
  }

  // Type constraints: V always present; opset 8 lists I first (to match
  // ONNX's TypeConstraint registration order), then V.
  EXPECT_EQ(scan_v8->type_constraints().size(), 2u);
  EXPECT_EQ(scan_v8->type_constraints()[0].type_param_str, "I");
  EXPECT_EQ(scan_v8->type_constraints()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(scan_v8->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kInt64);
  EXPECT_EQ(scan_v8->type_constraints()[1].type_param_str, "V");
  EXPECT_EQ(scan_v8->type_constraints()[1].allowed_type_strs.size(), 15u);
  EXPECT_EQ(scan_v9->type_constraints().size(), 1u);
  EXPECT_EQ(scan_v11->type_constraints().size(), 1u);
  EXPECT_EQ(scan_v11->type_constraints()[0].allowed_type_strs.size(), 15u);

  // Attributes: opset 8 has body, num_scan_inputs, directions.
  ASSERT_EQ(scan_v8->attributes().size(), 3u);
  EXPECT_EQ(scan_v8->attributes()[0].name, "body");
  EXPECT_EQ(scan_v8->attributes()[0].type, onnx_op::AttributeType::GRAPH);
  EXPECT_TRUE(scan_v8->attributes()[0].required);
  EXPECT_EQ(scan_v8->attributes()[1].name, "num_scan_inputs");
  EXPECT_EQ(scan_v8->attributes()[1].type, onnx_op::AttributeType::INT);
  EXPECT_TRUE(scan_v8->attributes()[1].required);
  EXPECT_EQ(scan_v8->attributes()[2].name, "directions");
  EXPECT_EQ(scan_v8->attributes()[2].type, onnx_op::AttributeType::INTS);
  EXPECT_FALSE(scan_v8->attributes()[2].required);

  // Opsets 9 and 11 replace ``directions`` with four split attributes.
  for (const auto *s : {scan_v9, scan_v11}) {
    ASSERT_EQ(s->attributes().size(), 6u);
    EXPECT_EQ(s->attributes()[0].name, "body");
    EXPECT_TRUE(s->attributes()[0].required);
    EXPECT_EQ(s->attributes()[1].name, "num_scan_inputs");
    EXPECT_TRUE(s->attributes()[1].required);
    EXPECT_EQ(s->attributes()[2].name, "scan_input_directions");
    EXPECT_EQ(s->attributes()[2].type, onnx_op::AttributeType::INTS);
    EXPECT_FALSE(s->attributes()[2].required);
    EXPECT_EQ(s->attributes()[3].name, "scan_output_directions");
    EXPECT_EQ(s->attributes()[4].name, "scan_input_axes");
    EXPECT_EQ(s->attributes()[5].name, "scan_output_axes");
  }

  EXPECT_EQ(scan_v8->attributes()[0].description,
            onnx_op::controlflow::MakeScanBodyAttributeDescription());
  EXPECT_EQ(scan_v8->attributes()[2].description,
            onnx_op::controlflow::MakeScanDirectionsAttributeDescription());
  EXPECT_EQ(scan_v11->attributes()[2].description,
            onnx_op::controlflow::MakeScanInputDirectionsAttributeDescription());
}

} // namespace Test
