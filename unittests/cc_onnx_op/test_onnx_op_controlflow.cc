// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedIfSchemaCount = 3;

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

  EXPECT_EQ(schemas.size(), kExpectedIfSchemaCount);

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

} // namespace Test
