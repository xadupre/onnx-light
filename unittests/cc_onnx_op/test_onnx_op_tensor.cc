// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"

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

const onnx_op::LightOpSchema *FindTensorSchema(const std::vector<onnx_op::LightOpSchema> &schemas,
                                               const std::string &op_type, int version) {
  for (const auto &schema : schemas) {
    if (schema.name() == op_type && schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCastSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();

  EXPECT_EQ(schemas.size(), 13u);

  const onnx_op::LightOpSchema *const cast_v1 = FindTensorSchema(schemas, "Cast", 1);
  const onnx_op::LightOpSchema *const cast_v6 = FindTensorSchema(schemas, "Cast", 6);
  const onnx_op::LightOpSchema *const cast_v9 = FindTensorSchema(schemas, "Cast", 9);
  const onnx_op::LightOpSchema *const cast_v13 = FindTensorSchema(schemas, "Cast", 13);
  const onnx_op::LightOpSchema *const cast_v19 = FindTensorSchema(schemas, "Cast", 19);
  const onnx_op::LightOpSchema *const cast_v21 = FindTensorSchema(schemas, "Cast", 21);
  const onnx_op::LightOpSchema *const cast_v23 = FindTensorSchema(schemas, "Cast", 23);
  const onnx_op::LightOpSchema *const cast_v24 = FindTensorSchema(schemas, "Cast", 24);
  const onnx_op::LightOpSchema *const cast_v25 = FindTensorSchema(schemas, "Cast", 25);
  ASSERT_NE(nullptr, cast_v1);
  ASSERT_NE(nullptr, cast_v6);
  ASSERT_NE(nullptr, cast_v9);
  ASSERT_NE(nullptr, cast_v13);
  ASSERT_NE(nullptr, cast_v19);
  ASSERT_NE(nullptr, cast_v21);
  ASSERT_NE(nullptr, cast_v23);
  ASSERT_NE(nullptr, cast_v24);
  ASSERT_NE(nullptr, cast_v25);
  EXPECT_EQ(cast_v25->domain(), "ai.onnx");
  EXPECT_EQ(cast_v25->inputs().size(), 1u);
  EXPECT_EQ(cast_v25->outputs().size(), 1u);
  EXPECT_EQ(cast_v25->type_constraints().size(), 2u);
  EXPECT_EQ(cast_v1->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer1And6());
  EXPECT_EQ(cast_v6->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer1And6());
  EXPECT_EQ(cast_v9->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer9());
  EXPECT_EQ(cast_v13->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer13());
  EXPECT_EQ(cast_v19->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer19());
  EXPECT_EQ(cast_v21->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer21());
  EXPECT_EQ(cast_v23->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer23());
  EXPECT_EQ(cast_v24->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer24());
  EXPECT_EQ(cast_v25->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer25());
  EXPECT_NE(cast_v1->type_constraints()[0].allowed_type_strs,
            cast_v9->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(cast_v19->type_constraints()[0].allowed_type_strs,
            cast_v21->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(cast_v25->inputs()[0].description, "Input tensor to be cast.");
  EXPECT_EQ(cast_v25->outputs()[0].description,
            "Output tensor with the same shape as input with type specified by the 'to' argument");
}

TEST(OnnxOpTensorRegistrationTest, ReturnsConcatSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();

  const onnx_op::LightOpSchema *const concat_v1 = FindTensorSchema(schemas, "Concat", 1);
  const onnx_op::LightOpSchema *const concat_v4 = FindTensorSchema(schemas, "Concat", 4);
  const onnx_op::LightOpSchema *const concat_v11 = FindTensorSchema(schemas, "Concat", 11);
  const onnx_op::LightOpSchema *const concat_v13 = FindTensorSchema(schemas, "Concat", 13);
  ASSERT_NE(nullptr, concat_v1);
  ASSERT_NE(nullptr, concat_v4);
  ASSERT_NE(nullptr, concat_v11);
  ASSERT_NE(nullptr, concat_v13);
  EXPECT_EQ(concat_v13->domain(), "ai.onnx");
  EXPECT_EQ(concat_v13->inputs().size(), 1u);
  EXPECT_EQ(concat_v13->outputs().size(), 1u);
  EXPECT_EQ(concat_v13->type_constraints().size(), 1u);
  EXPECT_EQ(concat_v13->inputs()[0].name, "inputs");
  EXPECT_EQ(concat_v13->inputs()[0].description, "List of tensors for concatenation");
  EXPECT_EQ(concat_v13->inputs()[0].type, "T");
  EXPECT_EQ(concat_v13->outputs()[0].name, "concat_result");
  EXPECT_EQ(concat_v13->outputs()[0].description, "Concatenated tensor");
  EXPECT_EQ(concat_v13->outputs()[0].type, "T");
  EXPECT_EQ(concat_v1->type_constraints()[0].allowed_type_strs, onnx_op::ConcatTypesVer1());
  EXPECT_EQ(concat_v4->type_constraints()[0].allowed_type_strs, onnx_op::ConcatTypesVer4And11());
  EXPECT_EQ(concat_v11->type_constraints()[0].allowed_type_strs, onnx_op::ConcatTypesVer4And11());
  EXPECT_EQ(concat_v13->type_constraints()[0].allowed_type_strs, onnx_op::ConcatTypesVer13());
  EXPECT_NE(concat_v1->type_constraints()[0].allowed_type_strs,
            concat_v4->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(concat_v11->type_constraints()[0].allowed_type_strs,
            concat_v13->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(concat_v1->type_constraints()[0].description,
            "Constrain output types to float tensors.");
  EXPECT_EQ(concat_v13->type_constraints()[0].description,
            "Constrain output types to any tensor type.");
}

} // namespace Test
