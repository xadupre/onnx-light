// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"

#include <gtest/gtest.h>

#include <set>
#include <variant>
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

TEST(OnnxOpTensorRegistrationTest, ReturnsAffineGridSchemaWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> affine_grid_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("AffineGrid");

  const onnx_op::LightOpSchema *const ag_v20 = FindByVersion(affine_grid_schemas, 20);
  ASSERT_NE(nullptr, ag_v20);
  EXPECT_EQ(ag_v20->name(), "AffineGrid");
  EXPECT_EQ(ag_v20->domain(), onnx_op::kOnnxDomain);
  EXPECT_EQ(ag_v20->since_version(), 20);

  ASSERT_EQ(ag_v20->inputs().size(), 2u);
  EXPECT_EQ(ag_v20->inputs()[0].name, "theta");
  EXPECT_EQ(ag_v20->inputs()[0].type, "T1");
  EXPECT_EQ(ag_v20->inputs()[1].name, "size");
  EXPECT_EQ(ag_v20->inputs()[1].type, "T2");

  ASSERT_EQ(ag_v20->outputs().size(), 1u);
  EXPECT_EQ(ag_v20->outputs()[0].name, "grid");
  EXPECT_EQ(ag_v20->outputs()[0].type, "T1");

  ASSERT_EQ(ag_v20->type_constraints().size(), 2u);
  EXPECT_EQ(ag_v20->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(ag_v20->type_constraints()[0].allowed_type_strs,
            (std::vector<onnx_op::TensorType>{
                onnx_op::TensorType::kBfloat16, onnx_op::TensorType::kFloat16,
                onnx_op::TensorType::kFloat, onnx_op::TensorType::kDouble}));
  EXPECT_EQ(ag_v20->type_constraints()[0].description, "Constrain grid types to float tensors.");
  EXPECT_EQ(ag_v20->type_constraints()[1].type_param_str, "T2");
  ASSERT_EQ(ag_v20->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(ag_v20->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kInt64);
  EXPECT_EQ(ag_v20->type_constraints()[1].description, "Constrain size's type to int64 tensors.");

  ASSERT_EQ(ag_v20->attributes().size(), 1u);
  EXPECT_EQ(ag_v20->attributes()[0].name, "align_corners");
  EXPECT_EQ(ag_v20->attributes()[0].type, onnx_op::AttributeType::INT);
  EXPECT_FALSE(ag_v20->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(ag_v20->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(ag_v20->attributes()[0].default_value), 0);

  EXPECT_TRUE(ag_v20->has_function_implementation());
  EXPECT_FALSE(ag_v20->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCastSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> cast_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Cast");

  EXPECT_EQ(schemas.size(), 22u);

  const onnx_op::LightOpSchema *const cast_v1 = FindByVersion(cast_schemas, 1);
  const onnx_op::LightOpSchema *const cast_v6 = FindByVersion(cast_schemas, 6);
  const onnx_op::LightOpSchema *const cast_v9 = FindByVersion(cast_schemas, 9);
  const onnx_op::LightOpSchema *const cast_v13 = FindByVersion(cast_schemas, 13);
  const onnx_op::LightOpSchema *const cast_v19 = FindByVersion(cast_schemas, 19);
  const onnx_op::LightOpSchema *const cast_v21 = FindByVersion(cast_schemas, 21);
  const onnx_op::LightOpSchema *const cast_v23 = FindByVersion(cast_schemas, 23);
  const onnx_op::LightOpSchema *const cast_v24 = FindByVersion(cast_schemas, 24);
  const onnx_op::LightOpSchema *const cast_v25 = FindByVersion(cast_schemas, 25);
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

TEST(OnnxOpTensorRegistrationTest, ReturnsCastLikeSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> cast_like_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("CastLike");

  const onnx_op::LightOpSchema *const cl_v15 = FindByVersion(cast_like_schemas, 15);
  const onnx_op::LightOpSchema *const cl_v19 = FindByVersion(cast_like_schemas, 19);
  const onnx_op::LightOpSchema *const cl_v21 = FindByVersion(cast_like_schemas, 21);
  const onnx_op::LightOpSchema *const cl_v23 = FindByVersion(cast_like_schemas, 23);
  const onnx_op::LightOpSchema *const cl_v24 = FindByVersion(cast_like_schemas, 24);
  const onnx_op::LightOpSchema *const cl_v25 = FindByVersion(cast_like_schemas, 25);
  ASSERT_NE(nullptr, cl_v15);
  ASSERT_NE(nullptr, cl_v19);
  ASSERT_NE(nullptr, cl_v21);
  ASSERT_NE(nullptr, cl_v23);
  ASSERT_NE(nullptr, cl_v24);
  ASSERT_NE(nullptr, cl_v25);
  EXPECT_EQ(cl_v25->domain(), "ai.onnx");
  ASSERT_EQ(cl_v25->inputs().size(), 2u);
  EXPECT_EQ(cl_v25->inputs()[0].name, "input");
  EXPECT_EQ(cl_v25->inputs()[0].type, "T1");
  EXPECT_EQ(cl_v25->inputs()[1].name, "target_type");
  EXPECT_EQ(cl_v25->inputs()[1].type, "T2");
  ASSERT_EQ(cl_v25->outputs().size(), 1u);
  EXPECT_EQ(cl_v25->outputs()[0].name, "output");
  EXPECT_EQ(cl_v25->outputs()[0].type, "T2");
  ASSERT_EQ(cl_v25->type_constraints().size(), 2u);
  EXPECT_EQ(cl_v15->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer13());
  EXPECT_EQ(cl_v19->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer19());
  EXPECT_EQ(cl_v21->type_constraints()[0].allowed_type_strs.size(),
            onnx_op::CastTypesVer21().size());
  // CastLike v21 uses ONNX IR version 10's `all_non_complex_tensor_types_ir10`
  // (uint8-first) ordering rather than Cast v21's float16-first CastTypesVer21;
  // compare as a set to verify the same dtypes are allowed without enforcing
  // an order that differs from Cast.
  {
    const auto &actual = cl_v21->type_constraints()[0].allowed_type_strs;
    const auto expected = onnx_op::CastTypesVer21();
    std::set<onnx_op::TensorType> actual_set(actual.begin(), actual.end());
    std::set<onnx_op::TensorType> expected_set(expected.begin(), expected.end());
    EXPECT_EQ(actual_set, expected_set);
  }
  EXPECT_EQ(cl_v23->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer23());
  EXPECT_EQ(cl_v24->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer24());
  EXPECT_EQ(cl_v25->type_constraints()[0].allowed_type_strs, onnx_op::CastTypesVer25());
  // T1 and T2 share the same allowed type set per the upstream spec.
  EXPECT_EQ(cl_v25->type_constraints()[0].allowed_type_strs,
            cl_v25->type_constraints()[1].allowed_type_strs);
  EXPECT_TRUE(cl_v25->has_function_implementation());
  EXPECT_FALSE(cl_v25->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsConcatSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> concat_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Concat");

  const onnx_op::LightOpSchema *const concat_v1 = FindByVersion(concat_schemas, 1);
  const onnx_op::LightOpSchema *const concat_v4 = FindByVersion(concat_schemas, 4);
  const onnx_op::LightOpSchema *const concat_v11 = FindByVersion(concat_schemas, 11);
  const onnx_op::LightOpSchema *const concat_v13 = FindByVersion(concat_schemas, 13);
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

TEST(OnnxOpTensorRegistrationTest, ReturnsExpandSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> expand_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Expand");

  const onnx_op::LightOpSchema *const expand_v8 = FindByVersion(expand_schemas, 8);
  const onnx_op::LightOpSchema *const expand_v13 = FindByVersion(expand_schemas, 13);
  ASSERT_NE(nullptr, expand_v8);
  ASSERT_NE(nullptr, expand_v13);

  EXPECT_EQ(expand_v13->domain(), "ai.onnx");
  ASSERT_EQ(expand_v13->inputs().size(), 2u);
  EXPECT_EQ(expand_v13->inputs()[0].name, "input");
  EXPECT_EQ(expand_v13->inputs()[0].description, "Input tensor");
  EXPECT_EQ(expand_v13->inputs()[0].type, "T");
  EXPECT_EQ(expand_v13->inputs()[1].name, "shape");
  EXPECT_EQ(expand_v13->inputs()[1].type, "tensor(int64)");
  ASSERT_EQ(expand_v13->outputs().size(), 1u);
  EXPECT_EQ(expand_v13->outputs()[0].name, "output");
  EXPECT_EQ(expand_v13->outputs()[0].description, "Output tensor");
  EXPECT_EQ(expand_v13->outputs()[0].type, "T");
  ASSERT_EQ(expand_v13->type_constraints().size(), 1u);
  EXPECT_EQ(expand_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(expand_v8->type_constraints()[0].allowed_type_strs, onnx_op::AllTensorTypes());
  EXPECT_EQ(expand_v13->type_constraints()[0].allowed_type_strs, onnx_op::ConcatTypesVer13());
  EXPECT_EQ(expand_v13->type_constraints()[0].description,
            "Constrain input and output types to all tensors.");
  EXPECT_FALSE(expand_v13->doc().empty());
}

} // namespace Test
