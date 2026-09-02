// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor.h"

#include <gtest/gtest.h>

#include <set>
#include <variant>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

static const core::schema::LightOpSchema *
FindByVersion(const std::vector<core::schema::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpTensorRegistrationTest, ReturnsAffineGridSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> affine_grid_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("AffineGrid");

  const core::schema::LightOpSchema *const ag_v20 = FindByVersion(affine_grid_schemas, 20);
  ASSERT_NE(nullptr, ag_v20);
  EXPECT_EQ(ag_v20->name(), "AffineGrid");
  EXPECT_EQ(ag_v20->domain(), core::schema::kOnnxDomain);
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
            (std::vector<core::schema::TensorType>{
                core::schema::TensorType::kBfloat16, core::schema::TensorType::kFloat16,
                core::schema::TensorType::kFloat, core::schema::TensorType::kDouble}));
  EXPECT_EQ(ag_v20->type_constraints()[0].description, "Constrain grid types to float tensors.");
  EXPECT_EQ(ag_v20->type_constraints()[1].type_param_str, "T2");
  ASSERT_EQ(ag_v20->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(ag_v20->type_constraints()[1].allowed_type_strs[0], core::schema::TensorType::kInt64);
  EXPECT_EQ(ag_v20->type_constraints()[1].description, "Constrain size's type to int64 tensors.");

  ASSERT_EQ(ag_v20->attributes().size(), 1u);
  EXPECT_EQ(ag_v20->attributes()[0].name, "align_corners");
  EXPECT_EQ(ag_v20->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(ag_v20->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(ag_v20->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(ag_v20->attributes()[0].default_value), 0);

  EXPECT_TRUE(ag_v20->has_function_implementation());
  EXPECT_FALSE(ag_v20->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsGridSampleSchemasForAllVersions) {
  const std::vector<core::schema::LightOpSchema> gs_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("GridSample");
  ASSERT_EQ(gs_schemas.size(), 3u);

  for (int version : {16, 20, 22}) {
    const core::schema::LightOpSchema *const s = FindByVersion(gs_schemas, version);
    ASSERT_NE(nullptr, s) << "missing GridSample schema for version " << version;
    EXPECT_EQ(s->name(), "GridSample");
    EXPECT_EQ(s->domain(), core::schema::kOnnxDomain);
    EXPECT_EQ(s->since_version(), version);
    ASSERT_EQ(s->inputs().size(), 2u);
    EXPECT_EQ(s->inputs()[0].name, "X");
    EXPECT_EQ(s->inputs()[0].type, "T1");
    EXPECT_EQ(s->inputs()[1].name, "grid");
    EXPECT_EQ(s->inputs()[1].type, "T2");
    ASSERT_EQ(s->outputs().size(), 1u);
    EXPECT_EQ(s->outputs()[0].name, "Y");
    EXPECT_EQ(s->outputs()[0].type, "T1");
    ASSERT_EQ(s->type_constraints().size(), 2u);
    EXPECT_EQ(s->type_constraints()[0].type_param_str, "T1");
    EXPECT_EQ(s->type_constraints()[1].type_param_str, "T2");
    ASSERT_EQ(s->attributes().size(), 3u);
    EXPECT_EQ(s->attributes()[0].name, "mode");
    EXPECT_EQ(s->attributes()[0].type, core::schema::AttributeType::STRING);
    EXPECT_EQ(s->attributes()[1].name, "padding_mode");
    EXPECT_EQ(s->attributes()[1].type, core::schema::AttributeType::STRING);
    EXPECT_EQ(s->attributes()[2].name, "align_corners");
    EXPECT_EQ(s->attributes()[2].type, core::schema::AttributeType::INT);
    EXPECT_FALSE(s->doc().empty());
  }

  // v16's default for ``mode`` is "bilinear"; v20/v22 use "linear".
  const core::schema::LightOpSchema *const s16 = FindByVersion(gs_schemas, 16);
  ASSERT_TRUE(std::holds_alternative<std::string>(s16->attributes()[0].default_value));
  EXPECT_EQ(std::get<std::string>(s16->attributes()[0].default_value), "bilinear");
  const core::schema::LightOpSchema *const s20 = FindByVersion(gs_schemas, 20);
  ASSERT_TRUE(std::holds_alternative<std::string>(s20->attributes()[0].default_value));
  EXPECT_EQ(std::get<std::string>(s20->attributes()[0].default_value), "linear");
}

TEST(OnnxOpTensorRegistrationTest, PadReflectExamplesAreValid) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Pad");

  for (int version : {11, 13, 18, 19, 21, 23, 24, 25}) {
    const core::schema::LightOpSchema *const schema = FindByVersion(schemas, version);
    ASSERT_NE(schema, nullptr) << "missing Pad schema for version " << version;
    EXPECT_NE(schema->doc().find("pads = [0, 1, 0, 1]"), std::string::npos) << version;
    EXPECT_NE(schema->doc().find("[1.2, 1.0, 1.2, 1.0]"), std::string::npos) << version;
  }
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCastSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> cast_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Cast");

  EXPECT_EQ(schemas.size(), 134u);

  const core::schema::LightOpSchema *const cast_v1 = FindByVersion(cast_schemas, 1);
  const core::schema::LightOpSchema *const cast_v6 = FindByVersion(cast_schemas, 6);
  const core::schema::LightOpSchema *const cast_v9 = FindByVersion(cast_schemas, 9);
  const core::schema::LightOpSchema *const cast_v13 = FindByVersion(cast_schemas, 13);
  const core::schema::LightOpSchema *const cast_v19 = FindByVersion(cast_schemas, 19);
  const core::schema::LightOpSchema *const cast_v21 = FindByVersion(cast_schemas, 21);
  const core::schema::LightOpSchema *const cast_v23 = FindByVersion(cast_schemas, 23);
  const core::schema::LightOpSchema *const cast_v24 = FindByVersion(cast_schemas, 24);
  const core::schema::LightOpSchema *const cast_v25 = FindByVersion(cast_schemas, 25);
  const core::schema::LightOpSchema *const cast_v28 = FindByVersion(cast_schemas, 28);
  ASSERT_NE(nullptr, cast_v1);
  ASSERT_NE(nullptr, cast_v6);
  ASSERT_NE(nullptr, cast_v9);
  ASSERT_NE(nullptr, cast_v13);
  ASSERT_NE(nullptr, cast_v19);
  ASSERT_NE(nullptr, cast_v21);
  ASSERT_NE(nullptr, cast_v23);
  ASSERT_NE(nullptr, cast_v24);
  ASSERT_NE(nullptr, cast_v25);
  ASSERT_NE(nullptr, cast_v28);
  EXPECT_EQ(cast_v25->domain(), "ai.onnx");
  EXPECT_EQ(cast_v25->inputs().size(), 1u);
  EXPECT_EQ(cast_v25->outputs().size(), 1u);
  EXPECT_EQ(cast_v25->type_constraints().size(), 2u);
  EXPECT_EQ(cast_v1->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer1And6());
  EXPECT_EQ(cast_v6->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer1And6());
  EXPECT_EQ(cast_v9->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer9());
  EXPECT_EQ(cast_v13->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer13());
  EXPECT_EQ(cast_v19->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer19());
  EXPECT_EQ(cast_v21->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer21());
  EXPECT_EQ(cast_v23->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer23());
  EXPECT_EQ(cast_v24->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer24());
  EXPECT_EQ(cast_v25->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer25());
  EXPECT_EQ(cast_v28->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer28());
  EXPECT_NE(cast_v1->type_constraints()[0].allowed_type_strs,
            cast_v9->type_constraints()[0].allowed_type_strs);
  EXPECT_NE(cast_v19->type_constraints()[0].allowed_type_strs,
            cast_v21->type_constraints()[0].allowed_type_strs);
  EXPECT_EQ(cast_v25->inputs()[0].description, "Input tensor to be cast.");
  EXPECT_EQ(cast_v25->outputs()[0].description,
            "Output tensor with the same shape as input with type specified by the 'to' argument");
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCastLikeSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> cast_like_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("CastLike");

  const core::schema::LightOpSchema *const cl_v15 = FindByVersion(cast_like_schemas, 15);
  const core::schema::LightOpSchema *const cl_v19 = FindByVersion(cast_like_schemas, 19);
  const core::schema::LightOpSchema *const cl_v21 = FindByVersion(cast_like_schemas, 21);
  const core::schema::LightOpSchema *const cl_v23 = FindByVersion(cast_like_schemas, 23);
  const core::schema::LightOpSchema *const cl_v24 = FindByVersion(cast_like_schemas, 24);
  const core::schema::LightOpSchema *const cl_v25 = FindByVersion(cast_like_schemas, 25);
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
  EXPECT_EQ(cl_v15->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer13());
  EXPECT_EQ(cl_v19->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer19());
  EXPECT_EQ(cl_v21->type_constraints()[0].allowed_type_strs.size(),
            core::schema::CastTypesVer21().size());
  // CastLike v21 uses ONNX IR version 10's `all_non_complex_tensor_types_ir10`
  // (uint8-first) ordering rather than Cast v21's float16-first CastTypesVer21;
  // compare as a set to verify the same dtypes are allowed without enforcing
  // an order that differs from Cast.
  {
    const auto &actual = cl_v21->type_constraints()[0].allowed_type_strs;
    const auto expected = core::schema::CastTypesVer21();
    std::set<core::schema::TensorType> actual_set(actual.begin(), actual.end());
    std::set<core::schema::TensorType> expected_set(expected.begin(), expected.end());
    EXPECT_EQ(actual_set, expected_set);
  }
  EXPECT_EQ(cl_v23->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer23());
  EXPECT_EQ(cl_v24->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer24());
  EXPECT_EQ(cl_v25->type_constraints()[0].allowed_type_strs, core::schema::CastTypesVer25());
  // T1 and T2 share the same allowed type set per the upstream spec.
  EXPECT_EQ(cl_v25->type_constraints()[0].allowed_type_strs,
            cl_v25->type_constraints()[1].allowed_type_strs);
  EXPECT_TRUE(cl_v25->has_function_implementation());
  EXPECT_FALSE(cl_v25->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsConcatSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory();
  const std::vector<core::schema::LightOpSchema> concat_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Concat");

  const core::schema::LightOpSchema *const concat_v1 = FindByVersion(concat_schemas, 1);
  const core::schema::LightOpSchema *const concat_v4 = FindByVersion(concat_schemas, 4);
  const core::schema::LightOpSchema *const concat_v11 = FindByVersion(concat_schemas, 11);
  const core::schema::LightOpSchema *const concat_v13 = FindByVersion(concat_schemas, 13);
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
  EXPECT_EQ(concat_v1->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer1());
  EXPECT_EQ(concat_v4->type_constraints()[0].allowed_type_strs,
            core::schema::ConcatTypesVer4And11());
  EXPECT_EQ(concat_v11->type_constraints()[0].allowed_type_strs,
            core::schema::ConcatTypesVer4And11());
  EXPECT_EQ(concat_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
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
  const std::vector<core::schema::LightOpSchema> expand_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Expand");

  const core::schema::LightOpSchema *const expand_v8 = FindByVersion(expand_schemas, 8);
  const core::schema::LightOpSchema *const expand_v13 = FindByVersion(expand_schemas, 13);
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
  EXPECT_EQ(expand_v8->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(expand_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(expand_v13->type_constraints()[0].description,
            "Constrain input and output types to all tensors.");
  EXPECT_FALSE(expand_v13->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsReshapeSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> reshape_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Reshape");

  const core::schema::LightOpSchema *const reshape_v13 = FindByVersion(reshape_schemas, 13);
  const core::schema::LightOpSchema *const reshape_v25 = FindByVersion(reshape_schemas, 25);
  ASSERT_NE(nullptr, reshape_v13);
  ASSERT_NE(nullptr, reshape_v25);

  EXPECT_EQ(reshape_v25->domain(), "ai.onnx");
  ASSERT_EQ(reshape_v25->inputs().size(), 2u);
  EXPECT_EQ(reshape_v25->inputs()[0].name, "data");
  EXPECT_EQ(reshape_v25->inputs()[1].name, "shape");
  ASSERT_EQ(reshape_v25->outputs().size(), 1u);
  EXPECT_EQ(reshape_v25->outputs()[0].name, "reshaped");
  ASSERT_EQ(reshape_v25->type_constraints().size(), 1u);
  EXPECT_EQ(reshape_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_GT(reshape_v25->type_constraints()[0].allowed_type_strs.size(),
            reshape_v13->type_constraints()[0].allowed_type_strs.size());
  ASSERT_EQ(reshape_v25->attributes().size(), 1u);
  EXPECT_EQ(reshape_v25->attributes()[0].name, "allowzero");
  EXPECT_EQ(reshape_v25->attributes()[0].type, core::schema::AttributeType::INT);
}

TEST(OnnxOpTensorRegistrationTest, ReturnsSliceSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> slice_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Slice");
  ASSERT_EQ(slice_schemas.size(), 1u);
  const core::schema::LightOpSchema &slice_v13 = slice_schemas[0];

  EXPECT_EQ(slice_v13.name(), "Slice");
  EXPECT_EQ(slice_v13.since_version(), 13);
  ASSERT_EQ(slice_v13.inputs().size(), 5u);
  ASSERT_EQ(slice_v13.outputs().size(), 1u);
  ASSERT_EQ(slice_v13.type_constraints().size(), 2u);
  EXPECT_EQ(slice_v13.type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(slice_v13.type_constraints()[1].type_param_str, "Tind");
  EXPECT_EQ(slice_v13.type_constraints()[1].allowed_type_strs,
            (std::vector<core::schema::TensorType>{core::schema::TensorType::kInt32,
                                                   core::schema::TensorType::kInt64}));
}

TEST(OnnxOpTensorRegistrationTest, ReturnsTileSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> tile_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Tile");

  const core::schema::LightOpSchema *const tile_v6 = FindByVersion(tile_schemas, 6);
  const core::schema::LightOpSchema *const tile_v13 = FindByVersion(tile_schemas, 13);
  ASSERT_NE(nullptr, tile_v6);
  ASSERT_NE(nullptr, tile_v13);

  EXPECT_EQ(tile_v13->domain(), "ai.onnx");
  ASSERT_EQ(tile_v13->inputs().size(), 2u);
  EXPECT_EQ(tile_v13->inputs()[0].name, "input");
  EXPECT_EQ(tile_v13->inputs()[0].description, "Input tensor of any shape.");
  EXPECT_EQ(tile_v13->inputs()[0].type, "T");
  EXPECT_EQ(tile_v13->inputs()[1].name, "repeats");
  EXPECT_EQ(tile_v13->inputs()[1].type, "T1");
  ASSERT_EQ(tile_v13->outputs().size(), 1u);
  EXPECT_EQ(tile_v13->outputs()[0].name, "output");
  EXPECT_EQ(tile_v13->outputs()[0].type, "T");
  ASSERT_EQ(tile_v13->type_constraints().size(), 2u);
  EXPECT_EQ(tile_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(tile_v6->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(tile_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(tile_v13->type_constraints()[0].description,
            "Constrain input and output types to all tensor types.");
  EXPECT_EQ(tile_v13->type_constraints()[1].type_param_str, "T1");
  ASSERT_EQ(tile_v13->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(tile_v13->type_constraints()[1].allowed_type_strs[0], core::schema::TensorType::kInt64);
  EXPECT_EQ(tile_v13->type_constraints()[1].description,
            "Constrain repeat's type to int64 tensors.");
  EXPECT_FALSE(tile_v13->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsDepthToSpaceSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> d2s_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("DepthToSpace");

  const core::schema::LightOpSchema *const d2s_v1 = FindByVersion(d2s_schemas, 1);
  const core::schema::LightOpSchema *const d2s_v11 = FindByVersion(d2s_schemas, 11);
  const core::schema::LightOpSchema *const d2s_v13 = FindByVersion(d2s_schemas, 13);
  ASSERT_NE(nullptr, d2s_v1);
  ASSERT_NE(nullptr, d2s_v11);
  ASSERT_NE(nullptr, d2s_v13);

  EXPECT_EQ(d2s_v13->domain(), "ai.onnx");
  ASSERT_EQ(d2s_v13->inputs().size(), 1u);
  EXPECT_EQ(d2s_v13->inputs()[0].name, "input");
  EXPECT_EQ(d2s_v13->inputs()[0].type, "T");
  ASSERT_EQ(d2s_v13->outputs().size(), 1u);
  EXPECT_EQ(d2s_v13->outputs()[0].name, "output");
  EXPECT_EQ(d2s_v13->outputs()[0].type, "T");
  ASSERT_EQ(d2s_v13->type_constraints().size(), 1u);
  EXPECT_EQ(d2s_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(d2s_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(d2s_v11->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(d2s_v1->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  // v1 has just the required blocksize attribute; v11/v13 also expose mode.
  EXPECT_EQ(d2s_v1->attributes().size(), 1u);
  EXPECT_EQ(d2s_v1->attributes()[0].name, "blocksize");
  EXPECT_EQ(d2s_v1->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_TRUE(d2s_v1->attributes()[0].required);
  EXPECT_EQ(d2s_v13->attributes().size(), 2u);
  EXPECT_EQ(d2s_v13->attributes()[1].name, "mode");
  EXPECT_EQ(d2s_v13->attributes()[1].type, core::schema::AttributeType::STRING);
  EXPECT_FALSE(d2s_v13->attributes()[1].required);
  EXPECT_FALSE(d2s_v13->doc().empty());
  EXPECT_NE(d2s_v13->doc().find("DepthToSpace"), std::string::npos);
}

TEST(OnnxOpTensorRegistrationTest, ReturnsSpaceToDepthSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> s2d_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("SpaceToDepth");

  const core::schema::LightOpSchema *const s2d_v1 = FindByVersion(s2d_schemas, 1);
  const core::schema::LightOpSchema *const s2d_v13 = FindByVersion(s2d_schemas, 13);
  ASSERT_NE(nullptr, s2d_v1);
  ASSERT_NE(nullptr, s2d_v13);

  EXPECT_EQ(s2d_v13->domain(), "ai.onnx");
  ASSERT_EQ(s2d_v13->inputs().size(), 1u);
  EXPECT_EQ(s2d_v13->inputs()[0].name, "input");
  EXPECT_EQ(s2d_v13->inputs()[0].type, "T");
  ASSERT_EQ(s2d_v13->outputs().size(), 1u);
  EXPECT_EQ(s2d_v13->outputs()[0].name, "output");
  EXPECT_EQ(s2d_v13->outputs()[0].type, "T");
  ASSERT_EQ(s2d_v13->type_constraints().size(), 1u);
  EXPECT_EQ(s2d_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(s2d_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(s2d_v1->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  // Both versions only expose the required blocksize attribute (no mode).
  ASSERT_EQ(s2d_v1->attributes().size(), 1u);
  EXPECT_EQ(s2d_v1->attributes()[0].name, "blocksize");
  EXPECT_EQ(s2d_v1->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_TRUE(s2d_v1->attributes()[0].required);
  ASSERT_EQ(s2d_v13->attributes().size(), 1u);
  EXPECT_EQ(s2d_v13->attributes()[0].name, "blocksize");
  EXPECT_FALSE(s2d_v13->doc().empty());
  EXPECT_NE(s2d_v13->doc().find("SpaceToDepth"), std::string::npos);
}

TEST(OnnxOpTensorRegistrationTest, ReturnsTransposeSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> transpose_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Transpose");

  const core::schema::LightOpSchema *const transpose_v1 = FindByVersion(transpose_schemas, 1);
  const core::schema::LightOpSchema *const transpose_v13 = FindByVersion(transpose_schemas, 13);
  const core::schema::LightOpSchema *const transpose_v21 = FindByVersion(transpose_schemas, 21);
  const core::schema::LightOpSchema *const transpose_v23 = FindByVersion(transpose_schemas, 23);
  const core::schema::LightOpSchema *const transpose_v24 = FindByVersion(transpose_schemas, 24);
  const core::schema::LightOpSchema *const transpose_v25 = FindByVersion(transpose_schemas, 25);
  ASSERT_NE(nullptr, transpose_v1);
  ASSERT_NE(nullptr, transpose_v13);
  ASSERT_NE(nullptr, transpose_v21);
  ASSERT_NE(nullptr, transpose_v23);
  ASSERT_NE(nullptr, transpose_v24);
  ASSERT_NE(nullptr, transpose_v25);

  EXPECT_EQ(transpose_v25->domain(), "ai.onnx");
  ASSERT_EQ(transpose_v25->inputs().size(), 1u);
  EXPECT_EQ(transpose_v25->inputs()[0].name, "data");
  EXPECT_EQ(transpose_v25->inputs()[0].description, "An input tensor.");
  EXPECT_EQ(transpose_v25->inputs()[0].type, "T");
  ASSERT_EQ(transpose_v25->outputs().size(), 1u);
  EXPECT_EQ(transpose_v25->outputs()[0].name, "transposed");
  EXPECT_EQ(transpose_v25->outputs()[0].description, "Transposed output.");
  EXPECT_EQ(transpose_v25->outputs()[0].type, "T");
  ASSERT_EQ(transpose_v25->type_constraints().size(), 1u);
  EXPECT_EQ(transpose_v25->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(transpose_v1->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(transpose_v13->type_constraints()[0].allowed_type_strs,
            core::schema::ConcatTypesVer13());
  EXPECT_EQ(transpose_v25->type_constraints()[0].description,
            "Constrain input and output types to all tensor types.");
  EXPECT_FALSE(transpose_v25->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsSqueezeSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> squeeze_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Squeeze");

  const core::schema::LightOpSchema *const sq_v1 = FindByVersion(squeeze_schemas, 1);
  const core::schema::LightOpSchema *const sq_v11 = FindByVersion(squeeze_schemas, 11);
  const core::schema::LightOpSchema *const sq_v13 = FindByVersion(squeeze_schemas, 13);
  const core::schema::LightOpSchema *const sq_v21 = FindByVersion(squeeze_schemas, 21);
  const core::schema::LightOpSchema *const sq_v23 = FindByVersion(squeeze_schemas, 23);
  const core::schema::LightOpSchema *const sq_v24 = FindByVersion(squeeze_schemas, 24);
  const core::schema::LightOpSchema *const sq_v25 = FindByVersion(squeeze_schemas, 25);
  ASSERT_NE(nullptr, sq_v1);
  ASSERT_NE(nullptr, sq_v11);
  ASSERT_NE(nullptr, sq_v13);
  ASSERT_NE(nullptr, sq_v21);
  ASSERT_NE(nullptr, sq_v23);
  ASSERT_NE(nullptr, sq_v24);
  ASSERT_NE(nullptr, sq_v25);

  EXPECT_EQ(sq_v25->domain(), "ai.onnx");
  ASSERT_EQ(sq_v25->inputs().size(), 2u);
  EXPECT_EQ(sq_v25->inputs()[0].name, "data");
  EXPECT_EQ(sq_v25->inputs()[1].name, "axes");
  EXPECT_EQ(sq_v25->inputs()[1].type, "tensor(int64)");
  ASSERT_EQ(sq_v25->outputs().size(), 1u);
  EXPECT_EQ(sq_v25->outputs()[0].name, "squeezed");
  ASSERT_EQ(sq_v25->type_constraints().size(), 1u);
  EXPECT_EQ(sq_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(sq_v11->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(sq_v1->attributes().size(), 1u);
  EXPECT_EQ(sq_v1->attributes()[0].name, "axes");
  EXPECT_EQ(sq_v1->attributes()[0].type, core::schema::AttributeType::INTS);
  EXPECT_FALSE(sq_v1->attributes()[0].required);
  EXPECT_FALSE(sq_v25->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsUnsqueezeSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> unsqueeze_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Unsqueeze");

  const core::schema::LightOpSchema *const us_v1 = FindByVersion(unsqueeze_schemas, 1);
  const core::schema::LightOpSchema *const us_v11 = FindByVersion(unsqueeze_schemas, 11);
  const core::schema::LightOpSchema *const us_v13 = FindByVersion(unsqueeze_schemas, 13);
  const core::schema::LightOpSchema *const us_v21 = FindByVersion(unsqueeze_schemas, 21);
  const core::schema::LightOpSchema *const us_v23 = FindByVersion(unsqueeze_schemas, 23);
  const core::schema::LightOpSchema *const us_v24 = FindByVersion(unsqueeze_schemas, 24);
  const core::schema::LightOpSchema *const us_v25 = FindByVersion(unsqueeze_schemas, 25);
  ASSERT_NE(nullptr, us_v1);
  ASSERT_NE(nullptr, us_v11);
  ASSERT_NE(nullptr, us_v13);
  ASSERT_NE(nullptr, us_v21);
  ASSERT_NE(nullptr, us_v23);
  ASSERT_NE(nullptr, us_v24);
  ASSERT_NE(nullptr, us_v25);

  EXPECT_EQ(us_v25->domain(), "ai.onnx");
  ASSERT_EQ(us_v25->inputs().size(), 2u);
  EXPECT_EQ(us_v25->inputs()[0].name, "data");
  EXPECT_EQ(us_v25->inputs()[1].name, "axes");
  EXPECT_EQ(us_v25->inputs()[1].type, "tensor(int64)");
  ASSERT_EQ(us_v25->outputs().size(), 1u);
  EXPECT_EQ(us_v25->outputs()[0].name, "expanded");
  ASSERT_EQ(us_v25->type_constraints().size(), 1u);
  EXPECT_EQ(us_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(us_v11->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(us_v1->attributes().size(), 1u);
  EXPECT_EQ(us_v1->attributes()[0].name, "axes");
  EXPECT_EQ(us_v1->attributes()[0].type, core::schema::AttributeType::INTS);
  EXPECT_TRUE(us_v1->attributes()[0].required);
  EXPECT_FALSE(us_v25->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsNonZeroSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> nonzero_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("NonZero");

  const core::schema::LightOpSchema *const nz_v9 = FindByVersion(nonzero_schemas, 9);
  const core::schema::LightOpSchema *const nz_v13 = FindByVersion(nonzero_schemas, 13);
  ASSERT_NE(nullptr, nz_v9);
  ASSERT_NE(nullptr, nz_v13);

  EXPECT_EQ(nz_v13->name(), "NonZero");
  EXPECT_EQ(nz_v13->domain(), "ai.onnx");
  EXPECT_EQ(nz_v13->since_version(), 13);

  ASSERT_EQ(nz_v13->inputs().size(), 1u);
  EXPECT_EQ(nz_v13->inputs()[0].name, "X");
  EXPECT_EQ(nz_v13->inputs()[0].description, "input");
  EXPECT_EQ(nz_v13->inputs()[0].type, "T");

  ASSERT_EQ(nz_v13->outputs().size(), 1u);
  EXPECT_EQ(nz_v13->outputs()[0].name, "Y");
  EXPECT_EQ(nz_v13->outputs()[0].description, "output");
  EXPECT_EQ(nz_v13->outputs()[0].type, "tensor(int64)");

  ASSERT_EQ(nz_v13->type_constraints().size(), 1u);
  EXPECT_EQ(nz_v13->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(nz_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(nz_v13->type_constraints()[0].description, "Constrain to all tensor types.");

  EXPECT_EQ(nz_v9->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_FALSE(nz_v13->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsShapeSchemasWithHistory) {
  const std::vector<core::schema::LightOpSchema> shape_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Shape");

  ASSERT_EQ(shape_schemas.size(), 8u);
  for (const core::schema::LightOpSchema &s : shape_schemas) {
    EXPECT_EQ(s.name(), "Shape");
    EXPECT_EQ(s.domain(), "ai.onnx");

    ASSERT_EQ(s.inputs().size(), 1u);
    EXPECT_EQ(s.inputs()[0].name, "data");
    EXPECT_EQ(s.inputs()[0].description, "An input tensor.");
    EXPECT_EQ(s.inputs()[0].type, "T");

    ASSERT_EQ(s.outputs().size(), 1u);
    EXPECT_EQ(s.outputs()[0].name, "shape");
    EXPECT_EQ(s.outputs()[0].description, "Shape of the input tensor");
    EXPECT_EQ(s.outputs()[0].type, "T1");

    ASSERT_EQ(s.type_constraints().size(), 2u);
    EXPECT_EQ(s.type_constraints()[0].type_param_str, "T");
    EXPECT_EQ(s.type_constraints()[1].type_param_str, "T1");
    EXPECT_EQ(s.type_constraints()[1].allowed_type_strs,
              std::vector<core::schema::TensorType>{core::schema::TensorType::kInt64});
    EXPECT_FALSE(s.doc().empty());
  }

  // Versions < 15 have no attributes; versions >= 15 declare ``start`` and ``end``.
  const core::schema::LightOpSchema *const s_v1 = FindByVersion(shape_schemas, 1);
  const core::schema::LightOpSchema *const s_v13 = FindByVersion(shape_schemas, 13);
  const core::schema::LightOpSchema *const s_v15 = FindByVersion(shape_schemas, 15);
  const core::schema::LightOpSchema *const s_v25 = FindByVersion(shape_schemas, 25);
  ASSERT_NE(nullptr, s_v1);
  ASSERT_NE(nullptr, s_v13);
  ASSERT_NE(nullptr, s_v15);
  ASSERT_NE(nullptr, s_v25);

  EXPECT_TRUE(s_v1->attributes().empty());
  EXPECT_TRUE(s_v13->attributes().empty());

  ASSERT_EQ(s_v15->attributes().size(), 2u);
  EXPECT_EQ(s_v15->attributes()[0].name, "start");
  EXPECT_EQ(s_v15->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(s_v15->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(s_v15->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(s_v15->attributes()[0].default_value), 0);
  EXPECT_EQ(s_v15->attributes()[1].name, "end");
  EXPECT_EQ(s_v15->attributes()[1].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(s_v15->attributes()[1].required);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(s_v15->attributes()[1].default_value));

  // Type sets evolve with IR versions.
  EXPECT_EQ(s_v1->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(s_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(s_v25->type_constraints()[0].allowed_type_strs.size(), 26u);
  EXPECT_EQ(s_v25->type_constraints()[0].allowed_type_strs.back(), core::schema::TensorType::kInt2);
}

TEST(OnnxOpTensorRegistrationTest, ReturnsTriluSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> trilu_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Trilu");

  const core::schema::LightOpSchema *const trilu_v14 = FindByVersion(trilu_schemas, 14);
  ASSERT_NE(nullptr, trilu_v14);
  EXPECT_EQ(trilu_v14->name(), "Trilu");
  EXPECT_EQ(trilu_v14->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(trilu_v14->since_version(), 14);

  ASSERT_EQ(trilu_v14->inputs().size(), 2u);
  EXPECT_EQ(trilu_v14->inputs()[0].name, "input");
  EXPECT_EQ(trilu_v14->inputs()[0].type, "T");
  EXPECT_EQ(trilu_v14->inputs()[1].name, "k");
  EXPECT_EQ(trilu_v14->inputs()[1].type, "tensor(int64)");

  ASSERT_EQ(trilu_v14->outputs().size(), 1u);
  EXPECT_EQ(trilu_v14->outputs()[0].name, "output");
  EXPECT_EQ(trilu_v14->outputs()[0].type, "T");

  ASSERT_EQ(trilu_v14->type_constraints().size(), 1u);
  EXPECT_EQ(trilu_v14->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(trilu_v14->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(trilu_v14->type_constraints()[0].description,
            "Constrain input and output types to all tensor types.");

  ASSERT_EQ(trilu_v14->attributes().size(), 1u);
  EXPECT_EQ(trilu_v14->attributes()[0].name, "upper");
  EXPECT_EQ(trilu_v14->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(trilu_v14->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(trilu_v14->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(trilu_v14->attributes()[0].default_value), 1);

  EXPECT_FALSE(trilu_v14->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCenterCropPadSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> ccp_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("CenterCropPad");

  const core::schema::LightOpSchema *const ccp_v18 = FindByVersion(ccp_schemas, 18);
  ASSERT_NE(nullptr, ccp_v18);
  EXPECT_EQ(ccp_v18->name(), "CenterCropPad");
  EXPECT_EQ(ccp_v18->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(ccp_v18->since_version(), 18);

  ASSERT_EQ(ccp_v18->inputs().size(), 2u);
  EXPECT_EQ(ccp_v18->inputs()[0].name, "input_data");
  EXPECT_EQ(ccp_v18->inputs()[0].type, "T");
  EXPECT_EQ(ccp_v18->inputs()[1].name, "shape");
  EXPECT_EQ(ccp_v18->inputs()[1].type, "Tind");

  ASSERT_EQ(ccp_v18->outputs().size(), 1u);
  EXPECT_EQ(ccp_v18->outputs()[0].name, "output_data");
  EXPECT_EQ(ccp_v18->outputs()[0].type, "T");

  ASSERT_EQ(ccp_v18->type_constraints().size(), 2u);
  EXPECT_EQ(ccp_v18->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(ccp_v18->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());
  EXPECT_EQ(ccp_v18->type_constraints()[1].type_param_str, "Tind");

  ASSERT_EQ(ccp_v18->attributes().size(), 1u);
  EXPECT_EQ(ccp_v18->attributes()[0].name, "axes");
  EXPECT_EQ(ccp_v18->attributes()[0].type, core::schema::AttributeType::INTS);
  EXPECT_FALSE(ccp_v18->attributes()[0].required);

  EXPECT_FALSE(ccp_v18->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsTensorScatterSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> ts_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("TensorScatter");

  const core::schema::LightOpSchema *const ts_v24 = FindByVersion(ts_schemas, 24);
  ASSERT_NE(nullptr, ts_v24);
  EXPECT_EQ(ts_v24->name(), "TensorScatter");
  EXPECT_EQ(ts_v24->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(ts_v24->since_version(), 24);

  ASSERT_EQ(ts_v24->inputs().size(), 3u);
  EXPECT_EQ(ts_v24->inputs()[0].name, "past_cache");
  EXPECT_EQ(ts_v24->inputs()[0].type, "T");
  EXPECT_EQ(ts_v24->inputs()[1].name, "update");
  EXPECT_EQ(ts_v24->inputs()[1].type, "T");
  EXPECT_EQ(ts_v24->inputs()[2].name, "write_indices");
  EXPECT_EQ(ts_v24->inputs()[2].type, "tensor(int64)");

  ASSERT_EQ(ts_v24->outputs().size(), 1u);
  EXPECT_EQ(ts_v24->outputs()[0].name, "present_cache");
  EXPECT_EQ(ts_v24->outputs()[0].type, "T");

  ASSERT_EQ(ts_v24->type_constraints().size(), 1u);
  EXPECT_EQ(ts_v24->type_constraints()[0].type_param_str, "T");

  ASSERT_EQ(ts_v24->attributes().size(), 2u);
  EXPECT_EQ(ts_v24->attributes()[0].name, "axis");
  EXPECT_EQ(ts_v24->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(ts_v24->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(ts_v24->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(ts_v24->attributes()[0].default_value), -2);
  EXPECT_EQ(ts_v24->attributes()[1].name, "mode");
  EXPECT_EQ(ts_v24->attributes()[1].type, core::schema::AttributeType::STRING);
  EXPECT_FALSE(ts_v24->attributes()[1].required);
  ASSERT_TRUE(std::holds_alternative<std::string>(ts_v24->attributes()[1].default_value));
  EXPECT_EQ(std::get<std::string>(ts_v24->attributes()[1].default_value), "linear");

  EXPECT_FALSE(ts_v24->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsReverseSequenceSchemaWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> rs_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("ReverseSequence");

  const core::schema::LightOpSchema *const rs_v10 = FindByVersion(rs_schemas, 10);
  ASSERT_NE(nullptr, rs_v10);
  EXPECT_EQ(rs_v10->name(), "ReverseSequence");
  EXPECT_EQ(rs_v10->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(rs_v10->since_version(), 10);

  ASSERT_EQ(rs_v10->inputs().size(), 2u);
  EXPECT_EQ(rs_v10->inputs()[0].name, "input");
  EXPECT_EQ(rs_v10->inputs()[0].type, "T");
  EXPECT_EQ(rs_v10->inputs()[1].name, "sequence_lens");
  EXPECT_EQ(rs_v10->inputs()[1].type, "tensor(int64)");

  ASSERT_EQ(rs_v10->outputs().size(), 1u);
  EXPECT_EQ(rs_v10->outputs()[0].name, "Y");
  EXPECT_EQ(rs_v10->outputs()[0].type, "T");

  ASSERT_EQ(rs_v10->type_constraints().size(), 1u);
  EXPECT_EQ(rs_v10->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(rs_v10->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(rs_v10->type_constraints()[0].description,
            "Input and output types can be of any tensor type.");

  ASSERT_EQ(rs_v10->attributes().size(), 2u);
  EXPECT_EQ(rs_v10->attributes()[0].name, "time_axis");
  EXPECT_EQ(rs_v10->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(rs_v10->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(rs_v10->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(rs_v10->attributes()[0].default_value), 0);
  EXPECT_EQ(rs_v10->attributes()[1].name, "batch_axis");
  EXPECT_EQ(rs_v10->attributes()[1].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(rs_v10->attributes()[1].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(rs_v10->attributes()[1].default_value));
  EXPECT_EQ(std::get<int64_t>(rs_v10->attributes()[1].default_value), 1);

  EXPECT_FALSE(rs_v10->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsCompressSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> compress_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Compress");

  const core::schema::LightOpSchema *const compress_v9 = FindByVersion(compress_schemas, 9);
  const core::schema::LightOpSchema *const compress_v11 = FindByVersion(compress_schemas, 11);
  ASSERT_NE(nullptr, compress_v9);
  ASSERT_NE(nullptr, compress_v11);

  EXPECT_EQ(compress_v11->name(), "Compress");
  EXPECT_EQ(compress_v11->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(compress_v11->since_version(), 11);

  ASSERT_EQ(compress_v11->inputs().size(), 2u);
  EXPECT_EQ(compress_v11->inputs()[0].name, "input");
  EXPECT_EQ(compress_v11->inputs()[0].type, "T");
  EXPECT_EQ(compress_v11->inputs()[1].name, "condition");
  EXPECT_EQ(compress_v11->inputs()[1].type, "T1");

  ASSERT_EQ(compress_v11->outputs().size(), 1u);
  EXPECT_EQ(compress_v11->outputs()[0].name, "output");
  EXPECT_EQ(compress_v11->outputs()[0].type, "T");

  ASSERT_EQ(compress_v11->type_constraints().size(), 2u);
  EXPECT_EQ(compress_v11->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(compress_v11->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());
  EXPECT_EQ(compress_v11->type_constraints()[1].type_param_str, "T1");
  EXPECT_EQ(compress_v11->type_constraints()[1].allowed_type_strs,
            std::vector<core::schema::TensorType>{core::schema::TensorType::kBool});

  ASSERT_EQ(compress_v11->attributes().size(), 1u);
  EXPECT_EQ(compress_v11->attributes()[0].name, "axis");
  EXPECT_EQ(compress_v11->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(compress_v11->attributes()[0].required);

  EXPECT_FALSE(compress_v11->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsSplitSchemasWithoutShapeInference) {
  const std::vector<core::schema::LightOpSchema> split_schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("Split");

  const core::schema::LightOpSchema *const split_v1 = FindByVersion(split_schemas, 1);
  const core::schema::LightOpSchema *const split_v2 = FindByVersion(split_schemas, 2);
  const core::schema::LightOpSchema *const split_v11 = FindByVersion(split_schemas, 11);
  const core::schema::LightOpSchema *const split_v13 = FindByVersion(split_schemas, 13);
  const core::schema::LightOpSchema *const split_v18 = FindByVersion(split_schemas, 18);
  ASSERT_NE(nullptr, split_v1);
  ASSERT_NE(nullptr, split_v2);
  ASSERT_NE(nullptr, split_v11);
  ASSERT_NE(nullptr, split_v13);
  ASSERT_NE(nullptr, split_v18);

  // v1 carries the optional ``split`` *input*; v2/v11 drop it; v13+ reintroduce
  // it as an input.
  ASSERT_EQ(split_v1->inputs().size(), 2u);
  EXPECT_EQ(split_v1->inputs()[0].name, "input");
  EXPECT_EQ(split_v1->inputs()[1].name, "split");
  EXPECT_EQ(split_v1->inputs()[1].type, "T");
  EXPECT_EQ(split_v1->type_constraints()[0].allowed_type_strs, core::schema::FloatTypes());

  ASSERT_EQ(split_v2->inputs().size(), 1u);
  EXPECT_EQ(split_v2->inputs()[0].name, "input");
  EXPECT_EQ(split_v2->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());

  ASSERT_EQ(split_v11->inputs().size(), 1u);
  EXPECT_EQ(split_v11->type_constraints()[0].allowed_type_strs, core::schema::AllTensorTypes());

  ASSERT_EQ(split_v13->inputs().size(), 2u);
  EXPECT_EQ(split_v13->inputs()[0].name, "input");
  EXPECT_EQ(split_v13->inputs()[1].name, "split");
  EXPECT_EQ(split_v13->inputs()[1].type, "tensor(int64)");
  EXPECT_EQ(split_v13->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());

  ASSERT_EQ(split_v18->inputs().size(), 2u);
  EXPECT_EQ(split_v18->inputs()[1].name, "split");
  EXPECT_EQ(split_v18->type_constraints()[0].allowed_type_strs, core::schema::ConcatTypesVer13());

  // ``outputs`` is variadic for every Split version. The single declared
  // output exposes ``set_max_output(int_max)`` and ``set_min_output(1)``.
  for (const core::schema::LightOpSchema *s :
       {split_v1, split_v2, split_v11, split_v13, split_v18}) {
    ASSERT_EQ(s->outputs().size(), 1u);
    EXPECT_EQ(s->outputs()[0].type, "T");
    EXPECT_EQ(s->min_output(), 1);
    EXPECT_GT(s->max_output(), 1);
  }
  EXPECT_EQ(split_v1->outputs()[0].name, "outputs...");
  EXPECT_EQ(split_v18->outputs()[0].name, "outputs");

  // v1 has axis + split attributes; v2/v11 have axis (with default 0) + split;
  // v13 has only axis; v18 has axis + num_outputs.
  ASSERT_EQ(split_v1->attributes().size(), 2u);
  ASSERT_EQ(split_v2->attributes().size(), 2u);
  ASSERT_EQ(split_v11->attributes().size(), 2u);
  ASSERT_EQ(split_v13->attributes().size(), 1u);
  EXPECT_EQ(split_v13->attributes()[0].name, "axis");
  ASSERT_EQ(split_v18->attributes().size(), 2u);
  EXPECT_EQ(split_v18->attributes()[0].name, "axis");
  EXPECT_EQ(split_v18->attributes()[1].name, "num_outputs");

  EXPECT_FALSE(split_v18->doc().empty());
}

TEST(OnnxOpTensorRegistrationTest, ReturnsOneHotSchemaWithAxisAttribute) {
  const std::vector<core::schema::LightOpSchema> schemas =
      onnx_op::tensor::GetAllOnnxOpTensorSchemasWithHistory("OneHot");

  const core::schema::LightOpSchema *const one_hot_v11 = FindByVersion(schemas, 11);
  const core::schema::LightOpSchema *const one_hot_v9 = FindByVersion(schemas, 9);
  ASSERT_NE(nullptr, one_hot_v11);
  ASSERT_NE(nullptr, one_hot_v9);
  EXPECT_EQ(one_hot_v11->name(), "OneHot");
  EXPECT_EQ(one_hot_v11->domain(), core::schema::kOnnxDomain);
  EXPECT_EQ(one_hot_v11->since_version(), 11);

  ASSERT_EQ(one_hot_v11->inputs().size(), 3u);
  EXPECT_EQ(one_hot_v11->inputs()[0].name, "indices");
  EXPECT_EQ(one_hot_v11->inputs()[0].type, "T1");
  EXPECT_EQ(one_hot_v11->inputs()[1].name, "depth");
  EXPECT_EQ(one_hot_v11->inputs()[1].type, "T2");
  EXPECT_EQ(one_hot_v11->inputs()[2].name, "values");
  EXPECT_EQ(one_hot_v11->inputs()[2].type, "T3");

  ASSERT_EQ(one_hot_v11->outputs().size(), 1u);
  EXPECT_EQ(one_hot_v11->outputs()[0].name, "output");
  EXPECT_EQ(one_hot_v11->outputs()[0].type, "T3");

  ASSERT_EQ(one_hot_v11->type_constraints().size(), 3u);
  EXPECT_EQ(one_hot_v11->type_constraints()[0].type_param_str, "T1");
  EXPECT_EQ(one_hot_v11->type_constraints()[1].type_param_str, "T2");
  EXPECT_EQ(one_hot_v11->type_constraints()[2].type_param_str, "T3");
  EXPECT_EQ(one_hot_v11->type_constraints()[0].allowed_type_strs, core::schema::AllNumericTypes());
  EXPECT_EQ(one_hot_v11->type_constraints()[1].allowed_type_strs, core::schema::AllNumericTypes());
  EXPECT_EQ(one_hot_v11->type_constraints()[2].allowed_type_strs, core::schema::AllTensorTypes());

  ASSERT_EQ(one_hot_v11->attributes().size(), 1u);
  EXPECT_EQ(one_hot_v11->attributes()[0].name, "axis");
  EXPECT_EQ(one_hot_v11->attributes()[0].type, core::schema::AttributeType::INT);
  EXPECT_FALSE(one_hot_v11->attributes()[0].required);
  ASSERT_TRUE(std::holds_alternative<int64_t>(one_hot_v11->attributes()[0].default_value));
  EXPECT_EQ(std::get<int64_t>(one_hot_v11->attributes()[0].default_value), -1);

  EXPECT_FALSE(one_hot_v11->doc().empty());
  EXPECT_FALSE(one_hot_v9->doc().empty());
  EXPECT_EQ(one_hot_v9->since_version(), 9);
  ASSERT_EQ(one_hot_v9->inputs().size(), 3u);
}

} // namespace Test
