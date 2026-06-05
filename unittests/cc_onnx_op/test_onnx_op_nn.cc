// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_nn.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef ONNX_LIGHT_NAMESPACE
#undef ONNX_LIGHT_NAMESPACE
#endif

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

constexpr size_t kExpectedAttentionSchemaCount = 2;
constexpr size_t kExpectedAveragePoolSchemaCount = 6;
constexpr size_t kExpectedBatchNormalizationSchemaCount = 6;
constexpr size_t kExpectedCol2ImSchemaCount = 1;
constexpr size_t kExpectedConvSchemaCount = 3;
constexpr size_t kExpectedConvIntegerSchemaCount = 1;
constexpr size_t kExpectedConvTransposeSchemaCount = 3;
constexpr size_t kExpectedDeformConvSchemaCount = 2;
constexpr size_t kExpectedDropoutSchemaCount = 7;
constexpr size_t kExpectedFlattenSchemaCount = 8;
constexpr size_t kExpectedGlobalAveragePoolSchemaCount = 2;
constexpr size_t kExpectedGlobalLpPoolSchemaCount = 3;
constexpr size_t kExpectedGlobalMaxPoolSchemaCount = 2;
constexpr size_t kExpectedGRUSchemaCount = 5;
constexpr size_t kExpectedGroupNormalizationSchemaCount = 2;
constexpr size_t kExpectedInstanceNormalizationSchemaCount = 3;
constexpr size_t kExpectedMeanVarianceNormalizationSchemaCount = 2;
constexpr size_t kExpectedLRNSchemaCount = 2;
constexpr size_t kExpectedLpNormalizationSchemaCount = 2;
constexpr size_t kExpectedLSTMSchemaCount = 4;
constexpr size_t kExpectedMaxPoolSchemaCount = 6;
constexpr size_t kExpectedMaxRoiPoolSchemaCount = 2;
constexpr size_t kExpectedMaxUnpoolSchemaCount = 3;
constexpr size_t kExpectedRNNSchemaCount = 4;
constexpr size_t kExpectedNnSchemaCount =
    kExpectedAttentionSchemaCount + kExpectedAveragePoolSchemaCount +
    kExpectedBatchNormalizationSchemaCount + kExpectedCol2ImSchemaCount + kExpectedConvSchemaCount +
    kExpectedConvIntegerSchemaCount + kExpectedConvTransposeSchemaCount +
    kExpectedDeformConvSchemaCount + kExpectedGlobalAveragePoolSchemaCount +
    kExpectedDropoutSchemaCount + kExpectedGlobalLpPoolSchemaCount + kExpectedFlattenSchemaCount +
    kExpectedGlobalMaxPoolSchemaCount + kExpectedGRUSchemaCount +
    kExpectedGroupNormalizationSchemaCount + kExpectedInstanceNormalizationSchemaCount +
    kExpectedLRNSchemaCount + kExpectedLpNormalizationSchemaCount + kExpectedLSTMSchemaCount +
    kExpectedMaxPoolSchemaCount + kExpectedMaxRoiPoolSchemaCount + kExpectedMaxUnpoolSchemaCount +
    kExpectedMeanVarianceNormalizationSchemaCount + kExpectedRNNSchemaCount;

static const onnx_op::LightOpSchema *
FindByVersion(const std::vector<onnx_op::LightOpSchema> &schemas, int version) {
  for (const auto &schema : schemas) {
    if (schema.since_version() == version) {
      return &schema;
    }
  }
  return nullptr;
}

TEST(OnnxOpNnRegistrationTest, ReturnsAveragePoolSchemasWithoutShapeInference) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> average_pool_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("AveragePool");

  EXPECT_EQ(schemas.size(), kExpectedNnSchemaCount);

  const onnx_op::LightOpSchema *const ap_v22 = FindByVersion(average_pool_schemas, 22);
  const onnx_op::LightOpSchema *const ap_v19 = FindByVersion(average_pool_schemas, 19);
  const onnx_op::LightOpSchema *const ap_v11 = FindByVersion(average_pool_schemas, 11);
  const onnx_op::LightOpSchema *const ap_v10 = FindByVersion(average_pool_schemas, 10);
  const onnx_op::LightOpSchema *const ap_v7 = FindByVersion(average_pool_schemas, 7);
  const onnx_op::LightOpSchema *const ap_v1 = FindByVersion(average_pool_schemas, 1);

  ASSERT_NE(nullptr, ap_v22);
  ASSERT_NE(nullptr, ap_v19);
  ASSERT_NE(nullptr, ap_v11);
  ASSERT_NE(nullptr, ap_v10);
  ASSERT_NE(nullptr, ap_v7);
  ASSERT_NE(nullptr, ap_v1);

  EXPECT_EQ(ap_v22->domain(), "ai.onnx");
  EXPECT_EQ(ap_v22->inputs().size(), 1u);
  EXPECT_EQ(ap_v22->outputs().size(), 1u);
  EXPECT_EQ(ap_v22->type_constraints().size(), 1u);
  EXPECT_EQ(ap_v22->inputs()[0].name, "X");
  EXPECT_EQ(ap_v22->inputs()[0].type, "T");
  EXPECT_EQ(ap_v22->outputs()[0].name, "Y");
  EXPECT_EQ(ap_v22->outputs()[0].type, "T");
  EXPECT_EQ(ap_v22->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(ap_v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(ap_v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);

  EXPECT_EQ(ap_v19->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v11->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v10->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v7->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[1], onnx_op::TensorType::kFloat);
  EXPECT_EQ(ap_v1->type_constraints()[0].allowed_type_strs[2], onnx_op::TensorType::kDouble);

  EXPECT_FALSE(ap_v1->doc().empty());
  EXPECT_FALSE(ap_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsMaxPoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> mp_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("MaxPool");

  EXPECT_EQ(mp_schemas.size(), kExpectedMaxPoolSchemaCount);

  const onnx_op::LightOpSchema *const mp_v22 = FindByVersion(mp_schemas, 22);
  const onnx_op::LightOpSchema *const mp_v12 = FindByVersion(mp_schemas, 12);
  const onnx_op::LightOpSchema *const mp_v11 = FindByVersion(mp_schemas, 11);
  const onnx_op::LightOpSchema *const mp_v10 = FindByVersion(mp_schemas, 10);
  const onnx_op::LightOpSchema *const mp_v8 = FindByVersion(mp_schemas, 8);
  const onnx_op::LightOpSchema *const mp_v1 = FindByVersion(mp_schemas, 1);

  ASSERT_NE(nullptr, mp_v22);
  ASSERT_NE(nullptr, mp_v12);
  ASSERT_NE(nullptr, mp_v11);
  ASSERT_NE(nullptr, mp_v10);
  ASSERT_NE(nullptr, mp_v8);
  ASSERT_NE(nullptr, mp_v1);

  // Opset 1 has a single output (Y); the second output ``Indices`` was
  // added in opset 8.
  EXPECT_EQ(mp_v1->inputs().size(), 1u);
  EXPECT_EQ(mp_v1->outputs().size(), 1u);
  EXPECT_EQ(mp_v8->outputs().size(), 2u);
  EXPECT_EQ(mp_v8->outputs()[1].name, "Indices");

  EXPECT_EQ(mp_v22->domain(), "ai.onnx");
  EXPECT_EQ(mp_v22->inputs().size(), 1u);
  EXPECT_EQ(mp_v22->outputs().size(), 2u);
  EXPECT_EQ(mp_v22->inputs()[0].name, "X");
  EXPECT_EQ(mp_v22->outputs()[0].name, "Y");
  EXPECT_EQ(mp_v22->outputs()[1].name, "Indices");

  // Type-constraint widening across opsets: 3 → 5 → 6 floating/integer types.
  EXPECT_EQ(mp_v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mp_v8->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mp_v10->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mp_v11->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mp_v12->type_constraints()[0].allowed_type_strs.size(), 5u);
  EXPECT_EQ(mp_v22->type_constraints()[0].allowed_type_strs.size(), 6u);

  EXPECT_FALSE(mp_v1->doc().empty());
  EXPECT_FALSE(mp_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsMaxUnpoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> mu_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("MaxUnpool");

  EXPECT_EQ(mu_schemas.size(), kExpectedMaxUnpoolSchemaCount);

  const onnx_op::LightOpSchema *const mu_v22 = FindByVersion(mu_schemas, 22);
  const onnx_op::LightOpSchema *const mu_v11 = FindByVersion(mu_schemas, 11);
  const onnx_op::LightOpSchema *const mu_v9 = FindByVersion(mu_schemas, 9);

  ASSERT_NE(nullptr, mu_v22);
  ASSERT_NE(nullptr, mu_v11);
  ASSERT_NE(nullptr, mu_v9);

  EXPECT_EQ(mu_v22->domain(), "ai.onnx");
  EXPECT_EQ(mu_v22->inputs().size(), 3u);
  EXPECT_EQ(mu_v22->outputs().size(), 1u);
  EXPECT_EQ(mu_v22->inputs()[0].name, "X");
  EXPECT_EQ(mu_v22->inputs()[1].name, "I");
  EXPECT_EQ(mu_v22->inputs()[2].name, "output_shape");
  EXPECT_EQ(mu_v22->outputs()[0].name, "output");
  EXPECT_EQ(mu_v22->type_constraints().size(), 2u);

  // T1 widens from 3 to 4 float types in opset 22; T2 is always int64.
  EXPECT_EQ(mu_v9->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mu_v11->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mu_v22->type_constraints()[0].allowed_type_strs.size(), 4u);

  EXPECT_FALSE(mu_v9->doc().empty());
  EXPECT_FALSE(mu_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsMaxRoiPoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> mrp_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("MaxRoiPool");

  EXPECT_EQ(mrp_schemas.size(), kExpectedMaxRoiPoolSchemaCount);

  const onnx_op::LightOpSchema *const mrp_v22 = FindByVersion(mrp_schemas, 22);
  const onnx_op::LightOpSchema *const mrp_v1 = FindByVersion(mrp_schemas, 1);

  ASSERT_NE(nullptr, mrp_v22);
  ASSERT_NE(nullptr, mrp_v1);

  EXPECT_EQ(mrp_v22->domain(), "ai.onnx");
  EXPECT_EQ(mrp_v22->inputs().size(), 2u);
  EXPECT_EQ(mrp_v22->outputs().size(), 1u);
  EXPECT_EQ(mrp_v22->inputs()[0].name, "X");
  EXPECT_EQ(mrp_v22->inputs()[1].name, "rois");
  EXPECT_EQ(mrp_v22->outputs()[0].name, "Y");
  EXPECT_EQ(mrp_v22->type_constraints().size(), 1u);

  // T widens from 3 to 4 float types in opset 22 (adds bfloat16).
  EXPECT_EQ(mrp_v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(mrp_v22->type_constraints()[0].allowed_type_strs.size(), 4u);

  // Both versions have the same two attributes: pooled_shape (required) and
  // spatial_scale (default 1.0).
  ASSERT_EQ(mrp_v22->attributes().size(), 2u);
  EXPECT_EQ(mrp_v22->attributes()[0].name, "pooled_shape");
  EXPECT_TRUE(mrp_v22->attributes()[0].required);
  EXPECT_EQ(mrp_v22->attributes()[1].name, "spatial_scale");
  EXPECT_FALSE(mrp_v22->attributes()[1].required);

  EXPECT_FALSE(mrp_v1->doc().empty());
  EXPECT_FALSE(mrp_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsRNNSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> rnn_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("RNN");

  const onnx_op::LightOpSchema *const rnn_v22 = FindByVersion(rnn_schemas, 22);
  const onnx_op::LightOpSchema *const rnn_v14 = FindByVersion(rnn_schemas, 14);
  const onnx_op::LightOpSchema *const rnn_v7 = FindByVersion(rnn_schemas, 7);
  const onnx_op::LightOpSchema *const rnn_v1 = FindByVersion(rnn_schemas, 1);

  ASSERT_NE(nullptr, rnn_v22);
  ASSERT_NE(nullptr, rnn_v14);
  ASSERT_NE(nullptr, rnn_v7);
  ASSERT_NE(nullptr, rnn_v1);

  EXPECT_EQ(rnn_v22->domain(), "ai.onnx");
  EXPECT_EQ(rnn_v22->inputs().size(), 6u);
  EXPECT_EQ(rnn_v22->outputs().size(), 2u);
  EXPECT_EQ(rnn_v22->type_constraints().size(), 2u);
  EXPECT_EQ(rnn_v22->inputs()[0].name, "X");
  EXPECT_EQ(rnn_v22->inputs()[1].name, "W");
  EXPECT_EQ(rnn_v22->inputs()[2].name, "R");
  EXPECT_EQ(rnn_v22->inputs()[3].name, "B");
  EXPECT_EQ(rnn_v22->inputs()[4].name, "sequence_lens");
  EXPECT_EQ(rnn_v22->inputs()[5].name, "initial_h");
  EXPECT_EQ(rnn_v22->outputs()[0].name, "Y");
  EXPECT_EQ(rnn_v22->outputs()[1].name, "Y_h");
  EXPECT_EQ(rnn_v22->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(rnn_v22->type_constraints()[1].type_param_str, "T1");
  EXPECT_EQ(rnn_v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(rnn_v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
  EXPECT_EQ(rnn_v22->type_constraints()[1].allowed_type_strs.size(), 1u);
  EXPECT_EQ(rnn_v22->type_constraints()[1].allowed_type_strs[0], onnx_op::TensorType::kInt32);

  EXPECT_EQ(rnn_v14->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(rnn_v14->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);

  EXPECT_NE(rnn_v1->outputs()[0].description, rnn_v22->outputs()[0].description);

  EXPECT_FALSE(rnn_v1->doc().empty());
  EXPECT_FALSE(rnn_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsGRUSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> gru_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("GRU");

  const onnx_op::LightOpSchema *const gru_v22 = FindByVersion(gru_schemas, 22);
  const onnx_op::LightOpSchema *const gru_v14 = FindByVersion(gru_schemas, 14);
  const onnx_op::LightOpSchema *const gru_v7 = FindByVersion(gru_schemas, 7);
  const onnx_op::LightOpSchema *const gru_v3 = FindByVersion(gru_schemas, 3);
  const onnx_op::LightOpSchema *const gru_v1 = FindByVersion(gru_schemas, 1);

  ASSERT_NE(nullptr, gru_v22);
  ASSERT_NE(nullptr, gru_v14);
  ASSERT_NE(nullptr, gru_v7);
  ASSERT_NE(nullptr, gru_v3);
  ASSERT_NE(nullptr, gru_v1);

  EXPECT_EQ(gru_v22->inputs().size(), 6u);
  EXPECT_EQ(gru_v22->outputs().size(), 2u);
  EXPECT_EQ(gru_v22->type_constraints().size(), 2u);

  EXPECT_FALSE(gru_v1->doc().empty());
  EXPECT_FALSE(gru_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsLSTMSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> lstm_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("LSTM");

  const onnx_op::LightOpSchema *const lstm_v22 = FindByVersion(lstm_schemas, 22);
  const onnx_op::LightOpSchema *const lstm_v14 = FindByVersion(lstm_schemas, 14);
  const onnx_op::LightOpSchema *const lstm_v7 = FindByVersion(lstm_schemas, 7);
  const onnx_op::LightOpSchema *const lstm_v1 = FindByVersion(lstm_schemas, 1);

  ASSERT_NE(nullptr, lstm_v22);
  ASSERT_NE(nullptr, lstm_v14);
  ASSERT_NE(nullptr, lstm_v7);
  ASSERT_NE(nullptr, lstm_v1);

  EXPECT_EQ(lstm_v22->inputs().size(), 8u);
  EXPECT_EQ(lstm_v22->outputs().size(), 3u);
  EXPECT_EQ(lstm_v22->type_constraints().size(), 2u);
  EXPECT_EQ(lstm_v22->inputs()[6].name, "initial_c");
  EXPECT_EQ(lstm_v22->inputs()[7].name, "P");
  EXPECT_EQ(lstm_v22->outputs()[2].name, "Y_c");

  EXPECT_FALSE(lstm_v1->doc().empty());
  EXPECT_FALSE(lstm_v22->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsBatchNormalizationSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory();
  const std::vector<onnx_op::LightOpSchema> batch_normalization_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("BatchNormalization");

  const onnx_op::LightOpSchema *const bn_v15 = FindByVersion(batch_normalization_schemas, 15);
  const onnx_op::LightOpSchema *const bn_v14 = FindByVersion(batch_normalization_schemas, 14);
  const onnx_op::LightOpSchema *const bn_v9 = FindByVersion(batch_normalization_schemas, 9);
  const onnx_op::LightOpSchema *const bn_v7 = FindByVersion(batch_normalization_schemas, 7);
  const onnx_op::LightOpSchema *const bn_v6 = FindByVersion(batch_normalization_schemas, 6);
  const onnx_op::LightOpSchema *const bn_v1 = FindByVersion(batch_normalization_schemas, 1);

  ASSERT_NE(nullptr, bn_v15);
  ASSERT_NE(nullptr, bn_v14);
  ASSERT_NE(nullptr, bn_v9);
  ASSERT_NE(nullptr, bn_v7);
  ASSERT_NE(nullptr, bn_v6);
  ASSERT_NE(nullptr, bn_v1);

  // v15: 5 inputs / 3 outputs / 3 type constraints (T, T1, T2).
  EXPECT_EQ(bn_v15->domain(), "ai.onnx");
  EXPECT_EQ(bn_v15->inputs().size(), 5u);
  EXPECT_EQ(bn_v15->outputs().size(), 3u);
  EXPECT_EQ(bn_v15->type_constraints().size(), 3u);
  EXPECT_EQ(bn_v15->inputs()[0].name, "X");
  EXPECT_EQ(bn_v15->inputs()[3].name, "input_mean");
  EXPECT_EQ(bn_v15->outputs()[0].name, "Y");
  EXPECT_EQ(bn_v15->outputs()[1].name, "running_mean");
  EXPECT_EQ(bn_v15->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(bn_v15->type_constraints()[1].type_param_str, "T1");
  EXPECT_EQ(bn_v15->type_constraints()[2].type_param_str, "T2");
  EXPECT_EQ(bn_v15->inputs()[1].type, "T1");
  EXPECT_EQ(bn_v15->inputs()[3].type, "T2");
  EXPECT_EQ(bn_v15->outputs()[1].type, "T2");

  // v14: 5 inputs / 3 outputs / 2 type constraints (T, U).
  EXPECT_EQ(bn_v14->inputs().size(), 5u);
  EXPECT_EQ(bn_v14->outputs().size(), 3u);
  EXPECT_EQ(bn_v14->type_constraints().size(), 2u);
  EXPECT_EQ(bn_v14->type_constraints()[1].type_param_str, "U");
  EXPECT_EQ(bn_v14->inputs()[3].type, "U");
  EXPECT_EQ(bn_v14->outputs()[1].type, "U");

  // v9: 5 inputs / 5 outputs / 1 type constraint, no spatial attribute hooks.
  EXPECT_EQ(bn_v9->inputs().size(), 5u);
  EXPECT_EQ(bn_v9->outputs().size(), 5u);
  EXPECT_EQ(bn_v9->type_constraints().size(), 1u);
  EXPECT_EQ(bn_v9->outputs()[3].name, "saved_mean");

  // v7 / v6 / v1 all have 5 inputs / 5 outputs.
  EXPECT_EQ(bn_v7->inputs().size(), 5u);
  EXPECT_EQ(bn_v7->outputs().size(), 5u);
  EXPECT_EQ(bn_v6->inputs().size(), 5u);
  EXPECT_EQ(bn_v6->outputs().size(), 5u);
  EXPECT_EQ(bn_v1->inputs().size(), 5u);
  EXPECT_EQ(bn_v1->outputs().size(), 5u);

  // Type-set width by opset.
  EXPECT_EQ(bn_v15->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(bn_v14->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(bn_v9->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(bn_v7->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(bn_v6->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(bn_v1->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_FALSE(bn_v1->doc().empty());
  EXPECT_FALSE(bn_v15->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsDropoutSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> dropout_schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("Dropout");

  const onnx_op::LightOpSchema *const d_v22 = FindByVersion(dropout_schemas, 22);
  const onnx_op::LightOpSchema *const d_v13 = FindByVersion(dropout_schemas, 13);
  const onnx_op::LightOpSchema *const d_v12 = FindByVersion(dropout_schemas, 12);
  const onnx_op::LightOpSchema *const d_v10 = FindByVersion(dropout_schemas, 10);
  const onnx_op::LightOpSchema *const d_v7 = FindByVersion(dropout_schemas, 7);
  const onnx_op::LightOpSchema *const d_v6 = FindByVersion(dropout_schemas, 6);
  const onnx_op::LightOpSchema *const d_v1 = FindByVersion(dropout_schemas, 1);

  ASSERT_NE(nullptr, d_v22);
  ASSERT_NE(nullptr, d_v13);
  ASSERT_NE(nullptr, d_v12);
  ASSERT_NE(nullptr, d_v10);
  ASSERT_NE(nullptr, d_v7);
  ASSERT_NE(nullptr, d_v6);
  ASSERT_NE(nullptr, d_v1);

  EXPECT_EQ(d_v22->inputs().size(), 3u);
  EXPECT_EQ(d_v22->outputs().size(), 2u);
  EXPECT_EQ(d_v22->type_constraints().size(), 3u);
  EXPECT_EQ(d_v22->outputs()[1].type, "T2");
  EXPECT_EQ(d_v22->type_constraints()[0].allowed_type_strs.size(), 8u);
  EXPECT_EQ(d_v22->type_constraints()[1].allowed_type_strs.size(), 8u);

  EXPECT_EQ(d_v13->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(d_v13->type_constraints()[1].allowed_type_strs.size(), 3u);

  EXPECT_EQ(d_v12->inputs().size(), 3u);
  EXPECT_EQ(d_v12->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(d_v12->outputs()[1].type, "T2");

  EXPECT_EQ(d_v10->inputs().size(), 1u);
  EXPECT_EQ(d_v10->outputs()[1].type, "T1");
  EXPECT_EQ(d_v10->type_constraints().size(), 2u);

  EXPECT_EQ(d_v7->outputs()[1].type, "T");
  EXPECT_EQ(d_v6->outputs()[1].description,
            "The output mask. If is_test is nonzero, this output is not filled.");
  EXPECT_EQ(d_v1->outputs()[1].type, "T");
}

TEST(OnnxOpNnRegistrationTest, RecurrentSchemasStripDocsWhenRequested) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory(/*op_type=*/"", /*init_doc=*/false);
  for (const auto &schema : schemas) {
    EXPECT_TRUE(schema.doc().empty());
  }
}

TEST(OnnxOpNnRegistrationTest, ReturnsGlobalAveragePoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("GlobalAveragePool");

  ASSERT_EQ(schemas.size(), kExpectedGlobalAveragePoolSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v1);

  EXPECT_EQ(v22->domain(), "ai.onnx");
  EXPECT_EQ(v22->inputs().size(), 1u);
  EXPECT_EQ(v22->outputs().size(), 1u);
  EXPECT_EQ(v22->type_constraints().size(), 1u);
  EXPECT_EQ(v22->inputs()[0].name, "X");
  EXPECT_EQ(v22->outputs()[0].name, "Y");
  EXPECT_EQ(v22->type_constraints()[0].type_param_str, "T");
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);

  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);

  EXPECT_FALSE(v22->doc().empty());
  EXPECT_FALSE(v1->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsGlobalMaxPoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("GlobalMaxPool");

  ASSERT_EQ(schemas.size(), kExpectedGlobalMaxPoolSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v1);

  EXPECT_EQ(v22->domain(), "ai.onnx");
  EXPECT_EQ(v22->inputs().size(), 1u);
  EXPECT_EQ(v22->outputs().size(), 1u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_FALSE(v22->doc().empty());
  EXPECT_FALSE(v1->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsGlobalLpPoolSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("GlobalLpPool");

  ASSERT_EQ(schemas.size(), kExpectedGlobalLpPoolSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v2 = FindByVersion(schemas, 2);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v2);
  ASSERT_NE(nullptr, v1);

  EXPECT_EQ(v22->domain(), "ai.onnx");
  EXPECT_EQ(v22->inputs().size(), 1u);
  EXPECT_EQ(v22->outputs().size(), 1u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v2->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);

  EXPECT_FALSE(v22->doc().empty());
  EXPECT_FALSE(v2->doc().empty());
  EXPECT_FALSE(v1->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsCol2ImSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("Col2Im");

  ASSERT_EQ(schemas.size(), kExpectedCol2ImSchemaCount);

  const onnx_op::LightOpSchema *const v18 = FindByVersion(schemas, 18);
  ASSERT_NE(nullptr, v18);

  EXPECT_EQ(v18->name(), "Col2Im");
  EXPECT_EQ(v18->domain(), "ai.onnx");
  ASSERT_EQ(v18->inputs().size(), 3u);
  EXPECT_EQ(v18->inputs()[0].name, "input");
  EXPECT_EQ(v18->inputs()[0].type, "T");
  EXPECT_EQ(v18->inputs()[1].name, "image_shape");
  EXPECT_EQ(v18->inputs()[1].type, "tensor(int64)");
  EXPECT_EQ(v18->inputs()[2].name, "block_shape");
  EXPECT_EQ(v18->inputs()[2].type, "tensor(int64)");
  ASSERT_EQ(v18->outputs().size(), 1u);
  EXPECT_EQ(v18->outputs()[0].name, "output");
  EXPECT_EQ(v18->outputs()[0].type, "T");
  ASSERT_EQ(v18->type_constraints().size(), 1u);
  EXPECT_EQ(v18->type_constraints()[0].type_param_str, "T");
  ASSERT_EQ(v18->attributes().size(), 3u);
  EXPECT_EQ(v18->attributes()[0].name, "dilations");
  EXPECT_EQ(v18->attributes()[0].type, onnx_op::AttributeType::INTS);
  EXPECT_FALSE(v18->attributes()[0].required);
  EXPECT_EQ(v18->attributes()[1].name, "pads");
  EXPECT_EQ(v18->attributes()[1].type, onnx_op::AttributeType::INTS);
  EXPECT_FALSE(v18->attributes()[1].required);
  EXPECT_EQ(v18->attributes()[2].name, "strides");
  EXPECT_EQ(v18->attributes()[2].type, onnx_op::AttributeType::INTS);
  EXPECT_FALSE(v18->attributes()[2].required);
  EXPECT_FALSE(v18->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsLRNSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("LRN");

  ASSERT_EQ(schemas.size(), kExpectedLRNSchemaCount);

  const onnx_op::LightOpSchema *const v13 = FindByVersion(schemas, 13);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v13}) {
    SCOPED_TRACE("LRN@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "LRN");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[0].type, "T");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    EXPECT_EQ(schema->outputs()[0].type, "T");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    ASSERT_EQ(schema->attributes().size(), 4u);
    EXPECT_EQ(schema->attributes()[0].name, "alpha");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::FLOAT);
    EXPECT_FALSE(schema->attributes()[0].required);
    EXPECT_EQ(schema->attributes()[1].name, "beta");
    EXPECT_EQ(schema->attributes()[1].type, onnx_op::AttributeType::FLOAT);
    EXPECT_FALSE(schema->attributes()[1].required);
    EXPECT_EQ(schema->attributes()[2].name, "bias");
    EXPECT_EQ(schema->attributes()[2].type, onnx_op::AttributeType::FLOAT);
    EXPECT_FALSE(schema->attributes()[2].required);
    EXPECT_EQ(schema->attributes()[3].name, "size");
    EXPECT_EQ(schema->attributes()[3].type, onnx_op::AttributeType::INT);
    EXPECT_TRUE(schema->attributes()[3].required);
    EXPECT_FALSE(schema->doc().empty());
  }

  // v13 adds bfloat16 to the type constraint set.
  EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
}

TEST(OnnxOpNnRegistrationTest, ReturnsLpNormalizationSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("LpNormalization");

  ASSERT_EQ(schemas.size(), kExpectedLpNormalizationSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v22}) {
    SCOPED_TRACE("LpNormalization@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "LpNormalization");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    EXPECT_EQ(schema->outputs()[0].type, "T");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    ASSERT_EQ(schema->attributes().size(), 2u);
    EXPECT_EQ(schema->attributes()[0].name, "axis");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[0].required);
    EXPECT_EQ(schema->attributes()[1].name, "p");
    EXPECT_EQ(schema->attributes()[1].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->attributes()[1].required);
    EXPECT_FALSE(schema->doc().empty());
  }

  // v22 adds bfloat16 to the type constraint set.
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
}

TEST(OnnxOpNnRegistrationTest, ReturnsFlattenSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("Flatten");

  ASSERT_EQ(schemas.size(), kExpectedFlattenSchemaCount);

  const onnx_op::LightOpSchema *const v25 = FindByVersion(schemas, 25);
  const onnx_op::LightOpSchema *const v24 = FindByVersion(schemas, 24);
  const onnx_op::LightOpSchema *const v23 = FindByVersion(schemas, 23);
  const onnx_op::LightOpSchema *const v21 = FindByVersion(schemas, 21);
  const onnx_op::LightOpSchema *const v13 = FindByVersion(schemas, 13);
  const onnx_op::LightOpSchema *const v11 = FindByVersion(schemas, 11);
  const onnx_op::LightOpSchema *const v9 = FindByVersion(schemas, 9);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v25);
  ASSERT_NE(nullptr, v24);
  ASSERT_NE(nullptr, v23);
  ASSERT_NE(nullptr, v21);
  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v11);
  ASSERT_NE(nullptr, v9);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v9, v11, v13, v21, v23, v24, v25}) {
    SCOPED_TRACE("Flatten@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "Flatten");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[0].type, "T");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    EXPECT_EQ(schema->outputs()[0].type, "T");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    ASSERT_EQ(schema->attributes().size(), 1u);
    EXPECT_EQ(schema->attributes()[0].name, "axis");
    EXPECT_EQ(schema->attributes()[0].type, onnx_op::AttributeType::INT);
    EXPECT_FALSE(schema->doc().empty());
  }

  // v1 only supports float types; v9+ support all tensor types.
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_GT(v9->type_constraints()[0].allowed_type_strs.size(), 3u);
}

TEST(OnnxOpNnRegistrationTest, ReturnsDeformConvSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("DeformConv");

  ASSERT_EQ(schemas.size(), kExpectedDeformConvSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v19 = FindByVersion(schemas, 19);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v19);

  for (const onnx_op::LightOpSchema *schema : {v19, v22}) {
    SCOPED_TRACE("DeformConv@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "DeformConv");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 5u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[1].name, "W");
    EXPECT_EQ(schema->inputs()[2].name, "offset");
    EXPECT_EQ(schema->inputs()[3].name, "B");
    EXPECT_EQ(schema->inputs()[4].name, "mask");
    for (const auto &input : schema->inputs()) {
      EXPECT_EQ(input.type, "T");
    }
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    EXPECT_EQ(schema->outputs()[0].type, "T");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].type_param_str, "T");
    EXPECT_FALSE(schema->doc().empty());
  }

  // Opset 22 widens T to include bfloat16.
  EXPECT_EQ(v19->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v19->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kFloat16);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
}

TEST(OnnxOpNnRegistrationTest, ReturnsConvSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("Conv");

  ASSERT_EQ(schemas.size(), kExpectedConvSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v11 = FindByVersion(schemas, 11);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v11);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v11, v22}) {
    SCOPED_TRACE("Conv@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "Conv");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 3u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[1].name, "W");
    EXPECT_EQ(schema->inputs()[2].name, "B");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_FALSE(schema->doc().empty());
  }

  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v11->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
}

TEST(OnnxOpNnRegistrationTest, ReturnsConvIntegerSchemaForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("ConvInteger");

  ASSERT_EQ(schemas.size(), kExpectedConvIntegerSchemaCount);

  const onnx_op::LightOpSchema *const v10 = FindByVersion(schemas, 10);
  ASSERT_NE(nullptr, v10);

  EXPECT_EQ(v10->name(), "ConvInteger");
  EXPECT_EQ(v10->domain(), "ai.onnx");
  ASSERT_EQ(v10->inputs().size(), 4u);
  EXPECT_EQ(v10->inputs()[0].name, "x");
  EXPECT_EQ(v10->inputs()[1].name, "w");
  EXPECT_EQ(v10->inputs()[2].name, "x_zero_point");
  EXPECT_EQ(v10->inputs()[3].name, "w_zero_point");
  ASSERT_EQ(v10->outputs().size(), 1u);
  EXPECT_EQ(v10->outputs()[0].name, "y");
  ASSERT_EQ(v10->type_constraints().size(), 3u);
  EXPECT_FALSE(v10->doc().empty());
}

TEST(OnnxOpNnRegistrationTest, ReturnsConvTransposeSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("ConvTranspose");

  ASSERT_EQ(schemas.size(), kExpectedConvTransposeSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v11 = FindByVersion(schemas, 11);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);

  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v11);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v11, v22}) {
    SCOPED_TRACE("ConvTranspose@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "ConvTranspose");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 3u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[1].name, "W");
    EXPECT_EQ(schema->inputs()[2].name, "B");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_FALSE(schema->doc().empty());
  }

  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs[0], onnx_op::TensorType::kBfloat16);
}

TEST(OnnxOpNnRegistrationTest, ReturnsInstanceNormalizationSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("InstanceNormalization");
  ASSERT_EQ(schemas.size(), kExpectedInstanceNormalizationSchemaCount);

  const onnx_op::LightOpSchema *const v22 = FindByVersion(schemas, 22);
  const onnx_op::LightOpSchema *const v6 = FindByVersion(schemas, 6);
  const onnx_op::LightOpSchema *const v1 = FindByVersion(schemas, 1);
  ASSERT_NE(nullptr, v22);
  ASSERT_NE(nullptr, v6);
  ASSERT_NE(nullptr, v1);

  for (const onnx_op::LightOpSchema *schema : {v1, v6, v22}) {
    SCOPED_TRACE("InstanceNormalization@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "InstanceNormalization");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 3u);
    EXPECT_EQ(schema->inputs()[0].name, "input");
    EXPECT_EQ(schema->inputs()[1].name, "scale");
    EXPECT_EQ(schema->inputs()[2].name, "B");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "output");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_FALSE(schema->doc().empty());
  }

  // v1 has the legacy ``consumed_inputs`` attribute in addition to ``epsilon``.
  EXPECT_EQ(v1->attributes().size(), 2u);
  EXPECT_EQ(v6->attributes().size(), 1u);
  EXPECT_EQ(v22->attributes().size(), 1u);

  // v22 widens to {bfloat16, float16, float, double}.
  EXPECT_EQ(v22->type_constraints()[0].allowed_type_strs.size(), 4u);
  // v1/v6 remain {float16, float, double}.
  EXPECT_EQ(v1->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v6->type_constraints()[0].allowed_type_strs.size(), 3u);
}

TEST(OnnxOpNnRegistrationTest, ReturnsGroupNormalizationSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("GroupNormalization");
  ASSERT_EQ(schemas.size(), kExpectedGroupNormalizationSchemaCount);

  const onnx_op::LightOpSchema *const v21 = FindByVersion(schemas, 21);
  const onnx_op::LightOpSchema *const v18 = FindByVersion(schemas, 18);
  ASSERT_NE(nullptr, v21);
  ASSERT_NE(nullptr, v18);

  for (const onnx_op::LightOpSchema *schema : {v18, v21}) {
    SCOPED_TRACE("GroupNormalization@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "GroupNormalization");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 3u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    EXPECT_EQ(schema->inputs()[1].name, "scale");
    EXPECT_EQ(schema->inputs()[2].name, "bias");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_EQ(schema->type_constraints()[0].allowed_type_strs.size(), 4u);
    EXPECT_FALSE(schema->doc().empty());
  }

  // v18 is deprecated and has only ``epsilon`` and ``num_groups``.
  EXPECT_TRUE(v18->deprecated());
  EXPECT_EQ(v18->attributes().size(), 2u);
  // v21 adds ``stash_type``.
  EXPECT_FALSE(v21->deprecated());
  EXPECT_EQ(v21->attributes().size(), 3u);
}

TEST(OnnxOpNnRegistrationTest, ReturnsMeanVarianceNormalizationSchemasForAllVersions) {
  const std::vector<onnx_op::LightOpSchema> schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("MeanVarianceNormalization");
  ASSERT_EQ(schemas.size(), kExpectedMeanVarianceNormalizationSchemaCount);

  const onnx_op::LightOpSchema *const v13 = FindByVersion(schemas, 13);
  const onnx_op::LightOpSchema *const v9 = FindByVersion(schemas, 9);
  ASSERT_NE(nullptr, v13);
  ASSERT_NE(nullptr, v9);

  for (const onnx_op::LightOpSchema *schema : {v9, v13}) {
    SCOPED_TRACE("MeanVarianceNormalization@" + std::to_string(schema->since_version()));
    EXPECT_EQ(schema->name(), "MeanVarianceNormalization");
    EXPECT_EQ(schema->domain(), "ai.onnx");
    ASSERT_EQ(schema->inputs().size(), 1u);
    EXPECT_EQ(schema->inputs()[0].name, "X");
    ASSERT_EQ(schema->outputs().size(), 1u);
    EXPECT_EQ(schema->outputs()[0].name, "Y");
    ASSERT_EQ(schema->attributes().size(), 1u);
    EXPECT_EQ(schema->attributes()[0].name, "axes");
    ASSERT_EQ(schema->type_constraints().size(), 1u);
    EXPECT_FALSE(schema->doc().empty());
  }

  EXPECT_EQ(v9->type_constraints()[0].allowed_type_strs.size(), 3u);
  EXPECT_EQ(v13->type_constraints()[0].allowed_type_strs.size(), 4u);
}

} // namespace Test
