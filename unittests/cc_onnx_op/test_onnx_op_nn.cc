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
constexpr size_t kExpectedDropoutSchemaCount = 7;
constexpr size_t kExpectedGRUSchemaCount = 5;
constexpr size_t kExpectedLSTMSchemaCount = 4;
constexpr size_t kExpectedRNNSchemaCount = 4;
constexpr size_t kExpectedNnSchemaCount =
    kExpectedAttentionSchemaCount + kExpectedAveragePoolSchemaCount +
    kExpectedBatchNormalizationSchemaCount + kExpectedDropoutSchemaCount + kExpectedGRUSchemaCount +
    kExpectedLSTMSchemaCount + kExpectedRNNSchemaCount;

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

} // namespace Test
