// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectNNTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectNNTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, AveragePoolCasesArePresent) {
  auto cases = CollectTestCases("AveragePool");
  const TestCase *def = nullptr;
  const TestCase *strides = nullptr;
  const TestCase *pads = nullptr;
  const TestCase *avp_1d_default = nullptr;
  const TestCase *avp_2d_ceil = nullptr;
  const TestCase *avp_2d_ceil_last = nullptr;
  const TestCase *avp_2d_pads = nullptr;
  const TestCase *avp_2d_pre_pads = nullptr;
  const TestCase *avp_2d_pre_pads_cip = nullptr;
  const TestCase *avp_2d_pre_strides = nullptr;
  const TestCase *avp_3d_default = nullptr;
  const TestCase *avp_2d_pre_same_upper = nullptr;
  const TestCase *avp_2d_same_upper = nullptr;
  const TestCase *avp_2d_same_lower = nullptr;
  const TestCase *avp_2d_dilations = nullptr;
  const TestCase *avp_3d_dilations_small = nullptr;
  const TestCase *avp_3d_dil_large_0_T = nullptr;
  const TestCase *avp_3d_dil_large_0_F = nullptr;
  const TestCase *avp_3d_dil_large_1_T = nullptr;
  const TestCase *avp_3d_dil_large_1_F = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_averagepool_2d_default") {
      def = &c;
    } else if (c.name == "test_cc_averagepool_2d_strides") {
      strides = &c;
    } else if (c.name == "test_cc_averagepool_2d_pads_count_include_pad") {
      pads = &c;
    } else if (c.name == "test_cc_averagepool_1d_default") {
      avp_1d_default = &c;
    } else if (c.name == "test_cc_averagepool_2d_ceil") {
      avp_2d_ceil = &c;
    } else if (c.name == "test_cc_averagepool_2d_ceil_last_window_starts_on_pad") {
      avp_2d_ceil_last = &c;
    } else if (c.name == "test_cc_averagepool_2d_pads") {
      avp_2d_pads = &c;
    } else if (c.name == "test_cc_averagepool_2d_precomputed_pads") {
      avp_2d_pre_pads = &c;
    } else if (c.name == "test_cc_averagepool_2d_precomputed_pads_count_include_pad") {
      avp_2d_pre_pads_cip = &c;
    } else if (c.name == "test_cc_averagepool_2d_precomputed_strides") {
      avp_2d_pre_strides = &c;
    } else if (c.name == "test_cc_averagepool_3d_default") {
      avp_3d_default = &c;
    } else if (c.name == "test_cc_averagepool_2d_precomputed_same_upper") {
      avp_2d_pre_same_upper = &c;
    } else if (c.name == "test_cc_averagepool_2d_same_upper") {
      avp_2d_same_upper = &c;
    } else if (c.name == "test_cc_averagepool_2d_same_lower") {
      avp_2d_same_lower = &c;
    } else if (c.name == "test_cc_averagepool_2d_dilations") {
      avp_2d_dilations = &c;
    } else if (c.name == "test_cc_averagepool_3d_dilations_small") {
      avp_3d_dilations_small = &c;
    } else if (c.name ==
               "test_cc_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True") {
      avp_3d_dil_large_0_T = &c;
    } else if (c.name ==
               "test_cc_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False") {
      avp_3d_dil_large_0_F = &c;
    } else if (c.name ==
               "test_cc_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True") {
      avp_3d_dil_large_1_T = &c;
    } else if (c.name ==
               "test_cc_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False") {
      avp_3d_dil_large_1_F = &c;
    }
  }
  ASSERT_NE(def, nullptr);
  ASSERT_NE(strides, nullptr);
  ASSERT_NE(pads, nullptr);
  ASSERT_NE(avp_1d_default, nullptr);
  ASSERT_NE(avp_2d_ceil, nullptr);
  ASSERT_NE(avp_2d_ceil_last, nullptr);
  ASSERT_NE(avp_2d_pads, nullptr);
  ASSERT_NE(avp_2d_pre_pads, nullptr);
  ASSERT_NE(avp_2d_pre_pads_cip, nullptr);
  ASSERT_NE(avp_2d_pre_strides, nullptr);
  ASSERT_NE(avp_3d_default, nullptr);
  ASSERT_NE(avp_2d_pre_same_upper, nullptr);
  ASSERT_NE(avp_2d_same_upper, nullptr);
  ASSERT_NE(avp_2d_same_lower, nullptr);
  ASSERT_NE(avp_2d_dilations, nullptr);
  ASSERT_NE(avp_3d_dilations_small, nullptr);
  ASSERT_NE(avp_3d_dil_large_0_T, nullptr);
  ASSERT_NE(avp_3d_dil_large_0_F, nullptr);
  ASSERT_NE(avp_3d_dil_large_1_T, nullptr);
  ASSERT_NE(avp_3d_dil_large_1_F, nullptr);

  // Default 2x2 case: single input, single output of shape 1x1x3x3.
  {
    const GraphProto &graph = def->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "AveragePool");
    EXPECT_EQ(graph.ref_input().size(), 1u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(def->data_sets.size(), 1u);
    const auto &ds = def->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 1, 3, 3}));
    EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[0], 3.5f);
    EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[8], 13.5f);
  }

  // Strides case: 1x1x2x2 output.
  {
    const auto &ds = strides->data_sets[0];
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[0], 7.0f);
    EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[3], 19.0f);
  }

  // Pads + count_include_pad case: 1x1x5x5 output (input is 1x1x5x5 and
  // (5 + 2 - 3)/1 + 1 = 5 per spatial dim).
  {
    const auto &ds = pads->data_sets[0];
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 1, 5, 5}));
  }
}

TEST(BackendTestCase, BatchNormalizationCasesArePresent) {
  auto cases = CollectTestCases("BatchNormalization");
  const TestCase *example = nullptr;
  const TestCase *epsilon = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_batchnorm_example") {
      example = &c;
    } else if (c.name == "test_cc_batchnorm_epsilon") {
      epsilon = &c;
    }
  }
  ASSERT_NE(example, nullptr);
  ASSERT_NE(epsilon, nullptr);

  // Example case: BatchNormalization node with 5 inputs and 1 output of shape
  // 1x2x1x3.
  {
    const GraphProto &graph = example->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "BatchNormalization");
    EXPECT_EQ(graph.ref_input().size(), 5u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(example->data_sets.size(), 1u);
    const auto &ds = example->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 5u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 2, 1, 3}));
  }

  // Epsilon case: 2x3x4x5 output.
  {
    const auto &ds = epsilon->data_sets[0];
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 3, 4, 5}));
  }
}

TEST(BackendTestCase, InstanceNormalizationCasesArePresent) {
  auto cases = CollectTestCases("InstanceNormalization");
  const TestCase *example = nullptr;
  const TestCase *epsilon = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_instancenorm_example") {
      example = &c;
    } else if (c.name == "test_cc_instancenorm_epsilon") {
      epsilon = &c;
    }
  }
  ASSERT_NE(example, nullptr);
  ASSERT_NE(epsilon, nullptr);
  ASSERT_EQ(example->data_sets.size(), 1u);
  const auto &ds = example->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
}

TEST(BackendTestCase, GroupNormalizationCasesArePresent) {
  auto cases = CollectTestCases("GroupNormalization");
  const TestCase *example = nullptr;
  const TestCase *epsilon = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_group_normalization_example") {
      example = &c;
    } else if (c.name == "test_cc_group_normalization_epsilon") {
      epsilon = &c;
    }
  }
  ASSERT_NE(example, nullptr);
  ASSERT_NE(epsilon, nullptr);
  ASSERT_EQ(example->data_sets.size(), 1u);
  const auto &ds = example->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
}

TEST(BackendTestCase, MeanVarianceNormalizationCasesArePresent) {
  auto cases = CollectTestCases("MeanVarianceNormalization");
  const TestCase *mvn = nullptr;
  const TestCase *mvn_expanded = nullptr;
  const TestCase *mvn_expanded_ver18 = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_mvn") {
      mvn = &c;
    } else if (c.name == "test_cc_mvn_expanded") {
      mvn_expanded = &c;
    } else if (c.name == "test_cc_mvn_expanded_ver18") {
      mvn_expanded_ver18 = &c;
    }
  }
  ASSERT_NE(mvn, nullptr);
  ASSERT_NE(mvn_expanded, nullptr);
  ASSERT_NE(mvn_expanded_ver18, nullptr);
  ASSERT_EQ(mvn->data_sets.size(), 1u);
  const auto &ds = mvn->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
}

TEST(BackendTestCase, RNNCasesArePresent) {
  auto cases = CollectTestCases();
  const TestCase *defaults = nullptr;
  const TestCase *with_initial_bias = nullptr;
  const TestCase *seq_length = nullptr;
  const TestCase *batchwise = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_simple_rnn_defaults") {
      defaults = &c;
    } else if (c.name == "test_cc_simple_rnn_with_initial_bias") {
      with_initial_bias = &c;
    } else if (c.name == "test_cc_rnn_seq_length") {
      seq_length = &c;
    } else if (c.name == "test_cc_simple_rnn_batchwise") {
      batchwise = &c;
    }
  }
  ASSERT_NE(defaults, nullptr);
  ASSERT_NE(with_initial_bias, nullptr);
  ASSERT_NE(seq_length, nullptr);
  ASSERT_NE(batchwise, nullptr);

  // ``simple_rnn_defaults``: RNN node with X/W/R inputs and Y_h output only
  // (Y is skipped via an empty output name). Y_h has shape [1, 3, 4].
  {
    const GraphProto &graph = defaults->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "RNN");
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(defaults->data_sets.size(), 1u);
    const auto &ds = defaults->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 4}));
  }

  // ``simple_rnn_with_initial_bias``: full Y output too, shape [2, 1, 3, 4].
  {
    const GraphProto &graph = with_initial_bias->model.ref_graph();
    ASSERT_EQ(graph.ref_output().size(), 2u);
    const auto &ds = with_initial_bias->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 1, 3, 4}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{1, 3, 4}));
  }

  // ``rnn_seq_length``: X/W/R/B inputs, Y_h-only output with shape
  // [1, 3, 5] (num_directions=1, batch_size=3, hidden_size=5).
  {
    const GraphProto &graph = seq_length->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 4u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const auto &ds = seq_length->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 4u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 5}));
  }

  // ``simple_rnn_batchwise``: layout=1 with batch_size=3, seq_length=1.
  // Y has shape [batch, seq, num_directions, hidden] = [3, 1, 1, 4] and
  // Y_h has shape [batch, num_directions, hidden] = [3, 1, 4].
  {
    const GraphProto &graph = batchwise->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 2u);
    const auto &ds = batchwise->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 1, 2}));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 1, 4}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{3, 1, 4}));
  }
}

TEST(BackendTestCase, AttentionCasesArePresent) {
  auto cases = CollectTestCases("Attention");
  const TestCase *basic = nullptr;
  const TestCase *gqa = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_attention_4d") {
      basic = &c;
    } else if (c.name == "test_cc_attention_4d_gqa") {
      gqa = &c;
    }
  }
  ASSERT_NE(basic, nullptr);
  ASSERT_NE(gqa, nullptr);

  for (const TestCase *tc : {basic, gqa}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Attention");
    ASSERT_EQ(node.ref_input().size(), 3u);
    ASSERT_EQ(node.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape.size(), 4u);
  }

  // Basic case output shape (B, Hq, Lq, Dv).
  EXPECT_EQ(basic->data_sets[0].outputs[0].shape, (std::vector<int64_t>{1, 2, 2, 2}));
  // GQA case output shape (B, Hq, Lq, Dv).
  EXPECT_EQ(gqa->data_sets[0].outputs[0].shape, (std::vector<int64_t>{1, 4, 2, 2}));
}

TEST(BackendTestCase, DropoutCasesArePresent) {
  auto cases = CollectTestCases("Dropout");
  const TestCase *inference = nullptr;
  const TestCase *training = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_dropout_default_inference") {
      inference = &c;
    } else if (c.name == "test_cc_dropout_training_mask") {
      training = &c;
    }
  }
  ASSERT_NE(inference, nullptr);
  ASSERT_NE(training, nullptr);

  {
    const GraphProto &graph = inference->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Dropout");
    ASSERT_EQ(node.ref_input().size(), 1u);
    ASSERT_EQ(node.ref_output().size(), 1u);
    const auto &ds = inference->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 3}));
  }

  {
    const GraphProto &graph = training->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Dropout");
    ASSERT_EQ(node.ref_input().size(), 3u);
    ASSERT_EQ(node.ref_output().size(), 2u);
    const auto &ds = training->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.outputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{2, 3}));
  }
}

TEST(BackendTestCase, FlattenCasesArePresent) {
  auto cases = CollectTestCases("Flatten");
  const TestCase *def = nullptr;
  const TestCase *axis0 = nullptr;
  const TestCase *axis2 = nullptr;
  const TestCase *neg1 = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_flatten_default_axis") {
      def = &c;
    } else if (c.name == "test_cc_flatten_axis0") {
      axis0 = &c;
    } else if (c.name == "test_cc_flatten_axis2") {
      axis2 = &c;
    } else if (c.name == "test_cc_flatten_negative_axis1") {
      neg1 = &c;
    }
  }
  ASSERT_NE(def, nullptr);
  ASSERT_NE(axis0, nullptr);
  ASSERT_NE(axis2, nullptr);
  ASSERT_NE(neg1, nullptr);

  {
    const auto &ds = def->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{5, 24}));
  }
  {
    const auto &ds = axis0->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 120}));
  }
  {
    const auto &ds = axis2->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{6, 20}));
  }
  {
    const auto &ds = neg1->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{24, 5}));
  }
}

TEST(BackendTestCase, LSTMCasesArePresent) {
  auto cases = CollectTestCases("LSTM");
  const TestCase *defaults = nullptr;
  const TestCase *with_initial_bias = nullptr;
  const TestCase *with_peepholes = nullptr;
  const TestCase *batchwise = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_lstm_defaults") {
      defaults = &c;
    } else if (c.name == "test_cc_lstm_with_initial_bias") {
      with_initial_bias = &c;
    } else if (c.name == "test_cc_lstm_with_peepholes") {
      with_peepholes = &c;
    } else if (c.name == "test_cc_lstm_batchwise") {
      batchwise = &c;
    }
  }
  ASSERT_NE(defaults, nullptr);
  ASSERT_NE(with_initial_bias, nullptr);
  ASSERT_NE(with_peepholes, nullptr);
  ASSERT_NE(batchwise, nullptr);

  // ``lstm_defaults``: LSTM node with X/W/R inputs and Y_h output only.
  // Y_h has shape [1, 3, 3] (num_directions=1, batch=3, hidden=3).
  {
    const GraphProto &graph = defaults->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "LSTM");
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(defaults->data_sets.size(), 1u);
    const auto &ds = defaults->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 3}));
  }

  // ``lstm_with_initial_bias``: X/W/R/B inputs, Y_h-only output of shape
  // [1, 3, 4] (num_directions=1, batch=3, hidden=4).
  {
    const GraphProto &graph = with_initial_bias->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 4u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const auto &ds = with_initial_bias->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 4u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 4}));
  }

  // ``lstm_with_peepholes``: 8 inputs (X, W, R, B, sequence_lens,
  // initial_h, initial_c, P), Y_h-only output of shape [1, 2, 3].
  {
    const GraphProto &graph = with_peepholes->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 8u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const auto &ds = with_peepholes->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 8u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    // ``sequence_lens`` is INT32 per the LSTM schema.
    EXPECT_EQ(ds.inputs[4].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 2, 3}));
  }

  // ``lstm_batchwise``: layout=1 with batch_size=3, seq_length=1,
  // hidden_size=7. Y has shape [batch, seq, num_directions, hidden] =
  // [3, 1, 1, 7] and Y_h has shape [batch, num_directions, hidden] = [3, 1, 7].
  {
    const GraphProto &graph = batchwise->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 2u);
    const auto &ds = batchwise->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 1, 2}));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 1, 7}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{3, 1, 7}));
  }
}

TEST(BackendTestCase, GRUCasesArePresent) {
  auto cases = CollectTestCases("GRU");
  const TestCase *defaults = nullptr;
  const TestCase *with_initial_bias = nullptr;
  const TestCase *seq_length = nullptr;
  const TestCase *batchwise = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_gru_defaults") {
      defaults = &c;
    } else if (c.name == "test_cc_gru_with_initial_bias") {
      with_initial_bias = &c;
    } else if (c.name == "test_cc_gru_seq_length") {
      seq_length = &c;
    } else if (c.name == "test_cc_gru_batchwise") {
      batchwise = &c;
    }
  }
  ASSERT_NE(defaults, nullptr);
  ASSERT_NE(with_initial_bias, nullptr);
  ASSERT_NE(seq_length, nullptr);
  ASSERT_NE(batchwise, nullptr);

  // ``gru_defaults``: GRU node with X/W/R inputs and Y_h output only
  // (Y is skipped via an empty output name). Y_h has shape [1, 3, 5]
  // (num_directions=1, batch=3, hidden=5).
  {
    const GraphProto &graph = defaults->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "GRU");
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(defaults->data_sets.size(), 1u);
    const auto &ds = defaults->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 5}));
  }

  // ``gru_with_initial_bias``: X/W/R/B inputs, Y_h-only output of shape
  // [1, 3, 3] (num_directions=1, batch=3, hidden=3).
  {
    const GraphProto &graph = with_initial_bias->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 4u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const auto &ds = with_initial_bias->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 4u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 3}));
  }

  // ``gru_seq_length``: X/W/R/B inputs, Y_h-only output with shape
  // [1, 3, 5] (num_directions=1, batch_size=3, hidden_size=5).
  {
    const GraphProto &graph = seq_length->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 4u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const auto &ds = seq_length->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 4u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1, 3, 5}));
  }

  // ``gru_batchwise``: layout=1 with batch_size=3, seq_length=1,
  // hidden_size=6. Y has shape [batch, seq, num_directions, hidden] =
  // [3, 1, 1, 6] and Y_h has shape [batch, num_directions, hidden] = [3, 1, 6].
  {
    const GraphProto &graph = batchwise->model.ref_graph();
    ASSERT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 2u);
    const auto &ds = batchwise->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3, 1, 2}));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 1, 6}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{3, 1, 6}));
  }
}

} // namespace Test
