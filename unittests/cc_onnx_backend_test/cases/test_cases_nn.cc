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
std::vector<onnx_backend_test::TestCase> CollectTestCases() {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectNNTestCases(registry);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, AveragePoolCasesArePresent) {
  auto cases = CollectTestCases();
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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

} // namespace Test
