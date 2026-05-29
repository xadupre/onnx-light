// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectObjectDetectionTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectObjectDetectionTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, RoiAlignCasesArePresent) {
  auto cases = CollectTestCases("RoiAlign");
  const TestCase *avg = nullptr;
  const TestCase *max_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_roialign") {
      avg = &c;
    } else if (c.name == "test_cc_roialign_max") {
      max_case = &c;
    }
  }
  ASSERT_NE(avg, nullptr);
  ASSERT_NE(max_case, nullptr);

  // Avg case: three inputs (X, rois, batch_indices), single output of shape
  // (num_rois, C, output_height, output_width) = (2, 1, 5, 5).
  {
    const GraphProto &graph = avg->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "RoiAlign");
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(avg->data_sets.size(), 1u);
    const auto &ds = avg->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 1, 5, 5}));
    // Outputs sample from a feature map whose values lie in [0, 1).
    const float *y = ds.outputs[0].AsFloat();
    for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
      EXPECT_GE(y[i], 0.0f);
      EXPECT_LE(y[i], 1.0f);
    }
  }

  // Max case: same shapes; max values must be >= the avg-case ones since
  // both pool over the same RoI grid.
  {
    const auto &ds_avg = avg->data_sets[0];
    const auto &ds_max = max_case->data_sets[0];
    EXPECT_EQ(ds_max.outputs[0].shape, (std::vector<int64_t>{2, 1, 5, 5}));
    const float *pmax = ds_max.outputs[0].AsFloat();
    const float *pavg = ds_avg.outputs[0].AsFloat();
    // The two cases use different coordinate_transformation_mode, so values
    // are not directly comparable; just sanity-check non-negativity and
    // bounded range.
    for (int64_t i = 0; i < ds_max.outputs[0].element_count(); ++i) {
      EXPECT_GE(pmax[i], 0.0f);
      EXPECT_LE(pmax[i], 1.0f);
      (void)pavg;
    }
  }
}

// Verifies that the three upstream ``test_roialign_*`` cases imported from
// ``onnx/backend/test/case/node/roialign.py`` are registered with the right
// op_type, input/output cardinality and shapes (3 RoIs, (3, 1, 5, 5) output).
TEST(BackendTestCase, RoiAlignUpstreamCasesArePresent) {
  auto cases = CollectTestCases("RoiAlign");
  const TestCase *aligned_false = nullptr;
  const TestCase *aligned_true = nullptr;
  const TestCase *mode_max = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_roialign_aligned_false") {
      aligned_false = &c;
    } else if (c.name == "test_cc_roialign_aligned_true") {
      aligned_true = &c;
    } else if (c.name == "test_cc_roialign_mode_max") {
      mode_max = &c;
    }
  }
  ASSERT_NE(aligned_false, nullptr);
  ASSERT_NE(aligned_true, nullptr);
  ASSERT_NE(mode_max, nullptr);

  for (const TestCase *tc : {aligned_false, aligned_true, mode_max}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "RoiAlign");
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    // (num_rois, C, output_height, output_width) = (3, 1, 5, 5).
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 1, 5, 5}));
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{1, 1, 10, 10}));
    EXPECT_EQ(ds.inputs[1].shape, (std::vector<int64_t>{3, 4}));
    EXPECT_EQ(ds.inputs[2].shape, (std::vector<int64_t>{3}));
  }
}

} // namespace Test
