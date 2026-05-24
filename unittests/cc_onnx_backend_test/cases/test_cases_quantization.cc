// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, QuantizeLinearCaseIsPresent) {
  auto cases = CollectTestCases();
  const TestCase *uint8_case = nullptr;
  const TestCase *int8_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_quantizelinear") {
      uint8_case = &c;
    } else if (c.name == "test_cc_quantizelinear_int8") {
      int8_case = &c;
    }
  }
  ASSERT_NE(uint8_case, nullptr);
  ASSERT_NE(int8_case, nullptr);

  // Default UINT8 case: two inputs (x, y_scale), single UINT8 output.
  {
    const GraphProto &graph = uint8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "QuantizeLinear");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(uint8_case->data_sets.size(), 1u);
    const auto &ds = uint8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::UINT8));
    const std::vector<int64_t> expected_shape = {6};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    EXPECT_EQ(static_cast<int>(ds.outputs[0].data[0]), 0);
    EXPECT_EQ(static_cast<int>(ds.outputs[0].data[3]), 255);
  }

  // INT8 case: three inputs (x, y_scale, y_zero_point), single INT8 output.
  {
    const GraphProto &graph = int8_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    EXPECT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(int8_case->data_sets.size(), 1u);
    const auto &ds = int8_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT8));
    const int8_t *py = reinterpret_cast<const int8_t *>(ds.outputs[0].data.data());
    EXPECT_EQ(static_cast<int>(py[0]), -10);
    EXPECT_EQ(static_cast<int>(py[3]), 127);
  }
}

} // namespace Test
