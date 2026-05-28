// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

namespace {
const TestCase *FindCase(const std::vector<TestCase> &cases, const std::string &name) {
  for (const auto &c : cases) {
    if (c.name == name) {
      return &c;
    }
  }
  return nullptr;
}
} // namespace

TEST(BackendTestCase, ConstantCaseIsPresent) {
  auto cases = CollectTestCases();
  const TestCase *constant = FindCase(cases, "test_cc_constant");
  ASSERT_NE(constant, nullptr);

  // Single-node ``Constant`` topology: no graph inputs, one tensor output
  // declared from the ``value`` attribute.
  const GraphProto &graph = constant->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Constant");
  EXPECT_EQ(graph.ref_input().size(), 0u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  ASSERT_EQ(node.ref_attribute().size(), 1u);
  const auto &attr = node.ref_attribute()[0];
  const auto &attr_name = attr.ref_name();
  EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "value");
  EXPECT_EQ(attr.type(), AttributeProto::AttributeType::TENSOR);
  ASSERT_TRUE(attr.has_t());

  ASSERT_EQ(constant->data_sets.size(), 1u);
  const auto &ds = constant->data_sets[0];
  EXPECT_EQ(ds.inputs.size(), 0u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {2, 3};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  ASSERT_EQ(ds.outputs[0].element_count(), 6);
  const float *py = ds.outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(py[0], -1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 1.5f);
  EXPECT_FLOAT_EQ(py[3], -2.25f);
  EXPECT_FLOAT_EQ(py[4], 3.5f);
  EXPECT_FLOAT_EQ(py[5], -4.75f);
}

TEST(BackendTestCase, ConstantUpstreamOnnxCaseHasExpectedShape) {
  // Mirrors the upstream ``onnx.backend.test.case.node.constant.Constant``
  // export: a single ``Constant`` node with a rank-2 ``[5, 5]`` float
  // ``value`` attribute and no graph inputs.
  auto cases = CollectTestCases();
  const TestCase *tc = FindCase(cases, "test_constant");
  ASSERT_NE(tc, nullptr);

  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Constant");
  EXPECT_EQ(graph.ref_input().size(), 0u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  ASSERT_EQ(node.ref_attribute().size(), 1u);
  const auto &attr = node.ref_attribute()[0];
  const auto &attr_name = attr.ref_name();
  EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "value");
  EXPECT_EQ(attr.type(), AttributeProto::AttributeType::TENSOR);
  ASSERT_TRUE(attr.has_t());

  ASSERT_EQ(tc->data_sets.size(), 1u);
  const auto &ds = tc->data_sets[0];
  EXPECT_EQ(ds.inputs.size(), 0u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{5, 5}));
  EXPECT_EQ(ds.outputs[0].element_count(), 25);
}

} // namespace Test
