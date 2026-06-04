// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectControlflowTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectControlflowTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, IfCasesArePresent) {
  auto cases = CollectTestCases("If");
  const TestCase *if_true = nullptr;
  const TestCase *if_false = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_if")
      if_true = &c;
    if (c.name == "test_cc_if_else")
      if_false = &c;
  }
  ASSERT_NE(if_true, nullptr);
  ASSERT_NE(if_false, nullptr);

  // Both cases share the same single-node ``If`` topology: scalar BOOL ``cond``
  // input and one tensor output.
  for (const TestCase *tc : {if_true, if_false}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "If");
    ASSERT_EQ(node.ref_attribute().size(), 2u);

    // Both attributes are GRAPH subgraphs named then_branch / else_branch.
    bool has_then = false, has_else = false;
    for (const auto &attr : node.ref_attribute()) {
      const std::string name(attr.ref_name().data(), attr.ref_name().size());
      EXPECT_EQ(attr.type(), AttributeProto::AttributeType::GRAPH);
      ASSERT_TRUE(attr.has_g());
      ASSERT_EQ(attr.ref_g().ref_node().size(), 1u);
      const auto &subop = attr.ref_g().ref_node()[0].ref_op_type();
      EXPECT_EQ(std::string(subop.data(), subop.size()), "Constant");
      if (name == "then_branch")
        has_then = true;
      if (name == "else_branch")
        has_else = true;
    }
    EXPECT_TRUE(has_then);
    EXPECT_TRUE(has_else);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 2);
  }

  // True branch returns [1, 2]; false branch returns [3, 4].
  EXPECT_EQ(if_true->data_sets[0].inputs[0].data[0], 1);
  EXPECT_FLOAT_EQ(if_true->data_sets[0].outputs[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(if_true->data_sets[0].outputs[0].AsFloat()[1], 2.0f);
  EXPECT_EQ(if_false->data_sets[0].inputs[0].data[0], 0);
  EXPECT_FLOAT_EQ(if_false->data_sets[0].outputs[0].AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(if_false->data_sets[0].outputs[0].AsFloat()[1], 4.0f);
}

} // namespace Test
namespace Test {

TEST(BackendTestCase, LoopCasesArePresent) {
  auto cases = CollectTestCases("Loop");
  const TestCase *trip3 = nullptr;
  const TestCase *trip0 = nullptr;
  const TestCase *loop11 = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_loop_basic_trip_count")
      trip3 = &c;
    if (c.name == "test_cc_loop_zero_trip_count")
      trip0 = &c;
    if (c.name == "test_cc_loop11_carried_state")
      loop11 = &c;
  }
  ASSERT_NE(trip3, nullptr);
  ASSERT_NE(trip0, nullptr);
  ASSERT_NE(loop11, nullptr);

  for (const TestCase *tc : {trip3, trip0}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Loop");
    ASSERT_EQ(node.ref_attribute().size(), 1u);
    const auto &attr = node.ref_attribute()[0];
    const std::string attr_name(attr.ref_name().data(), attr.ref_name().size());
    EXPECT_EQ(attr_name, "body");
    EXPECT_EQ(attr.type(), AttributeProto::AttributeType::GRAPH);
    ASSERT_TRUE(attr.has_g());

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
    ASSERT_EQ(ds.outputs[0].shape.size(), 2u);
    EXPECT_EQ(ds.outputs[0].shape[1], 1);
  }

  // Trip-count 3 stacks the constant [42] three times → shape [3, 1].
  EXPECT_EQ(trip3->data_sets[0].outputs[0].shape[0], 3);
  ASSERT_EQ(trip3->data_sets[0].outputs[0].element_count(), 3);
  // Trip-count 0 produces an empty leading axis → shape [0, 1].
  EXPECT_EQ(trip0->data_sets[0].outputs[0].shape[0], 0);
  EXPECT_EQ(trip0->data_sets[0].outputs[0].element_count(), 0);

  // ``test_cc_loop11_carried_state`` mirrors ONNX's ``test_loop11``:
  // trip-count 5, FLOAT[1] loop-carried state ``y = [-2]`` and a FLOAT[1]
  // scan output. Inputs are ``(trip_count, cond, y)`` and outputs are
  // ``(res_y, res_scan)``.
  {
    const GraphProto &graph = loop11->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    EXPECT_EQ(std::string(node.ref_op_type().data(), node.ref_op_type().size()), "Loop");
    ASSERT_EQ(node.ref_input().size(), 3u);
    ASSERT_EQ(node.ref_output().size(), 2u);

    ASSERT_EQ(loop11->data_sets.size(), 1u);
    const auto &ds = loop11->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    // res_y is FLOAT[1] = [13].
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 1);
    ASSERT_EQ(ds.outputs[0].element_count(), 1);
    EXPECT_FLOAT_EQ(ds.outputs[0].AsFloat()[0], 13.0f);
    // res_scan is FLOAT[5, 1] = [[-1], [1], [4], [8], [13]].
    EXPECT_EQ(ds.outputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(ds.outputs[1].shape.size(), 2u);
    EXPECT_EQ(ds.outputs[1].shape[0], 5);
    EXPECT_EQ(ds.outputs[1].shape[1], 1);
    ASSERT_EQ(ds.outputs[1].element_count(), 5);
    EXPECT_FLOAT_EQ(ds.outputs[1].AsFloat()[0], -1.0f);
    EXPECT_FLOAT_EQ(ds.outputs[1].AsFloat()[1], 1.0f);
    EXPECT_FLOAT_EQ(ds.outputs[1].AsFloat()[2], 4.0f);
    EXPECT_FLOAT_EQ(ds.outputs[1].AsFloat()[3], 8.0f);
    EXPECT_FLOAT_EQ(ds.outputs[1].AsFloat()[4], 13.0f);
  }
}

TEST(BackendTestCase, ScanCasesArePresent) {
  auto cases = CollectTestCases("Scan");
  const TestCase *trip3 = nullptr;
  const TestCase *trip0 = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_scan_basic_trip_count")
      trip3 = &c;
    if (c.name == "test_cc_scan_zero_trip_count")
      trip0 = &c;
  }
  ASSERT_NE(trip3, nullptr);
  ASSERT_NE(trip0, nullptr);

  for (const TestCase *tc : {trip3, trip0}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Scan");
    // Two attributes: body (GRAPH) and num_scan_inputs (INT).
    ASSERT_EQ(node.ref_attribute().size(), 2u);
    bool has_body = false;
    bool has_num = false;
    for (const auto &attr : node.ref_attribute()) {
      const std::string name(attr.ref_name().data(), attr.ref_name().size());
      if (name == "body") {
        EXPECT_EQ(attr.type(), AttributeProto::AttributeType::GRAPH);
        ASSERT_TRUE(attr.has_g());
        has_body = true;
      } else if (name == "num_scan_inputs") {
        EXPECT_EQ(attr.type(), AttributeProto::AttributeType::INT);
        EXPECT_EQ(attr.i(), 1);
        has_num = true;
      }
    }
    EXPECT_TRUE(has_body);
    EXPECT_TRUE(has_num);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
    ASSERT_EQ(ds.outputs[0].shape.size(), 2u);
    EXPECT_EQ(ds.outputs[0].shape[1], 2);
  }

  // Trip-count 3 → output shape [3, 2] containing [[0, 1], [2, 3], [4, 5]].
  EXPECT_EQ(trip3->data_sets[0].outputs[0].shape[0], 3);
  ASSERT_EQ(trip3->data_sets[0].outputs[0].element_count(), 6);
  EXPECT_FLOAT_EQ(trip3->data_sets[0].outputs[0].AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(trip3->data_sets[0].outputs[0].AsFloat()[5], 5.0f);
  // Trip-count 0 → output shape [0, 2].
  EXPECT_EQ(trip0->data_sets[0].outputs[0].shape[0], 0);
  EXPECT_EQ(trip0->data_sets[0].outputs[0].element_count(), 0);
}

} // namespace Test
