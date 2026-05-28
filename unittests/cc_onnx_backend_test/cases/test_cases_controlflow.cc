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
std::vector<onnx_backend_test::TestCase> CollectTestCases() {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectControlflowTestCases(registry);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, IfCasesArePresent) {
  auto cases = CollectTestCases();
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
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
