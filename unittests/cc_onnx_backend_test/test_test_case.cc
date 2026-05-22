// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::Expect;
using onnx_backend_test::Tensor;
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, TensorFromFloatRoundTrip) {
  Tensor t = Tensor::FromFloat("a", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_EQ(t.name, "a");
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(t.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(t.element_count(), 6);
  EXPECT_EQ(t.element_size(), sizeof(float));
  const float *p = t.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(p[i], static_cast<float>(i + 1));
  }
}

TEST(BackendTestCase, TensorScalarHasSingleElement) {
  Tensor t = Tensor::FromFloat("s", {}, {3.5f});
  EXPECT_EQ(t.element_count(), 1);
  EXPECT_EQ(t.shape.size(), 0u);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 3.5f);
}

TEST(BackendTestCase, TensorRejectsShapeValueMismatch) {
  EXPECT_THROW(Tensor::FromFloat("x", {2, 3}, {1.0f}), std::invalid_argument);
}

TEST(BackendTestCase, TensorAsRejectsWrongDtype) {
  Tensor t = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  EXPECT_THROW((void)t.AsInt64(), std::invalid_argument);
}

TEST(BackendTestCase, ExpectBuildsSingleNodeModel) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  OperatorSetIdProto osid;
  osid.set_domain("");
  osid.set_version(14);

  std::vector<TestCase> registry;
  Expect(node,
         {Tensor::FromFloat("x", {2}, {1.0f, 2.0f}), Tensor::FromFloat("y", {2}, {3.0f, 4.0f})},
         {Tensor::FromFloat("z", {2}, {4.0f, 6.0f})}, "test_dummy_add", {osid}, "backend-test",
         registry);

  ASSERT_EQ(registry.size(), 1u);
  const TestCase &tc = registry[0];
  EXPECT_EQ(tc.name, "test_dummy_add");
  EXPECT_EQ(tc.kind, "node");
  EXPECT_EQ(tc.data_sets.size(), 1u);
  EXPECT_EQ(tc.data_sets[0].inputs.size(), 2u);
  EXPECT_EQ(tc.data_sets[0].outputs.size(), 1u);

  const GraphProto &graph = tc.model.ref_graph();
  EXPECT_EQ(graph.ref_node().size(), 1u);
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_output().size(), 1u);
  EXPECT_EQ(std::string(graph.ref_node()[0].ref_op_type().data(),
                        graph.ref_node()[0].ref_op_type().size()),
            "Add");
  ASSERT_EQ(tc.model.ref_opset_import().size(), 1u);
  EXPECT_EQ(tc.model.ref_opset_import()[0].version(), 14);
}

TEST(BackendTestCase, ExpectRejectsArityMismatch) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");

  std::vector<TestCase> registry;
  EXPECT_THROW(Expect(node, /*inputs=*/{}, /*outputs=*/{Tensor::FromFloat("y", {1}, {1.0f})}, "bad",
                      {}, "backend-test", registry),
               std::invalid_argument);
}

TEST(BackendTestCase, CollectReturnsExpectedNames) {
  auto cases = CollectTestCases();
  ASSERT_GE(cases.size(), 3u);
  bool has_abs = false, has_add = false, has_add_bcast = false;
  for (const auto &c : cases) {
    if (c.name == "test_cc_abs")
      has_abs = true;
    if (c.name == "test_cc_add")
      has_add = true;
    if (c.name == "test_cc_add_bcast")
      has_add_bcast = true;
  }
  EXPECT_TRUE(has_abs);
  EXPECT_TRUE(has_add);
  EXPECT_TRUE(has_add_bcast);
}

TEST(BackendTestCase, AddCaseOutputsAreElementwiseSum) {
  auto cases = CollectTestCases();
  const TestCase *add = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_add") {
      add = &c;
      break;
    }
  }
  ASSERT_NE(add, nullptr);
  ASSERT_EQ(add->data_sets.size(), 1u);
  const auto &ds = add->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 2u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  const float *x = ds.inputs[0].AsFloat();
  const float *y = ds.inputs[1].AsFloat();
  const float *z = ds.outputs[0].AsFloat();
  ASSERT_EQ(ds.inputs[0].element_count(), ds.outputs[0].element_count());
  for (int64_t i = 0; i < ds.outputs[0].element_count(); ++i) {
    EXPECT_FLOAT_EQ(z[i], x[i] + y[i]);
  }
}

} // namespace Test
