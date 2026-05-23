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
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Expect;
using onnx_backend_test::OpsetId;
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

  OpsetId osid("", 14);

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

TEST(BackendTestCase, DefaultOpsetUsesEmptyDomain) {
  OpsetId osid = DefaultOpset(17);
  EXPECT_EQ(osid.domain, "");
  EXPECT_EQ(osid.version, 17);
}

TEST(BackendTestCase, OpsetIdConstructsWithCustomDomain) {
  OpsetId osid("ai.onnx.ml", 3);
  EXPECT_EQ(osid.domain, "ai.onnx.ml");
  EXPECT_EQ(osid.version, 3);
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

TEST(BackendTestCase, BlackmanWindowCasesArePresent) {
  auto cases = CollectTestCases();
  const TestCase *periodic = nullptr;
  const TestCase *symmetric = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_blackmanwindow")
      periodic = &c;
    if (c.name == "test_cc_blackmanwindow_symmetric")
      symmetric = &c;
  }
  ASSERT_NE(periodic, nullptr);
  ASSERT_NE(symmetric, nullptr);

  // Both variants take a scalar int32 size and produce a 1-D float window.
  for (const TestCase *tc : {periodic, symmetric}) {
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
    EXPECT_EQ(ds.inputs[0].shape.size(), 0u);
    ASSERT_EQ(ds.outputs[0].shape.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape[0], 10);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    ASSERT_EQ(tc->model.ref_opset_import().size(), 1u);
    EXPECT_EQ(tc->model.ref_opset_import()[0].version(), 17);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const auto &op_type = graph.ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "BlackmanWindow");
  }

  // The symmetric variant carries a ``periodic = 0`` attribute; the periodic
  // variant relies on the default and has no attribute.
  ASSERT_EQ(symmetric->model.ref_graph().ref_node()[0].ref_attribute().size(), 1u);
  const auto &attr = symmetric->model.ref_graph().ref_node()[0].ref_attribute()[0];
  EXPECT_EQ(std::string(attr.ref_name().data(), attr.ref_name().size()), "periodic");
  EXPECT_EQ(attr.type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.i(), 0);
  EXPECT_EQ(periodic->model.ref_graph().ref_node()[0].ref_attribute().size(), 0u);

  // Output values match the Blackman window reference formula.
  constexpr double kPi = 3.14159265358979323846;
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  const int32_t size = 10;
  for (const TestCase *tc : {periodic, symmetric}) {
    const double divisor =
        (tc == periodic) ? static_cast<double>(size) : static_cast<double>(size - 1);
    const float *y = tc->data_sets[0].outputs[0].AsFloat();
    for (int32_t n = 0; n < size; ++n) {
      const double k = static_cast<double>(n) / divisor;
      const float expected =
          static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
      EXPECT_FLOAT_EQ(y[n], expected);
    }
  }
}

} // namespace Test
