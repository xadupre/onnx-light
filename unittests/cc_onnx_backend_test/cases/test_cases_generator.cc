// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectGeneratorTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases() {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectGeneratorTestCases(registry);
  return registry;
}
} // namespace
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

TEST(BackendTestCase, ConstantAttributeVariantCasesArePresent) {
  // Each Constant attribute variant case registers a single-node graph with
  // no inputs and an output tensor whose dtype/shape mirror the attribute.
  auto cases = CollectTestCases();

  struct Variant {
    const char *name;
    const char *attr;
    AttributeProto::AttributeType attr_type;
    int32_t out_dtype;
    std::vector<int64_t> out_shape;
  };
  const std::vector<Variant> variants = {
      {"test_cc_constant_value_float",
       "value_float",
       AttributeProto::AttributeType::FLOAT,
       static_cast<int32_t>(TensorProto::DataType::FLOAT),
       {}},
      {"test_cc_constant_value_floats",
       "value_floats",
       AttributeProto::AttributeType::FLOATS,
       static_cast<int32_t>(TensorProto::DataType::FLOAT),
       {4}},
      {"test_cc_constant_value_int",
       "value_int",
       AttributeProto::AttributeType::INT,
       static_cast<int32_t>(TensorProto::DataType::INT64),
       {}},
      {"test_cc_constant_value_ints",
       "value_ints",
       AttributeProto::AttributeType::INTS,
       static_cast<int32_t>(TensorProto::DataType::INT64),
       {5}},
      {"test_cc_constant_value_string",
       "value_string",
       AttributeProto::AttributeType::STRING,
       static_cast<int32_t>(TensorProto::DataType::STRING),
       {}},
      {"test_cc_constant_value_strings",
       "value_strings",
       AttributeProto::AttributeType::STRINGS,
       static_cast<int32_t>(TensorProto::DataType::STRING),
       {3}},
  };

  for (const auto &v : variants) {
    SCOPED_TRACE(v.name);
    const TestCase *tc = FindCase(cases, v.name);
    ASSERT_NE(tc, nullptr);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Constant");
    EXPECT_EQ(graph.ref_input().size(), 0u);
    ASSERT_EQ(node.ref_attribute().size(), 1u);
    const auto &attr = node.ref_attribute()[0];
    const auto &attr_name = attr.ref_name();
    EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), v.attr);
    EXPECT_EQ(attr.type(), v.attr_type);
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    EXPECT_EQ(ds.inputs.size(), 0u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, v.out_dtype);
    EXPECT_EQ(ds.outputs[0].shape, v.out_shape);
  }
}

TEST(BackendTestCase, ConstantOfShapeCasesArePresent) {
  auto cases = CollectTestCases();

  // ``test_constantofshape_float_ones`` — output [4, 3, 2] of FLOAT 1.0.
  {
    const TestCase *tc = FindCase(cases, "test_constantofshape_float_ones");
    ASSERT_NE(tc, nullptr);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "ConstantOfShape");
    ASSERT_EQ(node.ref_attribute().size(), 1u);
    const auto &attr = node.ref_attribute()[0];
    const auto &attr_name = attr.ref_name();
    EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "value");
    EXPECT_EQ(attr.type(), AttributeProto::AttributeType::TENSOR);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3}));
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{4, 3, 2}));
    ASSERT_EQ(ds.outputs[0].element_count(), 24);
    const float *py = ds.outputs[0].AsFloat();
    for (int i = 0; i < 24; ++i) {
      EXPECT_FLOAT_EQ(py[i], 1.0f);
    }
  }

  // ``test_constantofshape_int_zeros`` — output [10, 6] of INT32 0.
  {
    const TestCase *tc = FindCase(cases, "test_constantofshape_int_zeros");
    ASSERT_NE(tc, nullptr);
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{10, 6}));
    EXPECT_EQ(ds.outputs[0].element_count(), 60);
  }

  // ``test_constantofshape_int_shape_zero`` — output [0] of INT32 1.
  {
    const TestCase *tc = FindCase(cases, "test_constantofshape_int_shape_zero");
    ASSERT_NE(tc, nullptr);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0}));
    EXPECT_EQ(ds.outputs[0].element_count(), 0);
  }

  // Library-local ``test_cc_constantofshape_int64_fortytwo``.
  {
    const TestCase *tc = FindCase(cases, "test_cc_constantofshape_int64_fortytwo");
    ASSERT_NE(tc, nullptr);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 3}));
    const int64_t *py = ds.outputs[0].AsInt64();
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(py[i], 42);
    }
  }
}

} // namespace Test
