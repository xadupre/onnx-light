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
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectGeneratorTestCases(registry, op_type);
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
  auto cases = CollectTestCases("Constant");
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
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
  auto cases = CollectTestCases("Constant");
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
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{5, 5}));
  EXPECT_EQ(ds.outputs[0].element_count(), 25);
}

TEST(BackendTestCase, ConstantAttributeVariantCasesArePresent) {
  // Each Constant attribute variant case registers a single-node graph with
  // no inputs and an output tensor whose dtype/shape mirror the attribute.
  auto cases = CollectTestCases("Constant");

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
       static_cast<int32_t>(onnx_kernels::DataType::FLOAT),
       {}},
      {"test_cc_constant_value_floats",
       "value_floats",
       AttributeProto::AttributeType::FLOATS,
       static_cast<int32_t>(onnx_kernels::DataType::FLOAT),
       {4}},
      {"test_cc_constant_value_int",
       "value_int",
       AttributeProto::AttributeType::INT,
       static_cast<int32_t>(onnx_kernels::DataType::INT64),
       {}},
      {"test_cc_constant_value_ints",
       "value_ints",
       AttributeProto::AttributeType::INTS,
       static_cast<int32_t>(onnx_kernels::DataType::INT64),
       {5}},
      {"test_cc_constant_value_string",
       "value_string",
       AttributeProto::AttributeType::STRING,
       static_cast<int32_t>(onnx_kernels::DataType::STRING),
       {}},
      {"test_cc_constant_value_strings",
       "value_strings",
       AttributeProto::AttributeType::STRINGS,
       static_cast<int32_t>(onnx_kernels::DataType::STRING),
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
  auto cases = CollectTestCases("ConstantOfShape");

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
    EXPECT_EQ(ds.inputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
    EXPECT_EQ(ds.inputs[0].shape, (std::vector<int64_t>{3}));
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{10, 6}));
    EXPECT_EQ(ds.outputs[0].element_count(), 60);
  }

  // ``test_constantofshape_int_shape_zero`` — output [0] of INT32 1.
  {
    const TestCase *tc = FindCase(cases, "test_constantofshape_int_shape_zero");
    ASSERT_NE(tc, nullptr);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0}));
    EXPECT_EQ(ds.outputs[0].element_count(), 0);
  }

  // Library-local ``test_cc_constantofshape_int64_fortytwo``.
  {
    const TestCase *tc = FindCase(cases, "test_cc_constantofshape_int64_fortytwo");
    ASSERT_NE(tc, nullptr);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 3}));
    const int64_t *py = ds.outputs[0].AsInt64();
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(py[i], 42);
    }
  }
}

TEST(BackendTestCase, EyeLikeCasesArePresent) {
  auto cases = CollectTestCases("EyeLike");

  {
    const TestCase *tc = FindCase(cases, "test_eyelike_without_dtype");
    ASSERT_NE(tc, nullptr);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "EyeLike");
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{3, 4}));
  }

  {
    const TestCase *tc = FindCase(cases, "test_eyelike_with_dtype");
    ASSERT_NE(tc, nullptr);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    ASSERT_EQ(node.ref_attribute().size(), 2u);
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 4}));
  }

  {
    const TestCase *tc = FindCase(cases, "test_eyelike_populate_off_main_diagonal");
    ASSERT_NE(tc, nullptr);
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    ASSERT_EQ(node.ref_attribute().size(), 2u);
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{4, 5}));
  }
}

TEST(BackendTestCase, BernoulliCasesArePresent) {
  auto cases = CollectTestCases("Bernoulli");
  ASSERT_FALSE(cases.empty());

  // Default (no attributes), DOUBLE dtype override and explicit seed cases.
  const TestCase *plain = FindCase(cases, "test_cc_bernoulli");
  const TestCase *doubled = FindCase(cases, "test_cc_bernoulli_double");
  const TestCase *seeded = FindCase(cases, "test_cc_bernoulli_seed");
  ASSERT_NE(plain, nullptr);
  ASSERT_NE(doubled, nullptr);
  ASSERT_NE(seeded, nullptr);

  for (const TestCase *tc : {plain, doubled, seeded}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Bernoulli");
    ASSERT_EQ(graph.ref_input().size(), 1u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, ds.inputs[0].shape);
  }

  // Plain case: no attributes, output dtype matches input (FLOAT).
  {
    const NodeProto &node = plain->model.ref_graph().ref_node()[0];
    EXPECT_EQ(node.ref_attribute().size(), 0u);
    EXPECT_EQ(plain->data_sets[0].outputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  }

  // DOUBLE case: ``dtype`` attribute promotes the output to DOUBLE.
  {
    const NodeProto &node = doubled->model.ref_graph().ref_node()[0];
    ASSERT_EQ(node.ref_attribute().size(), 1u);
    const auto &attr = node.ref_attribute()[0];
    const auto &attr_name = attr.ref_name();
    EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "dtype");
    EXPECT_EQ(attr.type(), AttributeProto::AttributeType::INT);
    EXPECT_EQ(attr.i(), static_cast<int64_t>(onnx_kernels::DataType::DOUBLE));
    EXPECT_EQ(doubled->data_sets[0].outputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::DOUBLE));
  }

  // Seeded case: ``seed`` attribute is present.
  {
    const NodeProto &node = seeded->model.ref_graph().ref_node()[0];
    ASSERT_EQ(node.ref_attribute().size(), 1u);
    const auto &attr = node.ref_attribute()[0];
    const auto &attr_name = attr.ref_name();
    EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "seed");
    EXPECT_EQ(attr.type(), AttributeProto::AttributeType::FLOAT);
  }

  // Outputs are 0/1 only.
  const float *py = plain->data_sets[0].outputs[0].AsFloat();
  for (int64_t i = 0; i < plain->data_sets[0].outputs[0].element_count(); ++i) {
    EXPECT_TRUE(py[i] == 0.0f || py[i] == 1.0f);
  }
}

namespace {

void CheckRandomCasePresent(const std::vector<TestCase> &cases, const std::string &name,
                            const std::string &expected_op_type, int expected_inputs) {
  const TestCase *tc = FindCase(cases, name);
  ASSERT_NE(tc, nullptr) << "missing case: " << name;
  const GraphProto &graph = tc->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), expected_op_type);
  ASSERT_EQ(graph.ref_input().size(), static_cast<size_t>(expected_inputs));
  ASSERT_EQ(graph.ref_output().size(), 1u);
  ASSERT_EQ(tc->data_sets.size(), 1u);
  ASSERT_EQ(tc->data_sets[0].outputs.size(), 1u);
}

} // namespace

TEST(BackendTestCase, RandomNormalCasesArePresent) {
  auto cases = CollectTestCases("RandomNormal");
  ASSERT_FALSE(cases.empty());
  CheckRandomCasePresent(cases, "test_cc_randomnormal", "RandomNormal", 0);
  CheckRandomCasePresent(cases, "test_cc_randomnormal_seeded", "RandomNormal", 0);
  CheckRandomCasePresent(cases, "test_cc_randomnormal_double", "RandomNormal", 0);
}

TEST(BackendTestCase, RandomUniformCasesArePresent) {
  auto cases = CollectTestCases("RandomUniform");
  ASSERT_FALSE(cases.empty());
  CheckRandomCasePresent(cases, "test_cc_randomuniform", "RandomUniform", 0);
  CheckRandomCasePresent(cases, "test_cc_randomuniform_seeded", "RandomUniform", 0);
  CheckRandomCasePresent(cases, "test_cc_randomuniform_double", "RandomUniform", 0);
}

TEST(BackendTestCase, RandomNormalLikeCasesArePresent) {
  auto cases = CollectTestCases("RandomNormalLike");
  ASSERT_FALSE(cases.empty());
  CheckRandomCasePresent(cases, "test_cc_randomnormallike", "RandomNormalLike", 1);
  CheckRandomCasePresent(cases, "test_cc_randomnormallike_double", "RandomNormalLike", 1);
  CheckRandomCasePresent(cases, "test_cc_randomnormallike_seeded", "RandomNormalLike", 1);
}

TEST(BackendTestCase, RandomUniformLikeCasesArePresent) {
  auto cases = CollectTestCases("RandomUniformLike");
  ASSERT_FALSE(cases.empty());
  CheckRandomCasePresent(cases, "test_cc_randomuniformlike", "RandomUniformLike", 1);
  CheckRandomCasePresent(cases, "test_cc_randomuniformlike_double", "RandomUniformLike", 1);
  CheckRandomCasePresent(cases, "test_cc_randomuniformlike_seeded", "RandomUniformLike", 1);
}

} // namespace Test
