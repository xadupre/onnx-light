// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_lib/shape_inference/implementation.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTrainingTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectTrainingTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_kernels::Tensor;
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

TEST(BackendTestCase, AdamCasesArePresent) {
  const auto cases = CollectTestCases("Adam");
  const TestCase *single = FindCase(cases, "test_cc_adam_single");
  const TestCase *multiple = FindCase(cases, "test_cc_adam_multiple");
  ASSERT_NE(single, nullptr);
  ASSERT_NE(multiple, nullptr);

  for (const TestCase *tc : {single, multiple}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Adam");
    const auto &domain = node.ref_domain();
    EXPECT_EQ(std::string(domain.data(), domain.size()), "ai.onnx.preview.training");

    // Adam-1 attributes: alpha, beta, epsilon, norm_coefficient,
    // norm_coefficient_post, all encoded as FLOAT scalars.
    ASSERT_EQ(node.ref_attribute().size(), 5u);
    for (const auto &attr : node.ref_attribute()) {
      EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
    }

    // Model must import the ``ai.onnx.preview.training`` opset at version 1.
    bool has_training_opset = false;
    for (const auto &osid : tc->model.ref_opset_import()) {
      const auto &d = osid.ref_domain();
      if (std::string(d.data(), d.size()) == "ai.onnx.preview.training") {
        EXPECT_EQ(osid.version(), 1);
        has_training_opset = true;
      }
    }
    EXPECT_TRUE(has_training_opset);

    ASSERT_EQ(tc->data_sets.size(), 1u);
  }

  // Single optimized tensor: inputs = {R, T, X, G, V, H} and outputs =
  // {X_new, V_new, H_new}.
  {
    const auto &ds = single->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 6u);
    ASSERT_EQ(ds.outputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i = 2; i < 6; ++i) {
      EXPECT_EQ(ds.inputs[i].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{3}));
    }
    for (const onnx_kernels::Tensor &t : ds.outputs) {
      EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(t.shape, (std::vector<int64_t>{3}));
    }
  }

  // Two optimized tensors: inputs = {R, T, X1, X2, G1, G2, V1, V2, H1, H2}
  // and outputs = {X1_new, X2_new, V1_new, V2_new, H1_new, H2_new}.
  {
    const auto &ds = multiple->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 10u);
    ASSERT_EQ(ds.outputs.size(), 6u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    // X1, G1, V1, H1 are rank-1; X2, G2, V2, H2 are rank-2.
    for (size_t i : {2u, 4u, 6u, 8u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    for (size_t i : {3u, 5u, 7u, 9u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2, 2}));
    }
    // Outputs mirror per-variable shapes in groups (X_new, V_new, H_new).
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(ds.outputs[2].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[3].shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(ds.outputs[4].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[5].shape, (std::vector<int64_t>{2, 2}));
  }
}

TEST(BackendTestCase, AdamOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases exported by ``RegisterAdamCases``: same
  // names as those produced by ``onnx.backend.test.case.node.adam.Adam``.
  const auto cases = CollectTestCases("Adam");
  const TestCase *single = FindCase(cases, "test_adam");
  const TestCase *multiple = FindCase(cases, "test_adam_multiple");
  ASSERT_NE(single, nullptr);
  ASSERT_NE(multiple, nullptr);

  for (const TestCase *tc : {single, multiple}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Adam");
    const auto &domain = node.ref_domain();
    EXPECT_EQ(std::string(domain.data(), domain.size()), "ai.onnx.preview.training");

    // Upstream Adam Python cases set exactly four FLOAT attributes
    // (norm_coefficient, alpha, beta, epsilon) and leave
    // ``norm_coefficient_post`` at its schema default.
    ASSERT_EQ(node.ref_attribute().size(), 4u);
    for (const auto &attr : node.ref_attribute()) {
      EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
    }

    ASSERT_EQ(tc->data_sets.size(), 1u);
  }

  // ``test_adam``: single optimized rank-1 tensor of length 2.
  {
    const auto &ds = single->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 6u);
    ASSERT_EQ(ds.outputs.size(), 3u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i = 2; i < 6; ++i) {
      EXPECT_EQ(ds.inputs[i].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    for (const onnx_kernels::Tensor &t : ds.outputs) {
      EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(t.shape, (std::vector<int64_t>{2}));
    }
  }

  // ``test_adam_multiple``: two optimized rank-1 tensors of lengths 1 and 2.
  {
    const auto &ds = multiple->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 10u);
    ASSERT_EQ(ds.outputs.size(), 6u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i : {2u, 4u, 6u, 8u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{1}));
    }
    for (size_t i : {3u, 5u, 7u, 9u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[2].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[3].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[4].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[5].shape, (std::vector<int64_t>{2}));
  }
}

TEST(BackendTestCase, AdamCasesPassShapeInference) {
  auto cases = CollectTestCases("Adam");
  ASSERT_FALSE(cases.empty());

  for (auto &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model));
  }
}

TEST(BackendTestCase, AdagradOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases registered by ``RegisterAdagradCases``:
  // same names as those produced by
  // ``onnx.backend.test.case.node.adagrad.Adagrad``.
  const auto cases = CollectTestCases("Adagrad");
  const TestCase *single = FindCase(cases, "test_adagrad");
  const TestCase *multiple = FindCase(cases, "test_adagrad_multiple");
  ASSERT_NE(single, nullptr);
  ASSERT_NE(multiple, nullptr);

  for (const TestCase *tc : {single, multiple}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Adagrad");
    const auto &domain = node.ref_domain();
    EXPECT_EQ(std::string(domain.data(), domain.size()), "ai.onnx.preview.training");

    // Adagrad-1 attributes set by the upstream Python cases: exactly three
    // FLOAT attributes (norm_coefficient, epsilon, decay_factor).
    ASSERT_EQ(node.ref_attribute().size(), 3u);
    for (const auto &attr : node.ref_attribute()) {
      EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
    }

    bool has_training_opset = false;
    for (const auto &osid : tc->model.ref_opset_import()) {
      const auto &d = osid.ref_domain();
      if (std::string(d.data(), d.size()) == "ai.onnx.preview.training") {
        EXPECT_EQ(osid.version(), 1);
        has_training_opset = true;
      }
    }
    EXPECT_TRUE(has_training_opset);

    ASSERT_EQ(tc->data_sets.size(), 1u);
  }

  // ``test_adagrad``: single optimized rank-1 tensor of length 1, with
  // inputs ``{R, T, X, G, H}`` and outputs ``{X_new, H_new}``.
  {
    const auto &ds = single->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 5u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i = 2; i < 5; ++i) {
      EXPECT_EQ(ds.inputs[i].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{1}));
    }
    for (const onnx_kernels::Tensor &t : ds.outputs) {
      EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(t.shape, (std::vector<int64_t>{1}));
    }
  }

  // ``test_adagrad_multiple``: two optimized rank-1 tensors of lengths
  // 1 and 2, with inputs ``{R, T, X1, X2, G1, G2, H1, H2}`` and outputs
  // ``{X1_new, X2_new, H1_new, H2_new}``.
  {
    const auto &ds = multiple->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 8u);
    ASSERT_EQ(ds.outputs.size(), 4u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i : {2u, 4u, 6u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{1}));
    }
    for (size_t i : {3u, 5u, 7u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[2].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[3].shape, (std::vector<int64_t>{2}));
  }
}

TEST(BackendTestCase, AdagradCasesPassShapeInference) {
  auto cases = CollectTestCases("Adagrad");
  ASSERT_FALSE(cases.empty());
  for (auto &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model));
  }
}

TEST(BackendTestCase, MomentumOnnxCasesArePresent) {
  // Upstream-ONNX-mirrored cases registered by ``RegisterMomentumCases``:
  // same names as those produced by
  // ``onnx.backend.test.case.node.momentum.Momentum``.
  const auto cases = CollectTestCases("Momentum");
  const TestCase *standard = FindCase(cases, "test_momentum");
  const TestCase *nesterov = FindCase(cases, "test_nesterov_momentum");
  const TestCase *multiple = FindCase(cases, "test_momentum_multiple");
  ASSERT_NE(standard, nullptr);
  ASSERT_NE(nesterov, nullptr);
  ASSERT_NE(multiple, nullptr);

  for (const TestCase *tc : {standard, nesterov, multiple}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Momentum");
    const auto &domain = node.ref_domain();
    EXPECT_EQ(std::string(domain.data(), domain.size()), "ai.onnx.preview.training");

    // Momentum-1 attributes set by the upstream Python cases: three FLOAT
    // attributes (norm_coefficient, alpha, beta) and one STRING attribute
    // (mode).
    ASSERT_EQ(node.ref_attribute().size(), 4u);
    int float_count = 0;
    int string_count = 0;
    for (const auto &attr : node.ref_attribute()) {
      if (attr.ref_type() == AttributeProto::AttributeType::FLOAT) {
        ++float_count;
      } else if (attr.ref_type() == AttributeProto::AttributeType::STRING) {
        ++string_count;
      }
    }
    EXPECT_EQ(float_count, 3);
    EXPECT_EQ(string_count, 1);

    ASSERT_EQ(tc->data_sets.size(), 1u);
  }

  // ``test_momentum`` and ``test_nesterov_momentum`` share the same shape
  // layout: single optimized rank-1 tensor of length 2 with inputs
  // ``{R, T, X, G, V}`` and outputs ``{X_new, V_new}``.
  for (const TestCase *tc : {standard, nesterov}) {
    SCOPED_TRACE(tc->name);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 5u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.inputs[0].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::FLOAT)); // R
    EXPECT_EQ(ds.inputs[1].data_type,
              static_cast<int32_t>(onnx_kernels::DataType::INT64)); // T
    for (size_t i = 2; i < 5; ++i) {
      EXPECT_EQ(ds.inputs[i].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    for (const onnx_kernels::Tensor &t : ds.outputs) {
      EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(t.shape, (std::vector<int64_t>{2}));
    }
  }

  // ``test_momentum_multiple``: two optimized rank-1 tensors of lengths
  // 1 and 2.
  {
    const auto &ds = multiple->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 8u);
    ASSERT_EQ(ds.outputs.size(), 4u);
    for (size_t i : {2u, 4u, 6u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{1}));
    }
    for (size_t i : {3u, 5u, 7u}) {
      EXPECT_EQ(ds.inputs[i].shape, (std::vector<int64_t>{2}));
    }
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ds.outputs[2].shape, (std::vector<int64_t>{1}));
    EXPECT_EQ(ds.outputs[3].shape, (std::vector<int64_t>{2}));
  }
}

TEST(BackendTestCase, MomentumCasesPassShapeInference) {
  auto cases = CollectTestCases("Momentum");
  ASSERT_FALSE(cases.empty());
  for (auto &tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_NO_THROW(shape_inference::InferShapes(tc.model));
  }
}

} // namespace Test
