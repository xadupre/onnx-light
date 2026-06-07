// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectPreviewTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectPreviewTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;
using onnx_kernels::Tensor;

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

TEST(BackendTestCase, FlexAttentionCasesArePresent) {
  const auto cases = CollectTestCases("FlexAttention");
  const TestCase *basic = FindCase(cases, "test_cc_flex_attention_basic");
  const TestCase *gqa = FindCase(cases, "test_cc_flex_attention_gqa");
  const TestCase *prob_mod_id = FindCase(cases, "test_cc_flex_attention_prob_mod_identity");
  const TestCase *prob_mod_scale = FindCase(cases, "test_cc_flex_attention_prob_mod_scale_half");
  ASSERT_NE(basic, nullptr);
  ASSERT_NE(gqa, nullptr);
  ASSERT_NE(prob_mod_id, nullptr);
  ASSERT_NE(prob_mod_scale, nullptr);

  for (const TestCase *tc : {basic, gqa, prob_mod_id, prob_mod_scale}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "FlexAttention");
    const auto &domain = node.ref_domain();
    EXPECT_EQ(std::string(domain.data(), domain.size()), "ai.onnx.preview");
    ASSERT_EQ(node.ref_input().size(), 3u);
    ASSERT_EQ(node.ref_output().size(), 1u);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    for (const Tensor &t : ds.inputs) {
      EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
      EXPECT_EQ(t.shape.size(), 4u);
    }
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
    EXPECT_EQ(ds.outputs[0].shape.size(), 4u);

    // Model must import the ``ai.onnx.preview`` opset at version 1.
    bool has_preview_opset = false;
    for (const auto &osid : tc->model.ref_opset_import()) {
      const auto &d = osid.ref_domain();
      if (std::string(d.data(), d.size()) == "ai.onnx.preview") {
        EXPECT_EQ(osid.version(), 1);
        has_preview_opset = true;
      }
    }
    EXPECT_TRUE(has_preview_opset);
  }

  // Basic case output shape matches (B, Hq, Lq, Dv) = (1, 2, 2, 2).
  EXPECT_EQ(basic->data_sets[0].outputs[0].shape, (std::vector<int64_t>{1, 2, 2, 2}));
  // GQA case output shape matches (B, Hq, Lq, Dv) = (1, 4, 2, 2).
  EXPECT_EQ(gqa->data_sets[0].outputs[0].shape, (std::vector<int64_t>{1, 4, 2, 2}));

  // The basic / gqa cases must not carry any modifier attributes.
  EXPECT_EQ(basic->model.ref_graph().ref_node()[0].ref_attribute().size(), 0u);
  EXPECT_EQ(gqa->model.ref_graph().ref_node()[0].ref_attribute().size(), 0u);

  // Both prob_mod cases must carry exactly one GRAPH attribute named
  // ``prob_mod``, with a structurally non-empty body (``node_size() > 0``)
  // so the function-body builder does not skip it as an identity.
  for (const TestCase *tc : {prob_mod_id, prob_mod_scale}) {
    const NodeProto &node = tc->model.ref_graph().ref_node()[0];
    ASSERT_EQ(node.ref_attribute().size(), 1u) << tc->name;
    const AttributeProto &attr = node.ref_attribute()[0];
    EXPECT_EQ(std::string(attr.ref_name().data(), attr.ref_name().size()), "prob_mod") << tc->name;
    EXPECT_EQ(attr.type(), AttributeProto::AttributeType::GRAPH) << tc->name;
    ASSERT_TRUE(attr.has_g()) << tc->name;
    const GraphProto &body = attr.ref_g();
    ASSERT_EQ(body.ref_input().size(), 1u) << tc->name;
    ASSERT_EQ(body.ref_output().size(), 1u) << tc->name;
    EXPECT_GT(body.ref_node().size(), 0u) << tc->name;
  }

  // Identity-prob_mod expected output equals the basic case output.
  ASSERT_EQ(prob_mod_id->data_sets[0].outputs[0].shape, basic->data_sets[0].outputs[0].shape);
  {
    const float *baseline = basic->data_sets[0].outputs[0].AsFloat();
    const float *modified = prob_mod_id->data_sets[0].outputs[0].AsFloat();
    const int64_t n = basic->data_sets[0].outputs[0].element_count();
    ASSERT_EQ(n, prob_mod_id->data_sets[0].outputs[0].element_count());
    for (int64_t i = 0; i < n; ++i) {
      EXPECT_FLOAT_EQ(modified[i], baseline[i]);
    }
  }
  // Scale-by-0.5 prob_mod expected output equals 0.5 times the basic case.
  ASSERT_EQ(prob_mod_scale->data_sets[0].outputs[0].shape, basic->data_sets[0].outputs[0].shape);
  {
    const float *baseline = basic->data_sets[0].outputs[0].AsFloat();
    const float *modified = prob_mod_scale->data_sets[0].outputs[0].AsFloat();
    const int64_t n = basic->data_sets[0].outputs[0].element_count();
    ASSERT_EQ(n, prob_mod_scale->data_sets[0].outputs[0].element_count());
    for (int64_t i = 0; i < n; ++i) {
      EXPECT_FLOAT_EQ(modified[i], 0.5f * baseline[i]);
    }
  }
}

} // namespace Test
