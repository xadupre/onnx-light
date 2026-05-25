// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::Tensor;
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

TEST(BackendTestCase, FlexAttentionCasesArePresent) {
  const auto cases = CollectTestCases();
  const TestCase *basic = FindCase(cases, "test_cc_flex_attention_basic");
  const TestCase *gqa = FindCase(cases, "test_cc_flex_attention_gqa");
  ASSERT_NE(basic, nullptr);
  ASSERT_NE(gqa, nullptr);

  for (const TestCase *tc : {basic, gqa}) {
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
      EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
      EXPECT_EQ(t.shape.size(), 4u);
    }
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
}

} // namespace Test
