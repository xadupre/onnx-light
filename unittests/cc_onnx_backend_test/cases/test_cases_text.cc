// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTextTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectTextTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, StringConcatCaseIsPresent) {
  auto cases = CollectTestCases("StringConcat");
  const TestCase *equal_case = nullptr;
  const TestCase *bcast_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_string_concat") {
      equal_case = &c;
    } else if (c.name == "test_cc_string_concat_bcast") {
      bcast_case = &c;
    }
  }
  ASSERT_NE(equal_case, nullptr);
  ASSERT_NE(bcast_case, nullptr);

  // Equal-shape case: single StringConcat node taking two STRING inputs and
  // producing a STRING output of the same shape, with the expected
  // element-wise concatenated values.
  {
    const GraphProto &graph = equal_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "StringConcat");
    EXPECT_EQ(graph.ref_input().size(), 2u);
    ASSERT_EQ(graph.ref_output().size(), 1u);

    ASSERT_EQ(equal_case->data_sets.size(), 1u);
    const auto &ds = equal_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING));
    const std::vector<int64_t> expected_shape = {3};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"abcdef", "xyz", "hello world"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }

  // Scalar-broadcast case: rhs is a scalar STRING tensor; output keeps the
  // lhs 2x2 shape and broadcasts the scalar to every element.
  {
    ASSERT_EQ(bcast_case->data_sets.size(), 1u);
    const auto &ds = bcast_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 2u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.inputs[1].shape, std::vector<int64_t>{});
    const std::vector<int64_t> expected_shape = {2, 2};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"a!", "b!", "c!", "d!"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }
}

TEST(BackendTestCase, StringNormalizerCaseIsPresent) {
  auto cases = CollectTestCases("StringNormalizer");
  const TestCase *lower_case = nullptr;
  const TestCase *upper_case = nullptr;
  const TestCase *all_dropped_case = nullptr;
  const TestCase *nostopwords_case = nullptr;
  const TestCase *case_sensitive_lower_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_string_normalizer_lower") {
      lower_case = &c;
    } else if (c.name == "test_cc_string_normalizer_upper") {
      upper_case = &c;
    } else if (c.name == "test_cc_string_normalizer_all_dropped") {
      all_dropped_case = &c;
    } else if (c.name == "test_cc_string_normalizer_nostopwords_nochangecase") {
      nostopwords_case = &c;
    } else if (c.name == "test_cc_string_normalizer_case_sensitive_lower") {
      case_sensitive_lower_case = &c;
    }
  }
  ASSERT_NE(lower_case, nullptr);
  ASSERT_NE(upper_case, nullptr);
  ASSERT_NE(all_dropped_case, nullptr);
  ASSERT_NE(nostopwords_case, nullptr);
  ASSERT_NE(case_sensitive_lower_case, nullptr);

  // Lowercase variant: 1-D input, every element lowercased, no stopwords.
  {
    const GraphProto &graph = lower_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "StringNormalizer");

    ASSERT_EQ(lower_case->data_sets.size(), 1u);
    const auto &ds = lower_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING));
    const std::vector<int64_t> expected_shape = {3};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"hello", "world", "foo"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }

  // Uppercase + stopwords variant on a 2-D [1, C] input — dropped "A" / "a".
  {
    ASSERT_EQ(upper_case->data_sets.size(), 1u);
    const auto &ds = upper_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    const std::vector<int64_t> expected_shape = {1, 2};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"HELLO", "WORLD"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }

  // All-dropped variant: input collapses to a single empty string at [1].
  {
    ASSERT_EQ(all_dropped_case->data_sets.size(), 1u);
    const auto &ds = all_dropped_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    const std::vector<int64_t> expected_shape = {1};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    ASSERT_EQ(ds.outputs[0].string_data.size(), 1u);
    EXPECT_EQ(ds.outputs[0].string_data[0], "");
  }

  // NOOP variant: no stopwords and no case change; output equals input.
  {
    ASSERT_EQ(nostopwords_case->data_sets.size(), 1u);
    const auto &ds = nostopwords_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING));
    const std::vector<int64_t> expected_shape = {2};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"monday", "tuesday"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }

  // Case-sensitive stopword drop + LOWER variant: "monday" stopword removes
  // the first element; the survivors are lowercased (already lower here).
  {
    ASSERT_EQ(case_sensitive_lower_case->data_sets.size(), 1u);
    const auto &ds = case_sensitive_lower_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    const std::vector<int64_t> expected_shape = {3};
    EXPECT_EQ(ds.outputs[0].shape, expected_shape);
    const std::vector<std::string> expected_strings = {"tuesday", "wednesday", "thursday"};
    EXPECT_EQ(ds.outputs[0].string_data, expected_strings);
  }
}

TEST(BackendTestCase, StringSplitCaseIsPresent) {
  auto cases = CollectTestCases("StringSplit");
  const TestCase *basic_case = nullptr;
  const TestCase *maxsplit_case = nullptr;
  const TestCase *empty_tensor_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_string_split_basic") {
      basic_case = &c;
    } else if (c.name == "test_cc_string_split_maxsplit") {
      maxsplit_case = &c;
    } else if (c.name == "test_cc_string_split_empty_tensor") {
      empty_tensor_case = &c;
    }
  }
  ASSERT_NE(basic_case, nullptr);
  ASSERT_NE(maxsplit_case, nullptr);
  ASSERT_NE(empty_tensor_case, nullptr);

  {
    const GraphProto &graph = basic_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "StringSplit");
    ASSERT_EQ(basic_case->data_sets.size(), 1u);
    const auto &ds = basic_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::STRING));
    EXPECT_EQ(ds.outputs[1].data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(ds.outputs[0].string_data, (std::vector<std::string>{"abc", "com", "def", "net"}));
  }

  {
    ASSERT_EQ(maxsplit_case->data_sets.size(), 1u);
    const auto &ds = maxsplit_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{2, 2, 3}));
    const int64_t *counts = ds.outputs[1].AsInt64();
    ASSERT_NE(counts, nullptr);
    EXPECT_EQ(counts[0], 2);
    EXPECT_EQ(counts[1], 1);
    EXPECT_EQ(counts[2], 3);
    EXPECT_EQ(counts[3], 3);
  }

  {
    ASSERT_EQ(empty_tensor_case->data_sets.size(), 1u);
    const auto &ds = empty_tensor_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 2u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0, 0}));
    EXPECT_TRUE(ds.outputs[0].string_data.empty());
    EXPECT_EQ(ds.outputs[1].shape, (std::vector<int64_t>{0}));
  }
}

TEST(BackendTestCase, RegexFullMatchCasesArePresent) {
  auto cases = CollectTestCases("RegexFullMatch");
  const TestCase *basic_case = nullptr;
  const TestCase *empty_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_regex_full_match_basic") {
      basic_case = &c;
    } else if (c.name == "test_cc_regex_full_match_empty") {
      empty_case = &c;
    }
  }
  ASSERT_NE(basic_case, nullptr);
  ASSERT_NE(empty_case, nullptr);

  {
    const GraphProto &graph = basic_case->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 1u);
    const NodeProto &node = graph.ref_node()[0];
    const auto &op_type = node.ref_op_type();
    EXPECT_EQ(std::string(op_type.data(), op_type.size()), "RegexFullMatch");
    ASSERT_EQ(basic_case->data_sets.size(), 1u);
    const auto &ds = basic_case->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 1u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
    EXPECT_EQ(ds.outputs[0].shape, ds.inputs[0].shape);
  }

  {
    ASSERT_EQ(empty_case->data_sets.size(), 1u);
    const auto &ds = empty_case->data_sets[0];
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].shape, (std::vector<int64_t>{0}));
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  }
}

} // namespace Test
