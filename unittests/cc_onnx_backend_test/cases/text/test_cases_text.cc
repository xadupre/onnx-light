// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, StringConcatCaseIsPresent) {
  auto cases = CollectTestCases();
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
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::STRING));
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

} // namespace Test
