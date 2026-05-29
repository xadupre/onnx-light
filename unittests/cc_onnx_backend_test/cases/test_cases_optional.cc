// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectOptionalTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectOptionalTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, OptionalCaseIsPresent) {
  auto cases = CollectTestCases("Optional");
  const TestCase *opt_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_optional") {
      opt_case = &c;
    }
  }
  ASSERT_NE(opt_case, nullptr);

  const GraphProto &graph = opt_case->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "Optional");
  EXPECT_EQ(graph.ref_input().size(), 1u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  // Single ``type`` TYPE_PROTO attribute wrapping Optional<Tensor<FLOAT>>.
  ASSERT_EQ(node.ref_attribute().size(), 1u);
  const AttributeProto &attr = node.ref_attribute()[0];
  const auto &attr_name = attr.ref_name();
  EXPECT_EQ(std::string(attr_name.data(), attr_name.size()), "type");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  ASSERT_TRUE(attr.has_tp());
  const TypeProto &tp = attr.ref_tp();
  ASSERT_TRUE(tp.has_optional_type());
  const TypeProto &elem = tp.ref_optional_type().ref_elem_type();
  ASSERT_TRUE(elem.has_tensor_type());
  EXPECT_EQ(elem.ref_tensor_type().ref_elem_type(), TensorProto::DataType::FLOAT);

  ASSERT_EQ(opt_case->data_sets.size(), 1u);
  const auto &ds = opt_case->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 1u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {2, 3};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  // Passthrough: output bytes match input bytes.
  EXPECT_EQ(ds.outputs[0].data, ds.inputs[0].data);
}

} // namespace Test
