// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectSequenceTestCases;

namespace {
std::vector<onnx_backend_test::TestCase> CollectTestCases(const std::string &op_type = "") {
  std::vector<onnx_backend_test::TestCase> registry;
  CollectSequenceTestCases(registry, op_type);
  return registry;
}
} // namespace
using onnx_backend_test::TestCase;

namespace Test {

TEST(BackendTestCase, SequenceConstructCaseIsPresent) {
  auto cases = CollectTestCases("SequenceConstruct");
  const TestCase *seq_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_sequence_construct") {
      seq_case = &c;
    }
  }
  ASSERT_NE(seq_case, nullptr);

  const GraphProto &graph = seq_case->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 1u);
  const NodeProto &node = graph.ref_node()[0];
  const auto &op_type = node.ref_op_type();
  EXPECT_EQ(std::string(op_type.data(), op_type.size()), "SequenceConstruct");
  EXPECT_EQ(graph.ref_input().size(), 3u);
  ASSERT_EQ(graph.ref_output().size(), 1u);

  // Output value-info must be promoted to Sequence<Tensor<FLOAT, [2, 3]>>.
  const ValueInfoProto &out_vi = graph.ref_output()[0];
  const TypeProto &out_tp = out_vi.ref_type();
  ASSERT_TRUE(out_tp.has_sequence_type());
  EXPECT_FALSE(out_tp.has_tensor_type());
  const TypeProto &elem = out_tp.ref_sequence_type().ref_elem_type();
  ASSERT_TRUE(elem.has_tensor_type());
  EXPECT_EQ(elem.ref_tensor_type().ref_elem_type(), TensorProto::DataType::FLOAT);

  ASSERT_EQ(seq_case->data_sets.size(), 1u);
  const auto &ds = seq_case->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {3, 2, 3};
  EXPECT_EQ(ds.outputs[0].shape, expected_shape);
  // Stacked output bytes equal the concatenation of the per-input buffers.
  std::vector<uint8_t> expected_bytes;
  for (const auto &in : ds.inputs) {
    expected_bytes.insert(expected_bytes.end(), in.data.begin(), in.data.end());
  }
  EXPECT_EQ(ds.outputs[0].data, expected_bytes);
}

} // namespace Test
