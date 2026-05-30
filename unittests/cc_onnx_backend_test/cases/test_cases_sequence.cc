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

TEST(BackendTestCase, ConcatFromSequenceCasesAreRegistered) {
  auto cases = CollectTestCases();
  const TestCase *axis_0 = nullptr;
  const TestCase *axis_1 = nullptr;
  const TestCase *new_axis = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_concat_from_sequence_axis_0") {
      axis_0 = &c;
    } else if (c.name == "test_cc_concat_from_sequence_axis_1") {
      axis_1 = &c;
    } else if (c.name == "test_cc_concat_from_sequence_new_axis") {
      new_axis = &c;
    }
  }
  ASSERT_NE(axis_0, nullptr);
  ASSERT_NE(axis_1, nullptr);
  ASSERT_NE(new_axis, nullptr);

  // All three cases share the same in-graph topology: a
  // SequenceConstruct that bundles tensor inputs into a sequence, then
  // a ConcatFromSequence node that consumes the sequence.
  for (const TestCase *tc : {axis_0, axis_1, new_axis}) {
    const GraphProto &graph = tc->model.ref_graph();
    ASSERT_EQ(graph.ref_node().size(), 2u) << tc->name;
    const NodeProto &seq_node = graph.ref_node()[0];
    EXPECT_EQ(seq_node.ref_op_type().as_string(), "SequenceConstruct");
    EXPECT_EQ(seq_node.ref_input().size(), 3u);
    EXPECT_EQ(seq_node.ref_output().size(), 1u);

    const NodeProto &concat_node = graph.ref_node()[1];
    EXPECT_EQ(concat_node.ref_op_type().as_string(), "ConcatFromSequence");
    ASSERT_EQ(concat_node.ref_input().size(), 1u);
    EXPECT_EQ(concat_node.ref_input()[0].as_string(), seq_node.ref_output()[0].as_string());
    ASSERT_EQ(concat_node.ref_output().size(), 1u);
    EXPECT_EQ(concat_node.ref_output()[0].as_string(), "concat_result");

    ASSERT_EQ(graph.ref_input().size(), 3u);
    ASSERT_EQ(graph.ref_output().size(), 1u);
    const ValueInfoProto &out_vi = graph.ref_output()[0];
    EXPECT_EQ(out_vi.ref_type().has_tensor_type(), true);

    ASSERT_EQ(tc->data_sets.size(), 1u);
    const auto &ds = tc->data_sets[0];
    ASSERT_EQ(ds.inputs.size(), 3u);
    ASSERT_EQ(ds.outputs.size(), 1u);
    EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  }

  // Expected output shapes per case.
  EXPECT_EQ(axis_0->data_sets[0].outputs[0].shape, (std::vector<int64_t>{6, 3}));
  EXPECT_EQ(axis_1->data_sets[0].outputs[0].shape, (std::vector<int64_t>{2, 9}));
  EXPECT_EQ(new_axis->data_sets[0].outputs[0].shape, (std::vector<int64_t>{3, 2, 3}));
}

TEST(BackendTestCase, SequenceLengthCaseIsPresent) {
  auto cases = CollectTestCases("SequenceLength");
  const TestCase *length_case = nullptr;
  for (const auto &c : cases) {
    if (c.name == "test_cc_sequence_length") {
      length_case = &c;
      break;
    }
  }
  ASSERT_NE(length_case, nullptr);

  const GraphProto &graph = length_case->model.ref_graph();
  ASSERT_EQ(graph.ref_node().size(), 2u);
  EXPECT_EQ(graph.ref_node()[0].ref_op_type().as_string(), "SequenceConstruct");
  EXPECT_EQ(graph.ref_node()[1].ref_op_type().as_string(), "SequenceLength");
  ASSERT_EQ(graph.ref_output().size(), 1u);
  const ValueInfoProto &out_vi = graph.ref_output()[0];
  ASSERT_TRUE(out_vi.ref_type().has_tensor_type());
  EXPECT_EQ(out_vi.ref_type().ref_tensor_type().ref_elem_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(out_vi.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 0u);

  ASSERT_EQ(length_case->data_sets.size(), 1u);
  const auto &ds = length_case->data_sets[0];
  ASSERT_EQ(ds.inputs.size(), 3u);
  ASSERT_EQ(ds.outputs.size(), 1u);
  EXPECT_EQ(ds.outputs[0].data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_TRUE(ds.outputs[0].shape.empty());
  ASSERT_EQ(ds.outputs[0].data.size(), sizeof(int64_t));
  EXPECT_EQ(*ds.outputs[0].AsInt64(), 3);
}

} // namespace Test
