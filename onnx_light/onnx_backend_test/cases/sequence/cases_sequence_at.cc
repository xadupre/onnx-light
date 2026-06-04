// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 13;

// Builds and registers one SequenceAt test case.
//
// The model graph contains two nodes:
//   1. ``SequenceConstruct`` that bundles ``inputs`` into a tensor sequence.
//   2. ``SequenceAt`` that selects the element at ``position``.
//
// The expected output tensor is computed with the reference kernels.
void RegisterSequenceAtCase(const std::string &name, const std::vector<Tensor> &inputs,
                            int64_t position, const OpsetId &opset,
                            std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // Compute expected output with the reference kernel.
  const Sequence seq = kernel::SequenceConstruct(ctx).AsSequence(inputs);
  Tensor position_tensor = Tensor::FromInt64("position", {}, {position});
  Tensor expected = kernel::SequenceAt(ctx)(seq, position_tensor);
  expected.name = "output_tensor";

  TestCase tc(name, name);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Node 1: SequenceConstruct <inputs…> → input_seq.
  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    seq_node->add_input(inputs[i].name);
  }
  seq_node->add_output("input_seq");

  // Node 2: SequenceAt input_seq, position → output_tensor.
  NodeProto *at_node = graph->add_node();
  at_node->set_op_type("SequenceAt");
  at_node->add_input("input_seq");
  at_node->add_input("position");
  at_node->add_output("output_tensor");

  // Graph inputs: the individual tensors and the position scalar.
  for (const Tensor &t : inputs) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(position_tensor, *graph->add_input());

  // Graph output: the selected tensor.
  FillValueInfo(expected, *graph->add_output());

  // DataSet: feed the original tensors and the position.
  DataSet ds;
  for (const Tensor &t : inputs) {
    ds.inputs.push_back(t);
  }
  ds.inputs.push_back(position_tensor);
  ds.outputs.push_back(expected);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceAt — returns the tensor at a given position in a tensor sequence
// (since opset 11 in the ai.onnx domain).
//
// Three cases are registered, each using three FLOAT tensors of shape [2, 3]:
//   * ``test_cc_sequence_at_pos0``: selects the first element.
//   * ``test_cc_sequence_at_pos2``: selects the last (third) element.
//   * ``test_cc_sequence_at_neg``:  selects element at position -2
//                                   (same as index 1 for n=3).
// ---------------------------------------------------------------------------
void RegisterSequenceAtCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  const std::vector<int64_t> elem_shape = {2, 3};
  Tensor a = Tensor::FromFloat("a", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor b = Tensor::FromFloat("b", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor c = Tensor::FromFloat("c", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

  // Case 1: position 0 → selects a.
  RegisterSequenceAtCase("test_cc_sequence_at_pos0", {a, b, c}, /*position=*/0, opset, registry);

  // Case 2: position 2 → selects c.
  RegisterSequenceAtCase("test_cc_sequence_at_pos2", {a, b, c}, /*position=*/2, opset, registry);

  // Case 3: position -2 → index 1 for n=3, selects b.
  RegisterSequenceAtCase("test_cc_sequence_at_neg", {a, b, c}, /*position=*/-2, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
