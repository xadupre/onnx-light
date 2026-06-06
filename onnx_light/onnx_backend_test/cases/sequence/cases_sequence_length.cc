// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_kernels/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Registers one SequenceLength node case where the input sequence is
// created in-graph via SequenceConstruct so the harness can still feed
// regular tensor inputs.
void RegisterSequenceLengthCase(const std::string &name, const OpsetId &opset,
                                std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};
  const Tensor a = Tensor::FromFloat("a", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  const Tensor b = Tensor::FromFloat("b", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  const Tensor c = Tensor::FromFloat("c", {2, 3}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
  const Sequence seq = kernel::SequenceConstruct(ctx).AsSequence({a, b, c});
  Tensor expected = kernel::SequenceLength(ctx)(seq);
  expected.name = "length";

  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name(name);
  model.set_ir_version(13);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  seq_node->add_input("a");
  seq_node->add_input("b");
  seq_node->add_input("c");
  seq_node->add_output("input_seq");

  NodeProto *len_node = graph->add_node();
  len_node->set_op_type("SequenceLength");
  len_node->add_input("input_seq");
  len_node->add_output("length");

  FillValueInfo(a, *graph->add_input());
  FillValueInfo(b, *graph->add_input());
  FillValueInfo(c, *graph->add_input());
  FillValueInfo(expected, *graph->add_output());

  TestCase tc(name, name);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  tc.model = std::move(model);
  DataSet ds;
  ds.inputs = {a, b, c};
  ds.outputs = {expected};
  tc.data_sets.emplace_back(std::move(ds));
  registry.emplace_back(std::move(tc));
}

} // namespace

void RegisterSequenceLengthCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);
  RegisterSequenceLengthCase("test_cc_sequence_length", opset, registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
