// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Registers one SequenceLength node case where the input sequence is
// created in-graph via SequenceConstruct so the harness can still feed
// regular tensor inputs.
void RegisterSequenceLengthCase(const std::string &name, const std::vector<Tensor> &inputs,
                                const OpsetId &opset, std::vector<TestCase> &registry) {
  TestCase lazy_case(name, name);
  lazy_case.rtol = 1e-3;
  lazy_case.atol = 1e-7;
  lazy_case.build = [=](bool) -> BuiltCase {
    const Sequence seq = MakeReferenceKernel<onnx_kernels::kernel::SequenceConstruct>(opset).Invoke(
        [&](const auto &kernel) { return kernel.AsSequence(inputs); });
    Tensor expected = MakeReferenceKernel<onnx_kernels::kernel::SequenceLength>(opset).Invoke(
        [&](const auto &kernel) { return kernel(seq); });
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
    for (const Tensor &t : inputs) {
      seq_node->add_input(t.name);
    }
    seq_node->add_output("input_seq");

    NodeProto *len_node = graph->add_node();
    len_node->set_op_type("SequenceLength");
    len_node->add_input("input_seq");
    len_node->add_output("length");

    for (const Tensor &t : inputs) {
      FillValueInfo(t, *graph->add_input());
    }
    FillValueInfo(expected, *graph->add_output());

    TestCase tc(name, name);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;
    tc.set_model(std::move(model));
    DataSet ds;
    ds.inputs = inputs;
    ds.outputs = {expected};
    tc.data_sets().emplace_back(std::move(ds));
    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace

void RegisterSequenceLengthCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> big_shape = {512, 512};
    std::vector<Tensor> inputs;
    inputs.reserve(8);
    for (int i = 0; i < 8; ++i) {
      inputs.push_back(
          Tensor::FromFloat("t" + std::to_string(i), big_shape, Randn<float>(big_shape, 2001 + i)));
    }
    RegisterSequenceLengthCase("test_cc_sequence_length_benchmark", inputs, opset, registry);
    return;
  }

  const Tensor a = Tensor::FromFloat("a", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  const Tensor b = Tensor::FromFloat("b", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  const Tensor c = Tensor::FromFloat("c", {2, 3}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});
  RegisterSequenceLengthCase("test_cc_sequence_length", {a, b, c}, opset, registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
