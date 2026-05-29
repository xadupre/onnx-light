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

// IR version used by the manually-built models below. Kept in sync with
// the default in ``test_case.cc``.
constexpr int64_t kDefaultIrVersion = 13;

// Builds a ``ConcatFromSequence`` test case with three FLOAT tensor
// graph inputs ``a``, ``b``, ``c`` (all of shape ``elem_shape``). The
// model graph contains two nodes: a ``SequenceConstruct`` node that
// bundles the three inputs into a tensor sequence, followed by a
// ``ConcatFromSequence`` node that concatenates (or stacks, when
// ``new_axis == 1``) the sequence along ``axis`` to produce
// ``concat_result``. ``concat_result`` is the single graph output.
//
// The graph cannot accept a sequence-typed input directly (the test
// harness feeds tensor data), so wrapping the inputs in an in-graph
// ``SequenceConstruct`` lets the ``ConcatFromSequence`` schema be
// exercised end-to-end with regular tensor inputs and outputs.
void RegisterConcatFromSequenceCase(const std::string &name, const std::vector<int64_t> &elem_shape,
                                    const Tensor &a, const Tensor &b, const Tensor &c, int64_t axis,
                                    int64_t new_axis, const OpsetId &opset,
                                    std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};
  Tensor expected = kernel::ConcatFromSequence(ctx)({a, b, c}, axis, new_axis);
  expected.name = "concat_result";

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "node";
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

  // Node 1: SequenceConstruct ``a``, ``b``, ``c`` → ``input_seq``.
  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  seq_node->add_input("a");
  seq_node->add_input("b");
  seq_node->add_input("c");
  seq_node->add_output("input_seq");

  // Node 2: ConcatFromSequence ``input_seq`` → ``concat_result``.
  NodeProto *concat_node = graph->add_node();
  concat_node->set_op_type("ConcatFromSequence");
  concat_node->add_input("input_seq");
  concat_node->add_output("concat_result");
  AttributeProto *axis_attr = concat_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::AttributeType::INT);
  axis_attr->set_i(axis);
  if (new_axis != 0) {
    AttributeProto *new_axis_attr = concat_node->add_attribute();
    new_axis_attr->set_name("new_axis");
    new_axis_attr->set_type(AttributeProto::AttributeType::INT);
    new_axis_attr->set_i(new_axis);
  }

  // Graph inputs ``a``, ``b``, ``c`` (tensor-typed).
  for (const Tensor *t : {&a, &b, &c}) {
    Tensor named = *t;
    if (named.name.empty()) {
      named.name = (t == &a) ? "a" : (t == &b ? "b" : "c");
    }
    FillValueInfo(named, *graph->add_input());
  }

  // Graph output ``concat_result`` (tensor-typed).
  FillValueInfo(expected, *graph->add_output());

  DataSet ds;
  ds.inputs.reserve(3);
  {
    Tensor ta = a;
    ta.name = "a";
    ds.inputs.emplace_back(std::move(ta));
    Tensor tb = b;
    tb.name = "b";
    ds.inputs.emplace_back(std::move(tb));
    Tensor tc_in = c;
    tc_in.name = "c";
    ds.inputs.emplace_back(std::move(tc_in));
  }
  ds.outputs.emplace_back(expected);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace

// ---------------------------------------------------------------------------
// ConcatFromSequence — concatenates (or stacks, when ``new_axis == 1``)
// the tensor elements of an input sequence along ``axis`` (since opset
// 11 in the ai.onnx domain).
//
// Three cases are registered, each using three FLOAT tensors of shape
// ``[2, 3]`` wrapped in an in-graph ``SequenceConstruct``:
//   * ``test_cc_concat_from_sequence_axis_0``: concatenate along axis 0
//     → FLOAT ``[6, 3]``.
//   * ``test_cc_concat_from_sequence_axis_1``: concatenate along axis 1
//     → FLOAT ``[2, 9]``.
//   * ``test_cc_concat_from_sequence_new_axis``: stack at new axis 0
//     → FLOAT ``[3, 2, 3]``.
// ---------------------------------------------------------------------------
void RegisterConcatFromSequenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  const std::vector<int64_t> elem_shape = {2, 3};
  const Tensor a = Tensor::FromFloat("", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  const Tensor b = Tensor::FromFloat("", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  const Tensor c = Tensor::FromFloat("", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

  RegisterConcatFromSequenceCase("test_cc_concat_from_sequence_axis_0", elem_shape, a, b, c,
                                 /*axis=*/0, /*new_axis=*/0, opset, registry);
  RegisterConcatFromSequenceCase("test_cc_concat_from_sequence_axis_1", elem_shape, a, b, c,
                                 /*axis=*/1, /*new_axis=*/0, opset, registry);
  RegisterConcatFromSequenceCase("test_cc_concat_from_sequence_new_axis", elem_shape, a, b, c,
                                 /*axis=*/0, /*new_axis=*/1, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
