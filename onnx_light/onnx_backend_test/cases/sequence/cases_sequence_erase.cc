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

// Promotes the most recently appended test case so that its single graph
// output is declared as a ``SequenceType<Tensor<elem_type, elem_shape>>``.
void PromoteOutputToSequenceType(std::vector<TestCase> &registry, int32_t elem_type,
                                 const std::vector<int64_t> &elem_shape) {
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &out_vi = *graph.mutable_output(0);
  TypeProto &out_tp = out_vi.ref_type();
  TypeProto::Sequence *out_seq = out_tp.add_sequence_type();
  TypeProto *out_elem = out_seq->add_elem_type();
  TypeProto::Tensor *out_tensor = out_elem->add_tensor_type();
  out_tensor->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto *out_shape = out_tensor->add_shape();
  for (int64_t d : elem_shape) {
    out_shape->add_dim()->set_dim_value(d);
  }
  out_tp.reset_tensor_type();
}

// Builds and registers one SequenceErase test case.
//
// The model graph contains two nodes:
//   1. ``SequenceConstruct`` that bundles ``inputs`` into a tensor sequence.
//   2. ``SequenceErase`` that removes the element at ``position`` (or the
//      last element when ``has_position`` is ``false``).
//
// The expected stacked output is computed with the reference kernels and
// materialized as a ``Tensor<elem_type, [n-1, *elem_shape]>`` so that the
// test harness can compare byte buffers. The graph output type is then
// promoted to ``SequenceType<Tensor<elem_type, elem_shape>>``.
void RegisterSequenceEraseCase(const std::string &name, const std::vector<Tensor> &inputs,
                               const std::vector<int64_t> &elem_shape, bool has_position,
                               int64_t position, const OpsetId &opset,
                               std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // Compute the expected output sequence with the reference kernel.
  const Sequence seq = kernel::SequenceConstruct(ctx).AsSequence(inputs);
  Tensor position_tensor;
  const Tensor *pos_ptr = nullptr;
  if (has_position) {
    position_tensor = Tensor::FromInt64("position", {}, {position});
    pos_ptr = &position_tensor;
  }
  const Sequence out_seq = kernel::SequenceErase(ctx)(seq, pos_ptr);

  // Materialise the output sequence as a stacked tensor.
  std::vector<Tensor> remaining(out_seq.values.begin(), out_seq.values.end());
  Tensor stacked = kernel::SequenceConstruct(ctx)(remaining);
  stacked.name = "output_sequence";

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

  // Node 1: SequenceConstruct <inputs…> → input_seq.
  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    seq_node->add_input(inputs[i].name);
  }
  seq_node->add_output("input_seq");

  // Node 2: SequenceErase input_seq [, position] → output_sequence.
  NodeProto *erase_node = graph->add_node();
  erase_node->set_op_type("SequenceErase");
  erase_node->add_input("input_seq");
  if (has_position) {
    erase_node->add_input("position");
  }
  erase_node->add_output("output_sequence");

  // Graph inputs: the individual tensors (and optionally the position scalar).
  for (const Tensor &t : inputs) {
    FillValueInfo(t, *graph->add_input());
  }
  if (has_position) {
    FillValueInfo(position_tensor, *graph->add_input());
  }

  // Graph output: ``output_sequence`` declared as tensor (promoted below).
  FillValueInfo(stacked, *graph->add_output());

  // DataSet: feed the original tensors (and position if provided).
  DataSet ds;
  for (const Tensor &t : inputs) {
    ds.inputs.push_back(t);
  }
  if (has_position) {
    ds.inputs.push_back(position_tensor);
  }
  ds.outputs.push_back(stacked);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  // Promote the output value-info to SequenceType.
  PromoteOutputToSequenceType(registry, static_cast<int32_t>(out_seq.elem_type), elem_shape);
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceErase — removes one element from a tensor sequence (since opset 11
// in the ai.onnx domain).
//
// Three cases are registered, each using three FLOAT tensors of shape [2, 3]:
//   * ``test_cc_sequence_erase_default``: position omitted (erases last).
//   * ``test_cc_sequence_erase_pos1``:    erases element at position 1.
//   * ``test_cc_sequence_erase_neg``:     erases element at position -2
//                                         (same as index 1 for n=3).
// ---------------------------------------------------------------------------
void RegisterSequenceEraseCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  const std::vector<int64_t> elem_shape = {2, 3};
  Tensor a = Tensor::FromFloat("a", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor b = Tensor::FromFloat("b", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor c = Tensor::FromFloat("c", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

  // Case 1: no position argument → erases last element (c).
  RegisterSequenceEraseCase("test_cc_sequence_erase_default", {a, b, c}, elem_shape,
                            /*has_position=*/false, /*position=*/0, opset, registry);

  // Case 2: explicit positive position 1 → erases b.
  RegisterSequenceEraseCase("test_cc_sequence_erase_pos1", {a, b, c}, elem_shape,
                            /*has_position=*/true, /*position=*/1, opset, registry);

  // Case 3: negative position -2 → index 1 for n=3, erases b.
  RegisterSequenceEraseCase("test_cc_sequence_erase_neg", {a, b, c}, elem_shape,
                            /*has_position=*/true, /*position=*/-2, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
