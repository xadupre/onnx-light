// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 13;

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
  TestCase lazy_case(name, name);
  lazy_case.rtol = 1e-3;
  lazy_case.atol = 1e-7;
  lazy_case.build = [opset, inputs, has_position, position, name, elem_shape](bool) -> BuiltCase {
    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::SequenceConstruct kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::SequenceErase kernel_2{ctx_2};
    const KernelContext ctx_3{opset};
    const onnx_kernels::kernel::SequenceConstruct kernel_3{ctx_3};

    // Compute the expected output sequence with the reference kernel.
    const Sequence seq = kernel_1.AsSequence(inputs);
    Tensor position_tensor;
    const Tensor *pos_ptr = nullptr;
    if (has_position) {
      position_tensor = Tensor::FromInt64("position", {}, {position});
      pos_ptr = &position_tensor;
    }
    const Sequence out_seq = kernel_2(seq, pos_ptr);

    // Materialise the output sequence as a stacked tensor.
    std::vector<Tensor> remaining(out_seq.values.begin(), out_seq.values.end());
    Tensor stacked = kernel_3(remaining);
    stacked.name = "output_sequence";

    TestCase tc(name, name);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
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

    // Graph output: ``output_sequence`` declared as a SequenceType.
    AppendValueInfo(
        *graph->add_output(), stacked.name,
        SequenceTypeSpec(TensorTypeSpec(static_cast<int32_t>(out_seq.elem_type), elem_shape)));

    // DataSet: feed the original tensors (and position if provided).
    DataSet ds;
    for (const Tensor &t : inputs) {
      ds.inputs.push_back(t);
    }
    if (has_position) {
      ds.inputs.push_back(position_tensor);
    }
    ds.outputs.push_back(stacked);
    tc.data_sets().emplace_back(std::move(ds));

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
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
void RegisterSequenceEraseCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> big_shape = {512, 512};
    std::vector<Tensor> inputs;
    inputs.reserve(8);
    for (int i = 0; i < 8; ++i) {
      inputs.push_back(
          Tensor::FromFloat("t" + std::to_string(i), big_shape, Randn<float>(big_shape, 2001 + i)));
    }
    RegisterSequenceEraseCase("test_cc_sequence_erase_benchmark", inputs, big_shape,
                              /*has_position=*/true, /*position=*/3, opset, registry);
    return;
  }

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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
