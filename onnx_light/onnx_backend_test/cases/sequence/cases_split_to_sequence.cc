// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
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

// Registers a single SplitToSequence test case. The model graph contains
// one ``SplitToSequence`` node consuming ``input`` (and optionally a
// ``split`` initializer). The expected sequence output is materialised
// as a stacked ``Tensor`` so the test harness can compare byte buffers;
// this requires the resolved per-chunk shapes to be identical, which
// every registered case satisfies.
void RegisterSplitToSequenceCase(const std::string &name, const Tensor &input, const Tensor *split,
                                 int64_t axis, int64_t keepdims,
                                 const std::vector<int64_t> &elem_shape, const OpsetId &opset,
                                 std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // Compute the expected output sequence with the reference kernel.
  const Sequence out_seq = kernel::SplitToSequence(ctx)(input, split, axis, keepdims);

  // Materialise the output sequence as a single stacked tensor.
  std::vector<Tensor> chunks(out_seq.values.begin(), out_seq.values.end());
  Tensor stacked = kernel::SequenceConstruct(ctx)(chunks);
  stacked.name = "output_sequence";

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

  // Single SplitToSequence node.
  NodeProto *node = graph->add_node();
  node->set_op_type("SplitToSequence");
  node->add_input(input.name);
  if (split != nullptr) {
    node->add_input(split->name);
  }
  node->add_output("output_sequence");
  AttributeProto *axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::INT);
  axis_attr->set_i(axis);
  if (split == nullptr) {
    AttributeProto *kd_attr = node->add_attribute();
    kd_attr->set_name("keepdims");
    kd_attr->set_type(AttributeProto::INT);
    kd_attr->set_i(keepdims);
  }

  // Graph inputs: data tensor (and split when present).
  FillValueInfo(input, *graph->add_input());
  if (split != nullptr) {
    FillValueInfo(*split, *graph->add_input());
  }

  // Graph output: ``output_sequence`` declared as tensor (promoted below).
  FillValueInfo(stacked, *graph->add_output());

  // DataSet: feed the data tensor (and split if provided).
  DataSet ds;
  ds.inputs.push_back(input);
  if (split != nullptr) {
    ds.inputs.push_back(*split);
  }
  ds.outputs.push_back(stacked);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  // Promote the output value-info to SequenceType<Tensor<elem_type, elem_shape>>.
  PromoteOutputToSequenceType(registry, static_cast<int32_t>(out_seq.elem_type), elem_shape);
}

} // namespace

// ---------------------------------------------------------------------------
// SplitToSequence — splits a tensor into a sequence of tensors along the
// specified axis (since opset 11 in the ai.onnx domain).
//
// Three cases are registered, each producing a sequence whose elements all
// share a common shape so they can be materialised as a single stacked
// output tensor for byte-buffer comparison by the test harness:
//
//   * ``test_cc_split_to_sequence_1``: input ``[3, 6]`` split by the scalar
//     ``2`` along ``axis=1`` → three chunks of shape ``[3, 2]``.
//   * ``test_cc_split_to_sequence_2``: input ``[4, 6]`` split by the 1-D
//     tensor ``[2, 2]`` along ``axis=0`` → two chunks of shape ``[2, 6]``.
//   * ``test_cc_split_to_sequence_nokeepdims``: input ``[3, 6]`` with
//     ``axis=1`` and ``keepdims=0`` (no ``split`` input) → six chunks of
//     shape ``[3]``.
// ---------------------------------------------------------------------------
void RegisterSplitToSequenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  // arange(18) reshaped to [3, 6] as the data input for cases 1 and nokeepdims.
  std::vector<float> data_3x6(18);
  for (int i = 0; i < 18; ++i) {
    data_3x6[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  Tensor data36 = Tensor::FromFloat("data", {3, 6}, data_3x6);

  // Case 1: scalar split.
  Tensor split_scalar = Tensor::FromInt64("split", {}, {2});
  RegisterSplitToSequenceCase("test_cc_split_to_sequence_1", data36, &split_scalar,
                              /*axis=*/1, /*keepdims=*/1, /*elem_shape=*/{3, 2}, opset, registry);

  // Case 2: 1-D split tensor with even chunks.
  std::vector<float> data_4x6(24);
  for (int i = 0; i < 24; ++i) {
    data_4x6[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  Tensor data46 = Tensor::FromFloat("data", {4, 6}, data_4x6);
  Tensor split_vec = Tensor::FromInt64("split", {2}, {2, 2});
  RegisterSplitToSequenceCase("test_cc_split_to_sequence_2", data46, &split_vec,
                              /*axis=*/0, /*keepdims=*/1, /*elem_shape=*/{2, 6}, opset, registry);

  // Case 3: no split input, keepdims=0 (axis is squeezed).
  RegisterSplitToSequenceCase("test_cc_split_to_sequence_nokeepdims", data36, /*split=*/nullptr,
                              /*axis=*/1, /*keepdims=*/0, /*elem_shape=*/{3}, opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
