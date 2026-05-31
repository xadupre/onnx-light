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

// Builds a trivial SequenceMap body subgraph:
//
//   inputs : (elem [<dtype>, elem_shape])
//   nodes  : out = Identity(elem)
//   outputs: (out [<dtype>, elem_shape])
//
// The body has one input (the per-iteration sequence element) and one
// output, so the SequenceMap node produces exactly one output sequence.
// The reference kernel does not execute the body; it is included in the
// model purely so the registered ``TestCase`` is a well-formed ONNX model
// with a valid ``SequenceMap`` node.
GraphProto BuildIdentityMapBody(int32_t elem_type, const std::vector<int64_t> &elem_shape) {
  GraphProto g;
  g.set_name("seq_map_body");

  // Input ``elem`` carries the per-iteration sequence element.
  ValueInfoProto *in_vi = g.add_input();
  in_vi->set_name("elem");
  TypeProto::Tensor *in_tt = in_vi->ref_type().mutable_tensor_type();
  in_tt->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto &in_shape = in_tt->ref_shape();
  for (int64_t d : elem_shape) {
    in_shape.add_dim()->set_dim_value(d);
  }

  // out = Identity(elem)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("elem");
    n->add_output("out");
  }

  // Output ``out`` has the same dtype and shape as the per-iteration element.
  ValueInfoProto *out_vi = g.add_output();
  out_vi->set_name("out");
  TypeProto::Tensor *out_tt = out_vi->ref_type().mutable_tensor_type();
  out_tt->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto &out_shape = out_tt->ref_shape();
  for (int64_t d : elem_shape) {
    out_shape.add_dim()->set_dim_value(d);
  }

  return g;
}

// Builds and registers a SequenceMap test case whose body simply forwards
// each input element via ``Identity``. The model graph contains two nodes:
//
//   1. ``SequenceConstruct`` that bundles ``inputs`` into a tensor sequence.
//   2. ``SequenceMap`` with the identity body subgraph above.
//
// The expected stacked output is computed with the reference kernels and
// materialized as a ``Tensor<elem_type, [n, *elem_shape]>``. The graph
// output type is then promoted to ``SequenceType<Tensor<elem_type,
// elem_shape>>``.
void RegisterSequenceMapIdentityCase(const std::string &name, const std::vector<Tensor> &inputs,
                                     const std::vector<int64_t> &elem_shape, int32_t elem_type,
                                     const OpsetId &opset, std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // Build the input sequence and the body-output rows (identity → inputs).
  const Sequence in_seq = kernel::SequenceConstruct(ctx).AsSequence(inputs);
  std::vector<std::vector<Tensor>> body_outputs_per_iter = {inputs};
  std::vector<Sequence> out_seqs = kernel::SequenceMap(ctx)(in_seq, body_outputs_per_iter);

  // Materialise the (single) output sequence as a stacked tensor.
  std::vector<Tensor> stacked_in(out_seqs[0].values.begin(), out_seqs[0].values.end());
  Tensor stacked = kernel::SequenceConstruct(ctx)(stacked_in);
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
  for (const Tensor &t : inputs) {
    seq_node->add_input(t.name);
  }
  seq_node->add_output("input_seq");

  // Node 2: SequenceMap(input_seq, body=identity) → output_sequence.
  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("input_seq");
  map_node->add_output("output_sequence");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildIdentityMapBody(elem_type, elem_shape);

  // Graph inputs: the individual tensors.
  for (const Tensor &t : inputs) {
    FillValueInfo(t, *graph->add_input());
  }

  // Graph output: ``output_sequence`` declared as tensor (promoted below).
  FillValueInfo(stacked, *graph->add_output());

  // DataSet: feed the original tensors, expect the stacked output.
  DataSet ds;
  for (const Tensor &t : inputs) {
    ds.inputs.push_back(t);
  }
  ds.outputs.push_back(std::move(stacked));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  // Promote the output value-info to SequenceType.
  PromoteOutputToSequenceType(registry, elem_type, elem_shape);
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceMap — applies a sub-graph to each element of a tensor sequence
// (since opset 17 in the ai.onnx domain).
//
// Two cases are registered, each using an identity body subgraph (the
// reference kernel does not execute the body; it merely assembles the
// per-iteration outputs into output sequences):
//
//   * ``test_cc_sequence_map_identity_float``: three FLOAT tensors of
//     shape [2, 3], mapped via Identity → sequence of three FLOAT [2, 3].
//   * ``test_cc_sequence_map_identity_int64``: two INT64 tensors of shape
//     [4], mapped via Identity → sequence of two INT64 [4].
// ---------------------------------------------------------------------------
void RegisterSequenceMapCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(17);

  // Case 1: three FLOAT tensors of shape [2, 3].
  {
    const std::vector<int64_t> elem_shape = {2, 3};
    Tensor a = Tensor::FromFloat("a", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor b = Tensor::FromFloat("b", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor c = Tensor::FromFloat("c", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    RegisterSequenceMapIdentityCase("test_cc_sequence_map_identity_float", {a, b, c}, elem_shape,
                                    static_cast<int32_t>(DataType::FLOAT), opset, registry);
  }

  // Case 2: two INT64 tensors of shape [4].
  {
    const std::vector<int64_t> elem_shape = {4};
    Tensor a = Tensor::FromInt64("a", elem_shape, {-1, 0, 1, 2});
    Tensor b = Tensor::FromInt64("b", elem_shape, {3, 4, 5, 6});

    RegisterSequenceMapIdentityCase("test_cc_sequence_map_identity_int64", {a, b}, elem_shape,
                                    static_cast<int32_t>(DataType::INT64), opset, registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
