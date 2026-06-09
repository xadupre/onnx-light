// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 13;

// Promotes the output value-info at ``out_index`` of the most recently
// appended test case so that it is declared as a
// ``SequenceType<Tensor<elem_type, elem_shape>>``.
void PromoteOutputToSequenceType(std::vector<TestCase> &registry, int32_t elem_type,
                                 const std::vector<int64_t> &elem_shape, int out_index = 0) {
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &out_vi = *graph.mutable_output(out_index);
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

// Appends a tensor-typed value-info to ``g`` with the given name, element
// type and shape (used to declare body subgraph inputs/outputs).
void AddBodyTensorIO(ValueInfoProto *vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &elem_shape) {
  vi->set_name(name);
  TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
  tt->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto &sh = tt->ref_shape();
  for (int64_t d : elem_shape) {
    sh.add_dim()->set_dim_value(d);
  }
}

// Sets up the boilerplate ``ModelProto`` of a SequenceMap test case
// (IR version, producer, opset import, named empty graph) and returns
// the freshly added graph for further population by the caller.
GraphProto *InitSequenceMapModel(TestCase &tc, const std::string &name, const OpsetId &opset) {
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
  return graph;
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

  AddBodyTensorIO(g.add_input(), "elem", elem_type, elem_shape);

  // out = Identity(elem)
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("elem");
    n->add_output("out");
  }

  AddBodyTensorIO(g.add_output(), "out", elem_type, elem_shape);

  return g;
}

// Builds a two-input/two-output identity body subgraph:
//
//   inputs : (in0 [<dtype>, shape0]), (in1 [<dtype>, shape1])
//   nodes  : out0 = Identity(in0); out1 = Identity(in1)
//   outputs: (out0 [<dtype>, shape0]), (out1 [<dtype>, shape1])
//
// Used by ``test_cc_sequence_map_identity_2_sequences``.
GraphProto BuildIdentity2InputsBody(int32_t elem_type, const std::vector<int64_t> &shape0,
                                    const std::vector<int64_t> &shape1) {
  GraphProto g;
  g.set_name("seq_map_body");

  AddBodyTensorIO(g.add_input(), "in0", elem_type, shape0);
  AddBodyTensorIO(g.add_input(), "in1", elem_type, shape1);

  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("in0");
    n->add_output("out0");
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Identity");
    n->add_input("in1");
    n->add_output("out1");
  }

  AddBodyTensorIO(g.add_output(), "out0", elem_type, shape0);
  AddBodyTensorIO(g.add_output(), "out1", elem_type, shape1);

  return g;
}

// Builds an element-wise add body subgraph:
//
//   inputs : (in0 [<dtype>, elem_shape]), (in1 [<dtype>, elem_shape])
//   nodes  : out0 = Add(in0, in1)
//   outputs: (out0 [<dtype>, elem_shape])
//
// Used by ``test_cc_sequence_map_add_2_sequences`` and
// ``test_cc_sequence_map_add_1_sequence_1_tensor``.
GraphProto BuildAddBody(int32_t elem_type, const std::vector<int64_t> &elem_shape) {
  GraphProto g;
  g.set_name("seq_map_body");

  AddBodyTensorIO(g.add_input(), "in0", elem_type, elem_shape);
  AddBodyTensorIO(g.add_input(), "in1", elem_type, elem_shape);

  {
    NodeProto *n = g.add_node();
    n->set_op_type("Add");
    n->add_input("in0");
    n->add_input("in1");
    n->add_output("out0");
  }

  AddBodyTensorIO(g.add_output(), "out0", elem_type, elem_shape);

  return g;
}

// Builds a ``Shape`` body subgraph:
//
//   inputs : (x [FLOAT, in_elem_shape])
//   nodes  : shape = Shape(x)
//   outputs: (shape [INT64, [rank]])
//
// Used by ``test_cc_sequence_map_extract_shapes``.
GraphProto BuildShapeBody(int32_t elem_type, const std::vector<int64_t> &in_elem_shape,
                          int64_t rank) {
  GraphProto g;
  g.set_name("seq_map_body");

  // Declare the body input with symbolic dim_params (one per axis) so the
  // runtime does not constant-fold ``Shape`` from a fixed declared shape;
  // the actual per-iteration dims vary across sequence elements.
  ValueInfoProto *vi = g.add_input();
  vi->set_name("x");
  TypeProto::Tensor *tt = vi->ref_type().mutable_tensor_type();
  tt->set_elem_type(static_cast<int>(elem_type));
  TensorShapeProto &sh = tt->ref_shape();
  for (std::size_t i = 0; i < in_elem_shape.size(); ++i) {
    sh.add_dim()->set_dim_param("d" + std::to_string(i));
  }

  {
    NodeProto *n = g.add_node();
    n->set_op_type("Shape");
    n->add_input("x");
    n->add_output("shape");
  }

  AddBodyTensorIO(g.add_output(), "shape", static_cast<int32_t>(DataType::INT64), {rank});

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

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

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

// Mirrors upstream ``test_sequence_map_identity_2_sequences``: two
// input sequences fed through a two-input/two-output Identity body, so
// the SequenceMap node produces two output sequences ``y0 == x0`` and
// ``y1 == x1``.
void RegisterSequenceMapIdentity2SequencesCase(const OpsetId &opset,
                                               std::vector<TestCase> &registry) {
  const std::string name = "test_cc_sequence_map_identity_2_sequences";
  const kernel::KernelContext ctx{opset};
  const std::vector<int64_t> shape0 = {3};
  const std::vector<int64_t> shape1 = {4};

  // Per-iteration tensors of each input sequence.
  std::vector<Tensor> x0 = {
      Tensor::FromFloat("x0_0", shape0, {0.0f, 1.0f, 2.0f}),
      Tensor::FromFloat("x0_1", shape0, {-0.5f, 1.5f, 2.5f}),
      Tensor::FromFloat("x0_2", shape0, {10.0f, 20.0f, 30.0f}),
  };
  std::vector<Tensor> x1 = {
      Tensor::FromFloat("x1_0", shape1, {0.0f, 1.0f, 2.0f, 3.0f}),
      Tensor::FromFloat("x1_1", shape1, {0.25f, 0.5f, 0.75f, 1.0f}),
      Tensor::FromFloat("x1_2", shape1, {-1.0f, -2.0f, -3.0f, -4.0f}),
  };
  const int32_t elem_type = static_cast<int32_t>(DataType::FLOAT);

  // Compose expected output sequences via the reference kernel.
  std::vector<std::vector<Tensor>> body_outputs_per_iter = {x0, x1};
  std::vector<Sequence> out_seqs = kernel::SequenceMap(ctx)(
      kernel::SequenceConstruct(ctx).AsSequence(x0), body_outputs_per_iter);

  // Materialise both output sequences as stacked tensors.
  Tensor stacked0 =
      kernel::SequenceConstruct(ctx)({out_seqs[0].values.begin(), out_seqs[0].values.end()});
  stacked0.name = "y0";
  Tensor stacked1 =
      kernel::SequenceConstruct(ctx)({out_seqs[1].values.begin(), out_seqs[1].values.end()});
  stacked1.name = "y1";

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

  // Node 1+2: build the two input sequences.
  NodeProto *sc0 = graph->add_node();
  sc0->set_op_type("SequenceConstruct");
  for (const Tensor &t : x0) {
    sc0->add_input(t.name);
  }
  sc0->add_output("x0_seq");
  NodeProto *sc1 = graph->add_node();
  sc1->set_op_type("SequenceConstruct");
  for (const Tensor &t : x1) {
    sc1->add_input(t.name);
  }
  sc1->add_output("x1_seq");

  // Node 3: SequenceMap(x0_seq, x1_seq, body=identity_2) → (y0, y1).
  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("x0_seq");
  map_node->add_input("x1_seq");
  map_node->add_output("y0");
  map_node->add_output("y1");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildIdentity2InputsBody(elem_type, shape0, shape1);

  // Graph inputs: individual tensors of both sequences.
  for (const Tensor &t : x0) {
    FillValueInfo(t, *graph->add_input());
  }
  for (const Tensor &t : x1) {
    FillValueInfo(t, *graph->add_input());
  }
  // Graph outputs: y0 and y1 (declared as tensor, promoted below).
  FillValueInfo(stacked0, *graph->add_output());
  FillValueInfo(stacked1, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : x0) {
    ds.inputs.push_back(t);
  }
  for (const Tensor &t : x1) {
    ds.inputs.push_back(t);
  }
  ds.outputs.push_back(std::move(stacked0));
  ds.outputs.push_back(std::move(stacked1));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, elem_type, shape0, /*out_index=*/0);
  PromoteOutputToSequenceType(registry, elem_type, shape1, /*out_index=*/1);
}

// Mirrors upstream ``test_sequence_map_add_2_sequences``: two input
// sequences are mapped through an element-wise Add body, yielding a
// single output sequence ``y0[i] == x0[i] + x1[i]``.
void RegisterSequenceMapAdd2SequencesCase(const OpsetId &opset, std::vector<TestCase> &registry) {
  const std::string name = "test_cc_sequence_map_add_2_sequences";
  const kernel::KernelContext ctx{opset};
  const std::vector<int64_t> elem_shape = {4};
  const int32_t elem_type = static_cast<int32_t>(DataType::FLOAT);

  std::vector<Tensor> x0 = {
      Tensor::FromFloat("x0_0", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f}),
      Tensor::FromFloat("x0_1", elem_shape, {-1.0f, -2.0f, -3.0f, -4.0f}),
      Tensor::FromFloat("x0_2", elem_shape, {0.5f, 1.5f, 2.5f, 3.5f}),
  };
  std::vector<Tensor> x1 = {
      Tensor::FromFloat("x1_0", elem_shape, {10.0f, 10.0f, 10.0f, 10.0f}),
      Tensor::FromFloat("x1_1", elem_shape, {0.25f, 0.5f, 0.75f, 1.0f}),
      Tensor::FromFloat("x1_2", elem_shape, {-0.5f, -1.5f, -2.5f, -3.5f}),
  };

  // Compute expected per-iteration outputs (Add).
  std::vector<Tensor> y_per_iter;
  y_per_iter.reserve(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    const float *a = x0[i].As<float>();
    const float *b = x1[i].As<float>();
    std::vector<float> out(static_cast<std::size_t>(x0[i].element_count()));
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j] = a[j] + b[j];
    }
    y_per_iter.push_back(Tensor::FromFloat("y0_" + std::to_string(i), elem_shape, out));
  }

  // Assemble through the reference kernel and stack into a single output.
  std::vector<std::vector<Tensor>> body_outputs_per_iter = {y_per_iter};
  std::vector<Sequence> out_seqs = kernel::SequenceMap(ctx)(
      kernel::SequenceConstruct(ctx).AsSequence(x0), body_outputs_per_iter);
  Tensor stacked =
      kernel::SequenceConstruct(ctx)({out_seqs[0].values.begin(), out_seqs[0].values.end()});
  stacked.name = "y0";

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

  NodeProto *sc0 = graph->add_node();
  sc0->set_op_type("SequenceConstruct");
  for (const Tensor &t : x0) {
    sc0->add_input(t.name);
  }
  sc0->add_output("x0_seq");
  NodeProto *sc1 = graph->add_node();
  sc1->set_op_type("SequenceConstruct");
  for (const Tensor &t : x1) {
    sc1->add_input(t.name);
  }
  sc1->add_output("x1_seq");

  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("x0_seq");
  map_node->add_input("x1_seq");
  map_node->add_output("y0");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildAddBody(elem_type, elem_shape);

  for (const Tensor &t : x0) {
    FillValueInfo(t, *graph->add_input());
  }
  for (const Tensor &t : x1) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(stacked, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : x0) {
    ds.inputs.push_back(t);
  }
  for (const Tensor &t : x1) {
    ds.inputs.push_back(t);
  }
  ds.outputs.push_back(std::move(stacked));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, elem_type, elem_shape);
}

// Mirrors upstream ``test_sequence_map_add_1_sequence_1_tensor``: one
// input sequence and one broadcast tensor are mapped through an
// element-wise Add body, yielding a single output sequence ``y0[i] ==
// x0[i] + x1`` (``x1`` is the same tensor for every iteration).
void RegisterSequenceMapAdd1Sequence1TensorCase(const OpsetId &opset,
                                                std::vector<TestCase> &registry) {
  const std::string name = "test_cc_sequence_map_add_1_sequence_1_tensor";
  const kernel::KernelContext ctx{opset};
  const std::vector<int64_t> elem_shape = {4};
  const int32_t elem_type = static_cast<int32_t>(DataType::FLOAT);

  std::vector<Tensor> x0 = {
      Tensor::FromFloat("x0_0", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f}),
      Tensor::FromFloat("x0_1", elem_shape, {-1.0f, -2.0f, -3.0f, -4.0f}),
      Tensor::FromFloat("x0_2", elem_shape, {0.5f, 1.5f, 2.5f, 3.5f}),
  };
  Tensor x1 = Tensor::FromFloat("x1", elem_shape, {100.0f, 200.0f, 300.0f, 400.0f});

  // Compute expected per-iteration outputs (Add with broadcast x1).
  std::vector<Tensor> y_per_iter;
  y_per_iter.reserve(x0.size());
  const float *b = x1.As<float>();
  for (std::size_t i = 0; i < x0.size(); ++i) {
    const float *a = x0[i].As<float>();
    std::vector<float> out(static_cast<std::size_t>(x0[i].element_count()));
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j] = a[j] + b[j];
    }
    y_per_iter.push_back(Tensor::FromFloat("y0_" + std::to_string(i), elem_shape, out));
  }

  std::vector<std::vector<Tensor>> body_outputs_per_iter = {y_per_iter};
  std::vector<Sequence> out_seqs = kernel::SequenceMap(ctx)(
      kernel::SequenceConstruct(ctx).AsSequence(x0), body_outputs_per_iter);
  Tensor stacked =
      kernel::SequenceConstruct(ctx)({out_seqs[0].values.begin(), out_seqs[0].values.end()});
  stacked.name = "y0";

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

  NodeProto *sc0 = graph->add_node();
  sc0->set_op_type("SequenceConstruct");
  for (const Tensor &t : x0) {
    sc0->add_input(t.name);
  }
  sc0->add_output("x0_seq");

  // SequenceMap takes the sequence and the broadcast tensor directly.
  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("x0_seq");
  map_node->add_input("x1");
  map_node->add_output("y0");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildAddBody(elem_type, elem_shape);

  for (const Tensor &t : x0) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(x1, *graph->add_input());
  FillValueInfo(stacked, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : x0) {
    ds.inputs.push_back(t);
  }
  ds.inputs.push_back(x1);
  ds.outputs.push_back(std::move(stacked));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, elem_type, elem_shape);
}

// Mirrors upstream ``test_sequence_map_extract_shapes``: one input
// sequence of FLOAT tensors with varying shapes is mapped through a
// ``Shape`` body, yielding one output sequence of fixed-rank INT64
// shape vectors.
void RegisterSequenceMapExtractShapesCase(const OpsetId &opset, std::vector<TestCase> &registry) {
  const std::string name = "test_cc_sequence_map_extract_shapes";
  const kernel::KernelContext ctx{opset};
  const int32_t in_elem_type = static_cast<int32_t>(DataType::FLOAT);
  const int32_t out_elem_type = static_cast<int32_t>(DataType::INT64);
  const std::vector<int64_t> out_elem_shape = {3};

  // Three per-iteration tensors with distinct shapes (rank fixed at 3).
  const std::vector<std::vector<int64_t>> shapes = {
      {4, 3, 2},
      {2, 5, 1},
      {1, 1, 7},
  };
  std::vector<Tensor> x;
  x.reserve(shapes.size());
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    std::vector<float> values(static_cast<std::size_t>(shapes[i][0] * shapes[i][1] * shapes[i][2]),
                              0.0f);
    x.push_back(Tensor::FromFloat("x_" + std::to_string(i), shapes[i], values));
  }

  // Per-iteration body outputs: shape vectors as INT64[3].
  std::vector<Tensor> y_per_iter;
  y_per_iter.reserve(shapes.size());
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    y_per_iter.push_back(
        Tensor::FromInt64("shape_" + std::to_string(i), out_elem_shape, shapes[i]));
  }

  std::vector<std::vector<Tensor>> body_outputs_per_iter = {y_per_iter};
  std::vector<Sequence> out_seqs =
      kernel::SequenceMap(ctx)(kernel::SequenceConstruct(ctx).AsSequence(x), body_outputs_per_iter);
  Tensor stacked =
      kernel::SequenceConstruct(ctx)({out_seqs[0].values.begin(), out_seqs[0].values.end()});
  stacked.name = "shapes";

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

  NodeProto *sc = graph->add_node();
  sc->set_op_type("SequenceConstruct");
  for (const Tensor &t : x) {
    sc->add_input(t.name);
  }
  sc->add_output("in_seq");

  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("in_seq");
  map_node->add_output("shapes");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  // The body input shape is left rank-3 with all dynamic dims (use the
  // first iteration's shape as a representative; this is only used by
  // upstream shape inference and never executed by the kernel).
  *body_attr->add_g() = BuildShapeBody(in_elem_type, shapes[0], 3);

  for (const Tensor &t : x) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(stacked, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : x) {
    ds.inputs.push_back(t);
  }
  ds.outputs.push_back(std::move(stacked));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, out_elem_type, out_elem_shape);
}

// Mirrors upstream ``test_sequence_map_identity_1_sequence_1_tensor``:
// one input sequence and one broadcast tensor fed through a
// two-input/two-output Identity body, so the SequenceMap node produces
// two output sequences: ``y0[i] == x0[i]`` and ``y1[i] == x1`` (the
// broadcast tensor is the same for every iteration).
void RegisterSequenceMapIdentity1Sequence1TensorCase(const OpsetId &opset,
                                                     std::vector<TestCase> &registry) {
  const std::string name = "test_cc_sequence_map_identity_1_sequence_1_tensor";
  const kernel::KernelContext ctx{opset};
  const std::vector<int64_t> seq_elem_shape = {5};
  const std::vector<int64_t> tensor_shape = {4};
  const int32_t elem_type = static_cast<int32_t>(DataType::FLOAT);

  // Three per-iteration tensors of the input sequence.
  std::vector<Tensor> x0 = {
      Tensor::FromFloat("x0_0", seq_elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}),
      Tensor::FromFloat("x0_1", seq_elem_shape, {-0.5f, -1.0f, -1.5f, -2.0f, -2.5f}),
      Tensor::FromFloat("x0_2", seq_elem_shape, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f}),
  };
  // The broadcast tensor (replicated once per iteration).
  Tensor x1 = Tensor::FromFloat("x1", tensor_shape, {10.0f, 20.0f, 30.0f, 40.0f});

  // Per-iteration body outputs: out0 = identity(x0[i]), out1 = identity(x1).
  std::vector<Tensor> y0_per_iter = x0;
  std::vector<Tensor> y1_per_iter = {x1, x1, x1};
  for (std::size_t i = 0; i < y0_per_iter.size(); ++i) {
    y0_per_iter[i].name = "y0_" + std::to_string(i);
  }
  for (std::size_t i = 0; i < y1_per_iter.size(); ++i) {
    y1_per_iter[i].name = "y1_" + std::to_string(i);
  }

  std::vector<std::vector<Tensor>> body_outputs_per_iter = {y0_per_iter, y1_per_iter};
  std::vector<Sequence> out_seqs = kernel::SequenceMap(ctx)(
      kernel::SequenceConstruct(ctx).AsSequence(x0), body_outputs_per_iter);

  // Materialise both output sequences as stacked tensors.
  Tensor stacked0 =
      kernel::SequenceConstruct(ctx)({out_seqs[0].values.begin(), out_seqs[0].values.end()});
  stacked0.name = "y0";
  Tensor stacked1 =
      kernel::SequenceConstruct(ctx)({out_seqs[1].values.begin(), out_seqs[1].values.end()});
  stacked1.name = "y1";

  TestCase tc(name, name);
  GraphProto *graph = InitSequenceMapModel(tc, name, opset);

  // Node 1: SequenceConstruct the input sequence.
  NodeProto *sc0 = graph->add_node();
  sc0->set_op_type("SequenceConstruct");
  for (const Tensor &t : x0) {
    sc0->add_input(t.name);
  }
  sc0->add_output("x0_seq");

  // Node 2: SequenceMap(x0_seq, x1, body=identity_2) → (y0, y1).
  NodeProto *map_node = graph->add_node();
  map_node->set_op_type("SequenceMap");
  map_node->add_input("x0_seq");
  map_node->add_input("x1");
  map_node->add_output("y0");
  map_node->add_output("y1");
  AttributeProto *body_attr = map_node->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = BuildIdentity2InputsBody(elem_type, seq_elem_shape, tensor_shape);

  // Graph inputs: individual tensors of x0 + the broadcast x1.
  for (const Tensor &t : x0) {
    FillValueInfo(t, *graph->add_input());
  }
  FillValueInfo(x1, *graph->add_input());
  // Graph outputs: y0 and y1 (declared as tensor, promoted below).
  FillValueInfo(stacked0, *graph->add_output());
  FillValueInfo(stacked1, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : x0) {
    ds.inputs.push_back(t);
  }
  ds.inputs.push_back(x1);
  ds.outputs.push_back(std::move(stacked0));
  ds.outputs.push_back(std::move(stacked1));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, elem_type, seq_elem_shape, /*out_index=*/0);
  PromoteOutputToSequenceType(registry, elem_type, tensor_shape, /*out_index=*/1);
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceMap — applies a sub-graph to each element of a tensor sequence
// (since opset 17 in the ai.onnx domain).
//
// Eight cases are registered, mirroring upstream
// ``onnx/backend/test/data/node/test_sequence_map_*``:
//
//   * ``test_cc_sequence_map_identity_float`` / ``..._identity_int64`` —
//     one input sequence mapped through a single-input/-output Identity
//     body (legacy names kept for backward compatibility).
//   * ``test_cc_sequence_map_identity_1_sequence`` — mirrors upstream
//     ``test_sequence_map_identity_1_sequence``: one input sequence,
//     Identity body, one output sequence.
//   * ``test_cc_sequence_map_identity_2_sequences`` — two input
//     sequences mapped through a two-input/two-output Identity body.
//   * ``test_cc_sequence_map_add_2_sequences`` — two input sequences
//     mapped through an element-wise Add body, producing a single
//     output sequence.
//   * ``test_cc_sequence_map_add_1_sequence_1_tensor`` — one input
//     sequence and one broadcast tensor mapped through an Add body.
//   * ``test_cc_sequence_map_identity_1_sequence_1_tensor`` — mirrors
//     upstream ``test_sequence_map_identity_1_sequence_1_tensor``: one
//     input sequence and one broadcast tensor, Identity body, two
//     output sequences.
//   * ``test_cc_sequence_map_extract_shapes`` — one input sequence of
//     FLOAT tensors with varying shapes mapped through a ``Shape``
//     body, producing a sequence of INT64 shape vectors.
//
// The reference kernel does not execute the body subgraph; it merely
// assembles the per-iteration outputs into output sequences, so each
// case ships the expected per-iteration outputs already computed.
// ---------------------------------------------------------------------------
void RegisterSequenceMapCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(17);

  // Case 1: three FLOAT tensors of shape [2, 3] (1 seq, Identity body).
  {
    const std::vector<int64_t> elem_shape = {2, 3};
    Tensor a = Tensor::FromFloat("a", elem_shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor b = Tensor::FromFloat("b", elem_shape, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor c = Tensor::FromFloat("c", elem_shape, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    RegisterSequenceMapIdentityCase("test_cc_sequence_map_identity_float", {a, b, c}, elem_shape,
                                    static_cast<int32_t>(DataType::FLOAT), opset, registry);
  }

  // Case 2: two INT64 tensors of shape [4] (1 seq, Identity body).
  {
    const std::vector<int64_t> elem_shape = {4};
    Tensor a = Tensor::FromInt64("a", elem_shape, {-1, 0, 1, 2});
    Tensor b = Tensor::FromInt64("b", elem_shape, {3, 4, 5, 6});

    RegisterSequenceMapIdentityCase("test_cc_sequence_map_identity_int64", {a, b}, elem_shape,
                                    static_cast<int32_t>(DataType::INT64), opset, registry);
  }

  // Case 3: mirrors upstream test_sequence_map_identity_1_sequence.
  {
    const std::vector<int64_t> elem_shape = {10};
    Tensor a = Tensor::FromFloat("a", elem_shape,
                                 {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor b = Tensor::FromFloat(
        "b", elem_shape, {-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -4.5f, -5.0f});
    Tensor c = Tensor::FromFloat("c", elem_shape,
                                 {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f});

    RegisterSequenceMapIdentityCase("test_cc_sequence_map_identity_1_sequence", {a, b, c},
                                    elem_shape, static_cast<int32_t>(DataType::FLOAT), opset,
                                    registry);
  }

  // Cases 4–7: upstream parity cases.
  RegisterSequenceMapIdentity2SequencesCase(opset, registry);
  RegisterSequenceMapAdd2SequencesCase(opset, registry);
  RegisterSequenceMapAdd1Sequence1TensorCase(opset, registry);
  RegisterSequenceMapIdentity1Sequence1TensorCase(opset, registry);
  RegisterSequenceMapExtractShapesCase(opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
