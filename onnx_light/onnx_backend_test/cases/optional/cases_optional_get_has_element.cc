// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/optional/include_optional_kernels.h"
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

// Adds a ``type`` TYPE_PROTO attribute wrapping
// ``Optional<Tensor<elem_type, shape>>`` onto ``node`` (used by the
// ``Optional`` node we chain in front of ``OptionalGetElement`` /
// ``OptionalHasElement``).
void AddOptionalOfTensorTypeAttr(NodeProto &node, int32_t elem_type,
                                 const std::vector<int64_t> &shape) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Optional *opt = tp->add_optional_type();
  TypeProto *elem = opt->add_elem_type();
  TypeProto::Tensor *tt = elem->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sp = tt->add_shape();
  for (int64_t d : shape) {
    sp->add_dim()->set_dim_value(d);
  }
}

// Builds a single-input model and registers it as a test case named
// ``name``. The model graph contains two nodes:
//
//   1. ``Optional`` that wraps the graph input ``input`` (a tensor) into
//      an ``Optional<Tensor<elem_type, shape>>`` value named
//      ``opt_value``;
//   2. ``op_type`` (one of ``OptionalGetElement`` or
//      ``OptionalHasElement``) that consumes ``opt_value`` and produces
//      ``output``.
//
// ``expected_output`` is materialised as the expected DataSet output.
void RegisterOptionalInputCase(const std::string &name, const std::string &op_type,
                               const Tensor &input, const Tensor &expected_output,
                               const std::vector<int64_t> &input_shape, int32_t input_elem_type,
                               const OpsetId &opset, std::vector<TestCase> &registry) {
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

  // Node 1: Optional input -> opt_value
  NodeProto *opt_node = graph->add_node();
  opt_node->set_op_type("Optional");
  opt_node->add_input("input");
  opt_node->add_output("opt_value");
  AddOptionalOfTensorTypeAttr(*opt_node, input_elem_type, input_shape);

  // Node 2: <op_type> opt_value -> output
  NodeProto *get_node = graph->add_node();
  get_node->set_op_type(op_type);
  get_node->add_input("opt_value");
  get_node->add_output("output");

  // Graph input/output value-infos (both are plain tensors).
  Tensor named_input = input;
  named_input.name = "input";
  FillValueInfo(named_input, *graph->add_input());
  Tensor named_output = expected_output;
  named_output.name = "output";
  FillValueInfo(named_output, *graph->add_output());

  DataSet ds;
  ds.inputs.push_back(named_input);
  ds.outputs.push_back(named_output);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

// Builds a single-node model that exercises the opset-18 tensor-input
// path of ``OptionalGetElement`` / ``OptionalHasElement``. The graph
// contains a single ``op_type`` node whose input is a plain tensor and
// whose output is ``expected_output``.
void RegisterTensorInputCase(const std::string &name, const std::string &op_type,
                             const Tensor &input, const Tensor &expected_output,
                             const OpsetId &opset, std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("input");
  node.add_output("output");
  Expect(node, {input}, {expected_output}, name, {opset}, "backend-test", registry);
}

// Promotes the most recently appended test case so that its single graph
// output is declared as a ``SequenceType<Tensor<elem_type, elem_shape>>``.
// Used by the sequence-output ``OptionalGetElement`` cases below so the
// model's output value-info matches the actual runtime type produced by
// the chained ``SequenceConstruct``.
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

// Builds a model that exercises ``OptionalGetElement`` on a sequence-typed
// input. The graph contains either:
//
//   * Two nodes (``with_optional == false``):
//       1. ``SequenceConstruct`` bundles ``inputs`` into a sequence ``seq``;
//       2. ``OptionalGetElement`` consumes ``seq`` (opset 18 passthrough
//          on a non-optional sequence) and produces ``output_sequence``.
//   * Three nodes (``with_optional == true``):
//       1. ``SequenceConstruct`` bundles ``inputs`` into a sequence ``seq``;
//       2. ``Optional`` wraps ``seq`` into ``Optional<Sequence<Tensor<...>>>``
//          (the wrapped type is declared via the ``type`` TYPE_PROTO
//          attribute);
//       3. ``OptionalGetElement`` consumes the optional sequence and
//          produces ``output_sequence``.
//
// In both flavours the expected output is the constructed sequence
// (``OptionalGetElement`` is a passthrough on the sequence element). It
// is materialised as a stacked tensor so the test harness can compare
// byte buffers, then the graph output value-info is promoted to
// ``SequenceType<Tensor<elem_type, elem_shape>>``.
void RegisterOptionalGetElementSequenceCase(const std::string &name, bool with_optional,
                                            const std::vector<Tensor> &inputs,
                                            const std::vector<int64_t> &elem_shape,
                                            int32_t elem_type, const OpsetId &opset,
                                            std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // The expected sequence equals the constructed sequence (passthrough).
  Tensor stacked = kernel::SequenceConstruct(ctx)(inputs);
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

  // Node 1: SequenceConstruct <inputs…> → seq.
  NodeProto *seq_node = graph->add_node();
  seq_node->set_op_type("SequenceConstruct");
  for (const Tensor &t : inputs) {
    seq_node->add_input(t.name);
  }
  seq_node->add_output("seq");

  std::string get_input = "seq";
  if (with_optional) {
    // Node 2: Optional seq → opt_seq with type attribute
    // ``Optional<Sequence<Tensor<elem_type, elem_shape>>>``.
    NodeProto *opt_node = graph->add_node();
    opt_node->set_op_type("Optional");
    opt_node->add_input("seq");
    opt_node->add_output("opt_seq");
    AttributeProto *attr = opt_node->add_attribute();
    attr->set_name("type");
    attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
    TypeProto *tp = attr->add_tp();
    TypeProto::Optional *opt_type = tp->add_optional_type();
    TypeProto *opt_elem = opt_type->add_elem_type();
    TypeProto::Sequence *opt_seq_type = opt_elem->add_sequence_type();
    TypeProto *opt_seq_elem = opt_seq_type->add_elem_type();
    TypeProto::Tensor *opt_tensor = opt_seq_elem->add_tensor_type();
    opt_tensor->set_elem_type(static_cast<int>(elem_type));
    TensorShapeProto *opt_shape = opt_tensor->add_shape();
    for (int64_t d : elem_shape) {
      opt_shape->add_dim()->set_dim_value(d);
    }
    get_input = "opt_seq";
  }

  // Final node: OptionalGetElement <get_input> → output_sequence.
  NodeProto *get_node = graph->add_node();
  get_node->set_op_type("OptionalGetElement");
  get_node->add_input(get_input);
  get_node->add_output("output_sequence");

  // Graph inputs: the individual tensors fed into SequenceConstruct.
  for (const Tensor &t : inputs) {
    FillValueInfo(t, *graph->add_input());
  }

  // Graph output: ``output_sequence`` declared as tensor (promoted below).
  FillValueInfo(stacked, *graph->add_output());

  DataSet ds;
  for (const Tensor &t : inputs) {
    ds.inputs.push_back(t);
  }
  ds.outputs.push_back(stacked);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));

  PromoteOutputToSequenceType(registry, elem_type, elem_shape);
}

// Builds and registers a single ``OptionalHasElement`` test case for the
// "empty input" scenarios introduced at opset 18 (where the operator may
// be invoked with no input or with an empty-string input slot). The
// resulting model contains a single ``OptionalHasElement`` node whose
// graph has zero inputs and whose only output is the scalar
// ``Tensor<bool, {}>{false}`` produced by the no-input kernel overload.
//
// ``with_empty_input_name`` controls whether the node carries one
// empty-string input slot (``true``, mirroring ONNX's
// ``empty_no_input_name`` flavour) or zero input slots (``false``,
// mirroring ONNX's ``empty_no_input`` and ``empty`` flavours). In both
// cases the model has zero ``graph.input`` entries and the DataSet has
// zero inputs, so the expected output (``false``) is consistent with
// the runtime semantics of an "empty optional" input.
void RegisterOptionalHasElementEmptyCase(const std::string &name, bool with_empty_input_name,
                                         const OpsetId &opset, std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};
  Tensor expected = kernel::OptionalHasElement(ctx)();
  expected.name = "output";

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

  NodeProto *node = graph->add_node();
  node->set_op_type("OptionalHasElement");
  if (with_empty_input_name) {
    node->add_input("");
  }
  node->add_output("output");

  // No graph inputs (the "empty optional" input is not materialised).
  FillValueInfo(expected, *graph->add_output());

  DataSet ds;
  ds.outputs.push_back(expected);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace

// ---------------------------------------------------------------------------
// OptionalGetElement — extracts the element from an optional input (since
// opset 15 in the ai.onnx domain). Since opset 18 the operator also accepts
// non-optional tensor or sequence inputs as a no-op. The cases registered
// here cover:
//
//   * test_cc_optional_get_element_optional_tensor: chains Optional →
//     OptionalGetElement on a FLOAT tensor (opset 15);
//   * test_cc_optional_get_element_tensor: single-node case with a plain
//     FLOAT tensor input (opset 18, passthrough);
//   * test_cc_optional_get_element_sequence: SequenceConstruct →
//     OptionalGetElement on a Sequence<Tensor<INT32, [4]>> (opset 18,
//     passthrough on a non-optional sequence);
//   * test_cc_optional_get_element_optional_sequence: SequenceConstruct →
//     Optional → OptionalGetElement on the same Sequence<Tensor<INT32,
//     [4]>> input (opset 15).
// ---------------------------------------------------------------------------
void RegisterOptionalGetElementCases(std::vector<TestCase> &registry) {
  const std::vector<int64_t> shape = {2, 3};
  Tensor input = Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  // Opset 15: Optional → OptionalGetElement.
  {
    const OpsetId opset = DefaultOpset(15);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalGetElement(ctx)(input);
    RegisterOptionalInputCase("test_cc_optional_get_element_optional_tensor", "OptionalGetElement",
                              input, output, shape, static_cast<int32_t>(DataType::FLOAT), opset,
                              registry);
  }

  // Opset 18: single-node case with a plain tensor input (passthrough).
  {
    const OpsetId opset = DefaultOpset(18);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalGetElement(ctx)(input);
    RegisterTensorInputCase("test_cc_optional_get_element_tensor", "OptionalGetElement", input,
                            output, opset, registry);
  }

  // Sequence-input cases: build an in-graph sequence via SequenceConstruct
  // so the test harness can feed plain tensor inputs.
  const std::vector<int64_t> elem_shape = {4};
  Tensor seq_elem = Tensor::FromInt32("seq_elem", elem_shape, {1, 2, 3, 4});

  // Opset 18: SequenceConstruct → OptionalGetElement (passthrough on a
  // non-optional sequence).
  {
    const OpsetId opset = DefaultOpset(18);
    RegisterOptionalGetElementSequenceCase("test_cc_optional_get_element_sequence",
                                           /*with_optional=*/false, {seq_elem}, elem_shape,
                                           static_cast<int32_t>(DataType::INT32), opset, registry);
  }

  // Opset 15: SequenceConstruct → Optional → OptionalGetElement, exercising
  // the Optional<Sequence<Tensor<INT32, [4]>>> path.
  {
    const OpsetId opset = DefaultOpset(15);
    RegisterOptionalGetElementSequenceCase("test_cc_optional_get_element_optional_sequence",
                                           /*with_optional=*/true, {seq_elem}, elem_shape,
                                           static_cast<int32_t>(DataType::INT32), opset, registry);
  }
}

// ---------------------------------------------------------------------------
// OptionalHasElement — returns a scalar bool indicating whether the input
// optional contains an element (since opset 15 in the ai.onnx domain).
// Since opset 18 the input may also be a non-optional tensor/sequence and
// the input may be omitted entirely. The cases registered here cover:
//
//   * test_cc_optional_has_element_optional_input: chains Optional →
//     OptionalHasElement on a FLOAT tensor (opset 15), expecting true;
//   * test_cc_optional_has_element_tensor_input: single-node case with a
//     plain FLOAT tensor input (opset 18), expecting true;
//   * test_cc_optional_has_element_empty_optional_input,
//     test_cc_optional_has_element_empty_no_input_optional_input,
//     test_cc_optional_has_element_empty_no_input_tensor_input,
//     test_cc_optional_has_element_empty_no_input_name_optional_input,
//     test_cc_optional_has_element_empty_no_input_name_tensor_input:
//     opset-18 "empty input" flavours where the operator is invoked with
//     no input (or with a single empty-string input slot) and returns
//     the scalar ``false``. These mirror the ONNX backend tests of the
//     same suffixes; the project's runtime ``Tensor`` type cannot
//     represent an "empty optional" graph input, so all five cases share
//     the same zero-graph-input model and rely on the no-input
//     ``OptionalHasElement()`` kernel overload to produce ``false``.
// ---------------------------------------------------------------------------
void RegisterOptionalHasElementCases(std::vector<TestCase> &registry) {
  const std::vector<int64_t> shape = {2, 3};
  Tensor input = Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  // Opset 15: Optional → OptionalHasElement, expecting true.
  {
    const OpsetId opset = DefaultOpset(15);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalHasElement(ctx)(input);
    RegisterOptionalInputCase("test_cc_optional_has_element_optional_input", "OptionalHasElement",
                              input, output, shape, static_cast<int32_t>(DataType::FLOAT), opset,
                              registry);
  }

  // Opset 18: single-node case with a plain tensor input, expecting true.
  {
    const OpsetId opset = DefaultOpset(18);
    const kernel::KernelContext ctx{opset};
    Tensor output = kernel::OptionalHasElement(ctx)(input);
    RegisterTensorInputCase("test_cc_optional_has_element_tensor_input", "OptionalHasElement",
                            input, output, opset, registry);
  }

  // Opset 18: "empty input" flavours, all expecting false. The first three
  // (``empty_*`` and ``empty_no_input_*``) model an omitted input via a
  // zero-input ``OptionalHasElement`` node; the two ``empty_no_input_name_*``
  // flavours add a single empty-string input slot to mirror ONNX's
  // ``inputs=[""]`` flavour.
  const OpsetId opset18 = DefaultOpset(18);
  RegisterOptionalHasElementEmptyCase("test_cc_optional_has_element_empty_optional_input",
                                      /*with_empty_input_name=*/false, opset18, registry);
  RegisterOptionalHasElementEmptyCase("test_cc_optional_has_element_empty_no_input_optional_input",
                                      /*with_empty_input_name=*/false, opset18, registry);
  RegisterOptionalHasElementEmptyCase("test_cc_optional_has_element_empty_no_input_tensor_input",
                                      /*with_empty_input_name=*/false, opset18, registry);
  RegisterOptionalHasElementEmptyCase(
      "test_cc_optional_has_element_empty_no_input_name_optional_input",
      /*with_empty_input_name=*/true, opset18, registry);
  RegisterOptionalHasElementEmptyCase(
      "test_cc_optional_has_element_empty_no_input_name_tensor_input",
      /*with_empty_input_name=*/true, opset18, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
